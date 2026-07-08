"""Knowledge graph repository: the only place Cypher (or its in-memory
equivalent) is allowed to live. Every other module — ingestion, the
knowledge graph builder, retrieval — talks to this repository's methods
(create_node, create_relation, find_neighbors, ...) and never touches a
query language directly.

Two implementations share one protocol:
  - GraphRepository: real Neo4j, used when a server is reachable.
  - InMemoryGraphRepository: adjacency-list fallback, used when it isn't
    (e.g. no Docker locally). Same method signatures, same semantics, so
    callers never branch on which backend is active.

Nodes are stored under a single :Entity label with a `type` property (an
EntityLabel value) rather than a dynamic Neo4j label per ontology type —
Cypher can't parameterize label names, and encoding the ontology as a
property keeps every query backend-portable and avoids a label-per-type
explosion in the schema.
"""

import logging
import re
import uuid
from typing import Protocol

from neo4j import GraphDatabase

from app.ontology.hierarchy import descendants_of
from app.ontology.labels import EntityLabel, RelationType

logger = logging.getLogger(__name__)

_ID_NAMESPACE = uuid.UUID("2f0b6f2a-4b8b-4e60-9f3f-2f6a5d6a8f10")
_RELATION_TYPE_PATTERN = re.compile(r"^[A-Z_]+$")


def entity_id_for(label: EntityLabel, canonical_name: str) -> str:
    """Deterministic id derived from (label, canonical_name) so repeated
    create_node calls for the same real-world entity are idempotent without
    a lookup round-trip first."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{label.value}:{canonical_name}"))


class GraphRepositoryProtocol(Protocol):
    def close(self) -> None: ...
    def ensure_constraints(self) -> None: ...
    def upsert_document(self, document_id: str, filename: str) -> str: ...
    def create_node(
        self,
        label: EntityLabel,
        canonical_name: str,
        aliases: frozenset[str] = frozenset(),
        confidence: float = 0.5,
        properties: dict | None = None,
    ) -> str: ...
    def create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 0.5,
        chunk_id: str | None = None,
    ) -> None: ...
    def find_entity(self, canonical_name: str, label: EntityLabel | None = None) -> dict | None: ...
    def find_neighbors(self, entity_id: str, relation_type: RelationType | None = None, depth: int = 1) -> list[dict]: ...
    def equipment_history(self, canonical_name: str) -> list[dict]: ...
    def incident_history(self, canonical_name: str) -> list[dict]: ...
    def maintenance_history(self, canonical_name: str) -> list[dict]: ...
    def subgraph(self, entity_ids: list[str], depth: int = 1) -> dict: ...
    def documents_mentioning(self, canonical_name: str) -> list[dict]: ...
    def entities_for_document(self, document_id: str) -> list[dict]: ...
    def distinct_entity_types(self) -> set[str]: ...
    def distinct_relation_types(self) -> set[str]: ...
    def node_count(self) -> int: ...
    def edge_count(self) -> int: ...
    def get_node_properties(self, node_id: str) -> dict: ...
    def all_nodes(self) -> list[dict]: ...
    def all_edges(self) -> list[dict]: ...


def _validate_relation_type(relation_type: RelationType) -> str:
    value = relation_type.value
    if not _RELATION_TYPE_PATTERN.fullmatch(value):
        # RelationType is a closed enum under our control, never user input,
        # but this guard makes that assumption explicit and fails loudly if
        # it's ever violated (e.g. a future enum member with odd characters).
        raise ValueError(f"Unsafe relation type for Cypher interpolation: {value!r}")
    return value


_MAINTENANCE_LABELS = [label.value for label in descendants_of(EntityLabel.MAINTENANCE_RECORD)]
_INCIDENT_LABELS = [label.value for label in descendants_of(EntityLabel.INCIDENT)]
_MAINTENANCE_ACTIVITY_LABELS = [
    label.value
    for label in (
        descendants_of(EntityLabel.INSPECTION)
        + descendants_of(EntityLabel.CORRECTIVE_ACTION)
        + descendants_of(EntityLabel.PREVENTIVE_ACTION)
    )
]


class GraphRepository:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self) -> None:
        self._driver.close()

    def ensure_constraints(self) -> None:
        with self._driver.session() as session:
            session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")

    def upsert_document(self, document_id: str, filename: str) -> str:
        node_id = entity_id_for(EntityLabel.DOCUMENT, document_id)
        with self._driver.session() as session:
            session.run(
                """
                MERGE (d:Entity {id: $id})
                ON CREATE SET d.type = $type, d.canonical_name = $canonical_name, d.aliases = [], d.confidence = 1.0
                SET d.filename = $filename
                """,
                id=node_id,
                type=EntityLabel.DOCUMENT.value,
                canonical_name=document_id,
                filename=filename,
            )
        return node_id

    def create_node(
        self,
        label: EntityLabel,
        canonical_name: str,
        aliases: frozenset[str] = frozenset(),
        confidence: float = 0.5,
        properties: dict | None = None,
    ) -> str:
        node_id = entity_id_for(label, canonical_name)
        alias_list = sorted(aliases)
        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                ON CREATE SET e.type = $type, e.canonical_name = $canonical_name,
                              e.aliases = $aliases, e.confidence = $confidence
                ON MATCH SET e.aliases = e.aliases + [a IN $aliases WHERE NOT a IN e.aliases],
                             e.confidence = CASE WHEN $confidence > e.confidence THEN $confidence ELSE e.confidence END
                SET e += $properties
                """,
                id=node_id,
                type=label.value,
                canonical_name=canonical_name,
                aliases=alias_list,
                confidence=confidence,
                properties=properties or {},
            )
        return node_id

    def get_node_properties(self, node_id: str) -> dict:
        with self._driver.session() as session:
            record = session.run("MATCH (e:Entity {id: $id}) RETURN properties(e) AS props", id=node_id).single()
            return dict(record["props"]) if record else {}

    def create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 0.5,
        chunk_id: str | None = None,
    ) -> None:
        rel_type = _validate_relation_type(relation_type)
        chunk_ids = [chunk_id] if chunk_id else []
        query = f"""
            MATCH (s:Entity {{id: $source_id}}), (t:Entity {{id: $target_id}})
            MERGE (s)-[r:{rel_type}]->(t)
            ON CREATE SET r.confidence = $confidence, r.chunk_ids = $chunk_ids,
                          r.observation_count = 1, r.first_seen = timestamp(), r.last_seen = timestamp()
            ON MATCH SET r.confidence = CASE WHEN $confidence > r.confidence THEN $confidence ELSE r.confidence END,
                        r.chunk_ids = r.chunk_ids + [c IN $chunk_ids WHERE NOT c IN r.chunk_ids],
                        r.observation_count = r.observation_count + 1,
                        r.last_seen = timestamp()
        """
        with self._driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id, confidence=confidence, chunk_ids=chunk_ids)

    def find_entity(self, canonical_name: str, label: EntityLabel | None = None) -> dict | None:
        query = "MATCH (e:Entity {canonical_name: $canonical_name}"
        query += ", type: $type})" if label is not None else "})"
        query += " RETURN e.id AS id, e.type AS type, e.canonical_name AS canonical_name, e.aliases AS aliases LIMIT 1"
        with self._driver.session() as session:
            record = session.run(query, canonical_name=canonical_name, type=label.value if label else None).single()
            return dict(record) if record else None

    def find_neighbors(self, entity_id: str, relation_type: RelationType | None = None, depth: int = 1) -> list[dict]:
        depth = max(1, min(depth, 5))
        rel_filter = f":{_validate_relation_type(relation_type)}" if relation_type is not None else ""
        query = f"""
            MATCH (e:Entity {{id: $entity_id}})-[r{rel_filter}*1..{depth}]-(n:Entity)
            RETURN DISTINCT n.id AS id, n.type AS type, n.canonical_name AS canonical_name,
                            [rel IN r | type(rel)][-1] AS relation_type
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, entity_id=entity_id)]

    def equipment_history(self, canonical_name: str) -> list[dict]:
        return self._history(canonical_name, _MAINTENANCE_LABELS)

    def incident_history(self, canonical_name: str) -> list[dict]:
        return self._history(canonical_name, _INCIDENT_LABELS)

    def maintenance_history(self, canonical_name: str) -> list[dict]:
        return self._history(canonical_name, _MAINTENANCE_ACTIVITY_LABELS)

    def _history(self, canonical_name: str, allowed_types: list[str]) -> list[dict]:
        query = """
            MATCH (e:Entity {canonical_name: $canonical_name})-[r]-(n:Entity)
            WHERE n.type IN $allowed_types
            RETURN DISTINCT n.id AS id, n.type AS type, n.canonical_name AS canonical_name, type(r) AS relation_type
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, canonical_name=canonical_name, allowed_types=allowed_types)]

    def subgraph(self, entity_ids: list[str], depth: int = 1) -> dict:
        # Two simple queries (collect reachable ids, then the induced edges
        # among them) instead of one query flattening variable-length paths —
        # easier to reason about and verify than nested list-of-lists Cypher.
        depth = max(1, min(depth, 3))
        with self._driver.session() as session:
            id_record = session.run(
                f"""
                MATCH (e:Entity) WHERE e.id IN $entity_ids
                OPTIONAL MATCH (e)-[*1..{depth}]-(n:Entity)
                RETURN collect(DISTINCT e.id) + collect(DISTINCT n.id) AS ids
                """,
                entity_ids=entity_ids,
            ).single()
            node_ids = [nid for nid in (id_record["ids"] if id_record else []) if nid is not None]
            if not node_ids:
                return {"nodes": [], "edges": []}

            nodes = [
                dict(record)
                for record in session.run(
                    "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id, e.type AS type, e.canonical_name AS canonical_name",
                    ids=node_ids,
                )
            ]
            edges = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (s:Entity)-[r]->(t:Entity)
                    WHERE s.id IN $ids AND t.id IN $ids
                    RETURN s.id AS source, t.id AS target, type(r) AS relation_type
                    """,
                    ids=node_ids,
                )
            ]
        return {"nodes": nodes, "edges": edges}

    def documents_mentioning(self, canonical_name: str) -> list[dict]:
        query = """
            MATCH (n:Entity {canonical_name: $canonical_name})-[:MENTIONS]-(d:Entity {type: $doc_type})
            RETURN DISTINCT d.id AS document_node_id, d.canonical_name AS document_id, d.filename AS filename
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, canonical_name=canonical_name, doc_type=EntityLabel.DOCUMENT.value)]

    def entities_for_document(self, document_id: str) -> list[dict]:
        document_node_id = entity_id_for(EntityLabel.DOCUMENT, document_id)
        query = """
            MATCH (d:Entity {id: $document_node_id})-[:MENTIONS]->(n:Entity)
            RETURN n.id AS id, n.type AS type, n.canonical_name AS canonical_name
        """
        with self._driver.session() as session:
            return [dict(record) for record in session.run(query, document_node_id=document_node_id)]

    def distinct_entity_types(self) -> set[str]:
        with self._driver.session() as session:
            result = session.run("MATCH (e:Entity) RETURN DISTINCT e.type AS type")
            return {record["type"] for record in result}

    def distinct_relation_types(self) -> set[str]:
        with self._driver.session() as session:
            result = session.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS relation_type")
            return {record["relation_type"] for record in result}

    def node_count(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH (e:Entity) RETURN count(e) AS count").single()["count"]

    def edge_count(self) -> int:
        with self._driver.session() as session:
            return session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]

    def all_nodes(self) -> list[dict]:
        """Full node export for the Graph Dataset Builder (M9) — every other
        query method here is a targeted lookup; this and all_edges() are the
        only two that materialize the entire graph."""
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.id AS id, e.type AS type, e.canonical_name AS canonical_name, "
                "e.confidence AS confidence, properties(e) AS properties"
            )
            return [dict(record) for record in result]

    def all_edges(self) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:Entity)-[r]->(t:Entity) RETURN s.id AS source, t.id AS target, "
                "type(r) AS relation_type, r.confidence AS confidence"
            )
            return [dict(record) for record in result]


class InMemoryGraphRepository:
    """No-persistence fallback used when Neo4j is unreachable. Mirrors
    GraphRepository's semantics with plain Python dicts/adjacency lists so
    the app still starts and the graph still works within a single process.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        # edge key (source_id, target_id, relation_type.value) -> edge dict
        self._edges: dict[tuple[str, str, str], dict] = {}
        self._outgoing: dict[str, set[str]] = {}
        self._incoming: dict[str, set[str]] = {}

    def close(self) -> None:
        pass

    def ensure_constraints(self) -> None:
        pass

    def upsert_document(self, document_id: str, filename: str) -> str:
        node_id = self.create_node(EntityLabel.DOCUMENT, document_id, confidence=1.0)
        self._nodes[node_id]["properties"]["filename"] = filename
        return node_id

    def create_node(
        self,
        label: EntityLabel,
        canonical_name: str,
        aliases: frozenset[str] = frozenset(),
        confidence: float = 0.5,
        properties: dict | None = None,
    ) -> str:
        node_id = entity_id_for(label, canonical_name)
        existing = self._nodes.get(node_id)
        if existing is None:
            self._nodes[node_id] = {
                "id": node_id,
                "type": label,
                "canonical_name": canonical_name,
                "aliases": set(aliases),
                "confidence": confidence,
                "properties": dict(properties or {}),
            }
        else:
            existing["aliases"] |= set(aliases)
            existing["confidence"] = max(existing["confidence"], confidence)
            if properties:
                existing["properties"].update(properties)
        return node_id

    def create_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        confidence: float = 0.5,
        chunk_id: str | None = None,
    ) -> None:
        key = (source_id, target_id, relation_type.value)
        chunk_ids = {chunk_id} if chunk_id else set()
        existing = self._edges.get(key)
        if existing is None:
            self._edges[key] = {
                "source": source_id,
                "target": target_id,
                "relation_type": relation_type,
                "confidence": confidence,
                "chunk_ids": chunk_ids,
                "observation_count": 1,
            }
        else:
            existing["confidence"] = max(existing["confidence"], confidence)
            existing["chunk_ids"] |= chunk_ids
            existing["observation_count"] += 1

        self._outgoing.setdefault(source_id, set()).add(target_id)
        self._incoming.setdefault(target_id, set()).add(source_id)

    def find_entity(self, canonical_name: str, label: EntityLabel | None = None) -> dict | None:
        for node in self._nodes.values():
            if node["canonical_name"] == canonical_name and (label is None or node["type"] == label):
                return self._public_node(node)
        return None

    def find_neighbors(self, entity_id: str, relation_type: RelationType | None = None, depth: int = 1) -> list[dict]:
        depth = max(1, min(depth, 5))
        visited: dict[str, str | None] = {entity_id: None}
        frontier = [entity_id]

        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                for neighbor_id, edge_relation_type in self._adjacent(node_id, relation_type):
                    if neighbor_id in visited:
                        continue
                    visited[neighbor_id] = edge_relation_type
                    next_frontier.append(neighbor_id)
            frontier = next_frontier

        results = []
        for node_id, rel_type in visited.items():
            if node_id == entity_id or node_id not in self._nodes:
                continue
            node = self._nodes[node_id]
            results.append(
                {
                    "id": node_id,
                    "type": node["type"].value,
                    "canonical_name": node["canonical_name"],
                    "relation_type": rel_type.value if rel_type else None,
                }
            )
        return results

    def _adjacent(self, node_id: str, relation_type: RelationType | None) -> list[tuple[str, RelationType]]:
        adjacent: list[tuple[str, RelationType]] = []
        for (source_id, target_id, rel_value), edge in self._edges.items():
            if relation_type is not None and rel_value != relation_type.value:
                continue
            if source_id == node_id:
                adjacent.append((target_id, edge["relation_type"]))
            elif target_id == node_id:
                adjacent.append((source_id, edge["relation_type"]))
        return adjacent

    def equipment_history(self, canonical_name: str) -> list[dict]:
        return self._history(canonical_name, set(descendants_of(EntityLabel.MAINTENANCE_RECORD)))

    def incident_history(self, canonical_name: str) -> list[dict]:
        return self._history(canonical_name, set(descendants_of(EntityLabel.INCIDENT)))

    def maintenance_history(self, canonical_name: str) -> list[dict]:
        allowed = set(
            descendants_of(EntityLabel.INSPECTION)
            + descendants_of(EntityLabel.CORRECTIVE_ACTION)
            + descendants_of(EntityLabel.PREVENTIVE_ACTION)
        )
        return self._history(canonical_name, allowed)

    def _history(self, canonical_name: str, allowed_labels: set[EntityLabel]) -> list[dict]:
        seed = self.find_entity(canonical_name)
        if seed is None:
            return []
        neighbors = self.find_neighbors(seed["id"], depth=1)
        return [n for n in neighbors if EntityLabel(n["type"]) in allowed_labels]

    def subgraph(self, entity_ids: list[str], depth: int = 1) -> dict:
        depth = max(1, min(depth, 3))
        all_node_ids: set[str] = set(entity_ids)
        edges: list[dict] = []
        frontier = list(entity_ids)

        for _ in range(depth):
            next_frontier: list[str] = []
            for node_id in frontier:
                for (source_id, target_id, _rel_value), edge in self._edges.items():
                    if node_id not in (source_id, target_id):
                        continue
                    other = target_id if source_id == node_id else source_id
                    if other not in all_node_ids:
                        next_frontier.append(other)
                    all_node_ids.add(other)
            frontier = next_frontier

        for (source_id, target_id, _rel_value), edge in self._edges.items():
            if source_id in all_node_ids and target_id in all_node_ids:
                edges.append({"source": source_id, "target": target_id, "relation_type": edge["relation_type"].value})

        nodes = [self._public_node(self._nodes[nid]) for nid in all_node_ids if nid in self._nodes]
        return {"nodes": nodes, "edges": edges}

    def documents_mentioning(self, canonical_name: str) -> list[dict]:
        entity = self.find_entity(canonical_name)
        if entity is None:
            return []
        neighbors = self.find_neighbors(entity["id"], relation_type=RelationType.MENTIONS, depth=1)
        return [
            {
                "document_node_id": n["id"],
                "document_id": n["canonical_name"],
                "filename": self._nodes[n["id"]]["properties"].get("filename", ""),
            }
            for n in neighbors
            if EntityLabel(n["type"]) == EntityLabel.DOCUMENT
        ]

    def entities_for_document(self, document_id: str) -> list[dict]:
        document_node_id = entity_id_for(EntityLabel.DOCUMENT, document_id)
        return self.find_neighbors(document_node_id, relation_type=RelationType.MENTIONS, depth=1)

    def distinct_entity_types(self) -> set[str]:
        return {node["type"].value for node in self._nodes.values()}

    def distinct_relation_types(self) -> set[str]:
        return {relation_type for (_source, _target, relation_type) in self._edges}

    def node_count(self) -> int:
        return len(self._nodes)

    def edge_count(self) -> int:
        return len(self._edges)

    def get_node_properties(self, node_id: str) -> dict:
        node = self._nodes.get(node_id)
        return dict(node["properties"]) if node else {}

    def all_nodes(self) -> list[dict]:
        return [
            {
                "id": node["id"],
                "type": node["type"].value,
                "canonical_name": node["canonical_name"],
                "confidence": node["confidence"],
                "properties": dict(node["properties"]),
            }
            for node in self._nodes.values()
        ]

    def all_edges(self) -> list[dict]:
        return [
            {
                "source": edge["source"],
                "target": edge["target"],
                "relation_type": edge["relation_type"].value,
                "confidence": edge["confidence"],
            }
            for edge in self._edges.values()
        ]

    @staticmethod
    def _public_node(node: dict) -> dict:
        return {
            "id": node["id"],
            "type": node["type"].value,
            "canonical_name": node["canonical_name"],
            "aliases": sorted(node["aliases"]),
        }


def build_graph_repository(uri: str, user: str, password: str) -> GraphRepositoryProtocol:
    """Attempt a real Neo4j connection; fall back to an in-memory graph if the
    server is unreachable, so the app still starts (e.g. no Docker locally)."""
    try:
        repository = GraphRepository(uri, user, password)
        repository._driver.verify_connectivity()
        return repository
    except Exception:
        logger.warning(
            "Neo4j unreachable at %s — falling back to InMemoryGraphRepository. "
            "Knowledge graph will not persist across restarts until Neo4j is available.",
            uri,
        )
        return InMemoryGraphRepository()
