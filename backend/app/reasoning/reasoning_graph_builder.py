"""Builds the Industrial Reasoning Graph from an EvidenceBundle and any
detected Contradictions — deterministic, structural construction (no LLM
call). Every text evidence item becomes a FACT node; graph-evidence
relation triples tagged with a causal relation type (FAILED_DUE_TO, CAUSES)
additionally produce CAUSE/EFFECT nodes and a RISK node, since a causal
failure relationship is itself risk-bearing. Contradictions become
HYPOTHESIS nodes (unresolved, low-confidence pending clarification) and
empty task results become MISSING_EVIDENCE nodes.

The Recommendation Engine (app/recommendation/engine.py) adds RECOMMENDATION
nodes to the graph this builder returns in a second pass, once
recommendations exist — so the graph ends up with the full chain the M3
spec's worked example describes (fact/cause/effect/risk -> recommendation).
"""

import re

from app.contradiction.models import Contradiction
from app.evidence.models import EvidenceBundle
from app.reasoning.graph_models import ReasoningEdge, ReasoningGraph, ReasoningNode, ReasoningNodeType

_TRIPLE_PATTERN = re.compile(r"^(.*?)\s+([A-Z_]+)\s+(.*)$")
_CAUSAL_RELATIONS = {"FAILED_DUE_TO", "CAUSES"}

_UNKNOWN_CONFIDENCE_THRESHOLD = 0.2


class ReasoningGraphBuilder:
    def build(self, bundle: EvidenceBundle, contradictions: list[Contradiction]) -> ReasoningGraph:
        graph = ReasoningGraph(nodes=[], edges=[], summary="")

        for item in bundle.evidence:
            graph.add_node(
                ReasoningNode(
                    id=f"fact:{item.chunk_id}",
                    node_type=ReasoningNodeType.FACT,
                    description=item.text[:200],
                    confidence=item.confidence,
                    supporting_evidence_ids=[item.chunk_id],
                )
            )

        for item in bundle.graph_evidence:
            graph.add_node(
                ReasoningNode(
                    id=f"fact:{item.chunk_id}",
                    node_type=ReasoningNodeType.FACT,
                    description=item.text,
                    confidence=item.confidence,
                    supporting_evidence_ids=[item.chunk_id],
                )
            )
            self._add_causal_structure(graph, item.chunk_id, item.text, item.confidence)

        for index, contradiction in enumerate(contradictions):
            graph.add_node(
                ReasoningNode(
                    id=f"hypothesis:{index}",
                    node_type=ReasoningNodeType.HYPOTHESIS,
                    description=contradiction.recommended_clarification,
                    confidence=round(1 - contradiction.confidence, 2),
                    supporting_evidence_ids=list(contradiction.conflicting_evidence),
                )
            )

        for index, description in enumerate(bundle.missing_information):
            graph.add_node(
                ReasoningNode(
                    id=f"missing:{index}",
                    node_type=ReasoningNodeType.MISSING_EVIDENCE,
                    description=description,
                    confidence=0.0,
                )
            )

        if not graph.nodes or bundle.confidence < _UNKNOWN_CONFIDENCE_THRESHOLD:
            graph.add_node(
                ReasoningNode(
                    id="unknown:0",
                    node_type=ReasoningNodeType.UNKNOWN,
                    description="Insufficient evidence to determine an answer confidently.",
                    confidence=0.0,
                )
            )

        graph.summary = self._generate_summary(graph, contradictions)
        return graph

    def _add_causal_structure(self, graph: ReasoningGraph, chunk_id: str, text: str, confidence: float) -> None:
        match = _TRIPLE_PATTERN.match(text)
        if match is None:
            return
        source, relation, target = match.group(1), match.group(2), match.group(3)
        if relation not in _CAUSAL_RELATIONS:
            return

        cause_id = f"cause:{chunk_id}"
        effect_id = f"effect:{chunk_id}"
        risk_id = f"risk:{chunk_id}"

        graph.add_node(ReasoningNode(id=cause_id, node_type=ReasoningNodeType.CAUSE, description=target, confidence=confidence, supporting_evidence_ids=[chunk_id]))
        graph.add_node(ReasoningNode(id=effect_id, node_type=ReasoningNodeType.EFFECT, description=source, confidence=confidence, supporting_evidence_ids=[chunk_id]))
        graph.add_edge(ReasoningEdge(source_id=cause_id, target_id=effect_id, relationship="causes"))

        graph.add_node(
            ReasoningNode(
                id=risk_id,
                node_type=ReasoningNodeType.RISK,
                description=f"Recurring risk: {source} {relation} {target}",
                confidence=confidence,
                supporting_evidence_ids=[chunk_id],
            )
        )
        graph.add_edge(ReasoningEdge(source_id=effect_id, target_id=risk_id, relationship="indicates_risk"))

    @staticmethod
    def _generate_summary(graph: ReasoningGraph, contradictions: list[Contradiction]) -> str:
        facts = graph.nodes_of_type(ReasoningNodeType.FACT)
        causes = graph.nodes_of_type(ReasoningNodeType.CAUSE)
        risks = graph.nodes_of_type(ReasoningNodeType.RISK)
        missing = graph.nodes_of_type(ReasoningNodeType.MISSING_EVIDENCE)
        unknowns = graph.nodes_of_type(ReasoningNodeType.UNKNOWN)

        parts = [f"Analyzed {len(facts)} piece(s) of evidence."]
        if causes:
            parts.append(f"Identified {len(causes)} cause-effect relationship(s) from the knowledge graph.")
        if risks:
            parts.append(f"{len(risks)} recurring risk pattern(s) detected.")
        if contradictions:
            parts.append(f"{len(contradictions)} contradiction(s) found in the evidence.")
        if missing:
            parts.append(f"{len(missing)} gap(s) in available information.")
        if unknowns:
            parts.append("Confidence is too low to draw a firm conclusion.")
        return " ".join(parts)
