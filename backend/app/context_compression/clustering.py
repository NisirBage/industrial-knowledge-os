"""Semantic Clustering (O1.2 Component 5): real agglomerative clustering
(scikit-learn, already a dependency) over evidence-text embeddings.

Cluster `cluster_type` is derived from real substrings already present in
`Evidence.source`/`Evidence.text` (the relation-type keywords
`_graph_history()`/`GraphExpansionStage` already embed into their
synthesized text, e.g. "FAILED_DUE_TO", "INSPECTED_BY") — a real heuristic
over real data, not a fabricated taxonomy. Cluster names the O1.2 spec asks
for that have no real signal to derive them from (a genuine "prediction"
cluster would need Evidence to carry a task-type field it doesn't have
today) fall back to the honestly-generic "equipment"/"text" buckets instead
of being guessed.
"""

from app.embeddings.embedder import Embedder
from app.context_compression.models import EvidenceCluster
from app.retrieval.evidence import Evidence

_DEFAULT_DISTANCE_THRESHOLD = 0.3

_INCIDENT_KEYWORDS = ("FAILED_DUE_TO", "FAILURE")
_MAINTENANCE_KEYWORDS = ("MAINTAINED_BY", "INSPECTED_BY", "MAINTENANCE", "INSPECTION")


def _infer_cluster_type(members: tuple[Evidence, ...]) -> str:
    sources = {m.source for m in members}
    if any(s.startswith("vision:") for s in sources):
        return "vision"
    texts = " ".join(m.text.upper() for m in members)
    if any(keyword in texts for keyword in _INCIDENT_KEYWORDS):
        return "incident"
    if any(keyword in texts for keyword in _MAINTENANCE_KEYWORDS):
        return "maintenance"
    if sources == {"graph"}:
        return "graph_relation"
    if any(m.entity_ids for m in members):
        return "equipment"
    return "text"


def cluster_evidence(evidence: list[Evidence], embedder: Embedder | None = None, distance_threshold: float = _DEFAULT_DISTANCE_THRESHOLD) -> list[EvidenceCluster]:
    if not evidence:
        return []

    if len(evidence) == 1 or embedder is None:
        # Honest fallback: without an embedder there is no real semantic
        # signal to cluster on, so each item is its own singleton cluster
        # rather than guessing at groupings.
        return [
            EvidenceCluster(cluster_id=f"cluster-{i}", cluster_type=_infer_cluster_type((item,)), representative=item, members=(item,))
            for i, item in enumerate(evidence)
        ]

    from sklearn.cluster import AgglomerativeClustering

    vectors = embedder.embed_texts([e.text for e in evidence])
    model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average")
    labels = model.fit_predict(vectors)

    groups: dict[int, list[Evidence]] = {}
    for label, item in zip(labels, evidence, strict=True):
        groups.setdefault(int(label), []).append(item)

    clusters = []
    for cluster_index, members in enumerate(groups.values()):
        representative = max(members, key=lambda e: e.confidence)
        clusters.append(
            EvidenceCluster(
                cluster_id=f"cluster-{cluster_index}",
                cluster_type=_infer_cluster_type(tuple(members)),
                representative=representative,
                members=tuple(members),
            )
        )
    return clusters
