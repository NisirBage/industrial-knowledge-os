"""EntityResolver: combines every real similarity signal in `similarity.py`
into a single merge decision. Confidence is a plain average of whichever
signals are actually available for a given (candidate, known-entity) pair —
a documented, explainable combination, not a black box (O1's Component 5
explainability standard applies here too, even outside the ranker).
"""

from dataclasses import dataclass

from app.embeddings.embedder import Embedder
from app.entity_resolution.models import ResolvedEntity, SimilaritySignal
from app.entity_resolution.similarity import (
    cosine_similarity,
    edit_distance_similarity,
    expand_acronym,
    neighborhood_similarity,
    ontology_similarity,
)
from app.ontology.labels import EntityLabel

_DEFAULT_MERGE_THRESHOLD = 0.82


@dataclass(frozen=True)
class KnownEntity:
    node_id: str
    canonical_name: str
    label: EntityLabel
    neighbor_ids: frozenset[str] = frozenset()
    embedding: list[float] | None = None


class EntityResolver:
    def __init__(self, merge_threshold: float = _DEFAULT_MERGE_THRESHOLD, embedder: Embedder | None = None):
        self._merge_threshold = merge_threshold
        self._embedder = embedder

    def resolve(
        self,
        candidate_name: str,
        candidate_label: EntityLabel,
        known_entities: list[KnownEntity],
        candidate_neighbor_ids: frozenset[str] = frozenset(),
    ) -> ResolvedEntity | None:
        """Returns the best-matching known entity as a `ResolvedEntity` if
        its combined confidence clears `merge_threshold`, else `None` —
        meaning the caller should treat `candidate_name` as a genuinely new
        canonical entity, not force a match onto the nearest (but not close
        enough) existing one."""
        best: tuple[float, KnownEntity, list[SimilaritySignal]] | None = None
        for known in known_entities:
            signals = self._score(candidate_name, candidate_label, candidate_neighbor_ids, known)
            combined = self._combine(signals)
            if best is None or combined > best[0]:
                best = (combined, known, signals)

        if best is None or best[0] < self._merge_threshold:
            return None

        combined, known, signals = best
        return ResolvedEntity(
            canonical_id=known.node_id,
            canonical_name=known.canonical_name,
            aliases=frozenset({candidate_name}),
            confidence=combined,
            merge_reason=self._merge_reason(signals),
            signals=tuple(signals),
        )

    def _score(
        self, candidate_name: str, candidate_label: EntityLabel, candidate_neighbor_ids: frozenset[str], known: KnownEntity
    ) -> list[SimilaritySignal]:
        signals = [
            SimilaritySignal("edit_distance", edit_distance_similarity(candidate_name, known.canonical_name)),
            SimilaritySignal("ontology", ontology_similarity(candidate_label, known.label)),
        ]

        if candidate_neighbor_ids or known.neighbor_ids:
            signals.append(SimilaritySignal("neighborhood", neighborhood_similarity(candidate_neighbor_ids, known.neighbor_ids)))

        if self._embedder is not None and known.embedding is not None:
            candidate_vector = self._embedder.embed_texts([candidate_name])[0]
            signals.append(SimilaritySignal("embedding", cosine_similarity(candidate_vector, known.embedding)))

        expanded_prefix = expand_acronym(candidate_name)
        if expanded_prefix is not None and known.canonical_name.upper().startswith(expanded_prefix):
            signals.append(SimilaritySignal("acronym_expansion", 1.0))

        return signals

    @staticmethod
    def _combine(signals: list[SimilaritySignal]) -> float:
        if not signals:
            return 0.0
        return round(sum(s.score for s in signals) / len(signals), 4)

    @staticmethod
    def _merge_reason(signals: list[SimilaritySignal]) -> str:
        best_signal = max(signals, key=lambda s: s.score)
        return f"{best_signal.name}={best_signal.score:.2f}"
