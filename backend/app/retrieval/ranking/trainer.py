"""LTRTrainer: trains a real `lightgbm.LGBMRanker` from labeled
(query, relevant_document_ids) pairs — the exact same labeled-data shape
`HybridRetrievalBenchmark` (app/research/benchmarks/runners.py) already
consumes, itself built from `SyntheticDatasetGenerator`
(app/research/datasets/generator.py) via `Tell me about {tag}` queries
paired with the one document that generated them. Reusing that shape (not
inventing a parallel labeled-dataset format) is what lets this trainer run
against the same synthetic corpus the retrieval benchmark already uses.

Relevance label: 1 if a candidate's document_id is in the query's relevant
set, 0 otherwise — a real, if coarse, binary judgment (not a fabricated
graded score), matching this project's single-document-per-query ground
truth convention documented in docs/benchmarks/README.md.
"""

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ranking.lightgbm_ranker import LightGBMRanker


class LTRTrainer:
    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever

    def fit(self, labeled_queries: list[tuple[str, set[str]]], top_k: int = 8, n_estimators: int = 50) -> LightGBMRanker:
        """Returns `None`-estimator-backed `LightGBMRanker` (falls back to
        RRF) if fewer than 2 labeled queries produce any candidates at all —
        matching this project's established "don't fake a trained model"
        discipline (see M6 SklearnTrainer, M9 GraphNodeClassificationTrainer)
        rather than returning a model fit on degenerate data."""
        import lightgbm as lgb

        X: list[list[float]] = []
        y: list[int] = []
        groups: list[int] = []

        for query, relevant_document_ids in labeled_queries:
            examples = self._retriever.training_examples(query, top_k=top_k)
            if not examples:
                continue
            for document_id, features in examples:
                X.append(features.vector())
                y.append(1 if document_id in relevant_document_ids else 0)
            groups.append(len(examples))

        if len(groups) < 2 or sum(y) == 0:
            return LightGBMRanker(estimator=None)

        estimator = lgb.LGBMRanker(n_estimators=n_estimators, random_state=42, verbosity=-1)
        estimator.fit(X, y, group=groups)
        return LightGBMRanker(estimator=estimator)
