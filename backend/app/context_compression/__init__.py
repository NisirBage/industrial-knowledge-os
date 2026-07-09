"""Context Compression Engine (O1.2 Components 3-7).

Retrieved Evidence -> exact-duplicate removal (hash) -> near-duplicate
removal (embedding + text-overlap + shared-entity signals) -> semantic
clustering (real agglomerative clustering over embeddings) -> same-entity
graph-history merge -> CompressedEvidenceBundle, with real, computed
statistics (not estimated/fabricated) for explainability.

Standalone and additive: nothing in app/planning, app/reasoning,
app/recommendation, or app/prediction is touched or required to use this —
per this milestone's explicit "do not modify Planner/Reasoning
Graph/Recommendation Engine/Prediction Engine" constraint, compression lives
entirely as an optional post-processing step over a `list[Evidence]`.
"""
