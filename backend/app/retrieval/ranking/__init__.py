"""Learning-to-Rank Retrieval (O1 Component 1).

Replaces the hardcoded "sort by cross-encoder score" final step in
`HybridRetriever.retrieve()` with a pluggable `Ranker` strategy — Reciprocal
Rank Fusion (a real, formula-based ranker with no tunable weights) and a
trainable LightGBM LambdaMART ranker both implement the same `Ranker`
Protocol, injected exactly like `Reranker` already is. Omitting the injected
ranker entirely (`None`, the default) preserves the exact prior behavior
(sort by cross-encoder score), so no existing caller is affected.
"""
