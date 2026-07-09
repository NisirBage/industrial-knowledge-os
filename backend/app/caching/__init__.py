"""Multi-Level Caching Framework (O1 Component 6).

A single generic `Cache` Protocol plus one real implementation
(`LRUTTLCache`) — reused everywhere a cache is needed (query embeddings,
graph expansion results, reranker scores, retrieval results) rather than a
bespoke cache class per call site. Every cache instance is constructed
explicitly in `main.py` and injected via `app.state` / constructor
parameters — no module-level singletons, no hidden global state.
"""
