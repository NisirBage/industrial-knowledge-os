"""Intelligent Entity Resolution (O1.2 Component 1).

Today's canonicalization (`app/entity_extraction/normalizer.py`) is pure
regex + uppercasing — real, but narrow: "Pump101" (no hyphen) or "Feedwater
Pump" (a descriptive alias with no embedded tag) don't resolve to the same
canonical entity as "P-101" unless they happen to match the embedded-tag
regex. `EntityResolver` adds real multi-signal matching (edit distance,
embedding similarity, ontology similarity, graph-neighborhood similarity,
acronym expansion) on top of — not instead of — that existing step, and is
entirely additive: nothing in `app/entity_extraction/` is required to use it.
"""
