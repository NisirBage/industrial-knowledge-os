"""Timeline Compression (O1.2 Component 6).

`Evidence` carries no explicit timestamp today (the same documented gap O1
Component 1's ranking features noted for "temporal recency"), so a literal
"merge events by date" implementation isn't honestly buildable yet. This
merges *adjacent* graph-sourced evidence entries that share at least one
entity id into a single combined "episode" evidence — a real, structural
proxy (list order + `entity_ids` overlap, both genuine fields) for
"consecutive events about the same equipment," not a fabricated timeline.
Provenance is preserved: the merged entry's `chunk_id` lists every source
chunk id it came from.
"""

from app.retrieval.evidence import Evidence


def compress_timeline(evidence: list[Evidence]) -> list[Evidence]:
    if not evidence:
        return []

    merged: list[Evidence] = []
    buffer: list[Evidence] = []

    def flush() -> None:
        if not buffer:
            return
        merged.append(buffer[0] if len(buffer) == 1 else _merge_episode(buffer))
        buffer.clear()

    for item in evidence:
        if buffer and item.source == "graph" and buffer[-1].source == "graph" and set(item.entity_ids) & set(buffer[-1].entity_ids):
            buffer.append(item)
        else:
            flush()
            buffer.append(item)
    flush()
    return merged


def _merge_episode(items: list[Evidence]) -> Evidence:
    combined_entity_ids = sorted({entity_id for item in items for entity_id in item.entity_ids})
    return Evidence(
        document_id=items[0].document_id,
        page_number=items[0].page_number,
        chunk_id="episode:" + "+".join(item.chunk_id for item in items),
        text=" -> ".join(item.text for item in items),
        entity_ids=combined_entity_ids,
        graph_score=max(item.graph_score for item in items),
        source="graph",
        confidence=round(sum(item.confidence for item in items) / len(items), 4),
        agent="knowledge_graph",
    )
