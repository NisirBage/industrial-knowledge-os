"""Evidence Fusion Engine (M3 Component 2): merges task-graph evidence and
knowledge-graph expansion evidence into a single EvidenceBundle — dedup by
chunk id, rank by confidence, build a best-effort chronological timeline,
surface missing information from empty task results, and aggregate an
overall confidence score. Contradictions are attached afterward (a separate
pass, see app/contradiction) since detecting them requires the fused bundle
as input.
"""

import re

from app.evidence.models import EvidenceBundle, MetadataEvidence, TimelineEvent
from app.planning.models import TaskGraph
from app.retrieval.evidence import Evidence

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
_DATE_PATTERN = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)

_TOP_CONFIDENCE_WEIGHT = 0.6
_COVERAGE_WEIGHT = 0.3
_GRAPH_EVIDENCE_BONUS = 0.1
_MISSING_INFO_PENALTY_PER_ITEM = 0.05
_MISSING_INFO_PENALTY_CAP = 0.3


class EvidenceFusionEngine:
    def fuse(self, task_graph: TaskGraph, graph_expansion_evidence: list[Evidence]) -> EvidenceBundle:
        combined = task_graph.flattened_evidence() + graph_expansion_evidence
        text_evidence = self._deduplicate([e for e in combined if e.source != "graph"])
        graph_evidence = self._deduplicate([e for e in combined if e.source == "graph"])

        text_evidence.sort(key=lambda e: e.confidence, reverse=True)
        graph_evidence.sort(key=lambda e: e.confidence, reverse=True)

        timeline = self._build_timeline(text_evidence)
        metadata = self._extract_metadata(text_evidence)
        missing_information = self._missing_information(task_graph)
        confidence = self._aggregate_confidence(text_evidence, graph_evidence, missing_information)

        return EvidenceBundle(
            evidence=text_evidence,
            graph_evidence=graph_evidence,
            timeline=timeline,
            metadata=metadata,
            missing_information=missing_information,
            confidence=confidence,
        )

    @staticmethod
    def _deduplicate(evidence: list[Evidence]) -> list[Evidence]:
        best_by_chunk: dict[str, Evidence] = {}
        for item in evidence:
            existing = best_by_chunk.get(item.chunk_id)
            if existing is None or item.confidence > existing.confidence:
                best_by_chunk[item.chunk_id] = item
        return list(best_by_chunk.values())

    @staticmethod
    def _build_timeline(text_evidence: list[Evidence]) -> list[TimelineEvent]:
        dated: list[tuple[tuple[int, int], TimelineEvent]] = []
        undated: list[TimelineEvent] = []

        for item in text_evidence:
            match = _DATE_PATTERN.search(item.text)
            event = TimelineEvent(
                description=item.text[:200],
                timestamp=match.group(0) if match else None,
                source_chunk_id=item.chunk_id,
                confidence=item.confidence,
            )
            if match:
                year = int(match.group(2))
                month = _MONTHS[match.group(1).lower()]
                dated.append(((year, month), event))
            else:
                undated.append(event)

        dated.sort(key=lambda pair: pair[0])
        return [event for _sort_key, event in dated] + undated

    @staticmethod
    def _extract_metadata(text_evidence: list[Evidence]) -> list[MetadataEvidence]:
        seen_documents: set[str] = set()
        metadata: list[MetadataEvidence] = []
        for item in text_evidence:
            if item.document_id and item.document_id not in seen_documents:
                seen_documents.add(item.document_id)
                metadata.append(MetadataEvidence(key="source_document", value=item.document_id, source_chunk_id=item.chunk_id))
        return metadata

    @staticmethod
    def _missing_information(task_graph: TaskGraph) -> list[str]:
        return [node.description for node in task_graph.nodes if not node.result]

    @staticmethod
    def _aggregate_confidence(text_evidence: list[Evidence], graph_evidence: list[Evidence], missing_information: list[str]) -> float:
        if not text_evidence and not graph_evidence:
            return 0.0
        top_confidence = max((e.confidence for e in text_evidence), default=0.0)
        coverage = min(len(text_evidence) / 4, 1.0)
        graph_bonus = _GRAPH_EVIDENCE_BONUS if graph_evidence else 0.0
        missing_penalty = min(len(missing_information) * _MISSING_INFO_PENALTY_PER_ITEM, _MISSING_INFO_PENALTY_CAP)
        raw = top_confidence * _TOP_CONFIDENCE_WEIGHT + coverage * _COVERAGE_WEIGHT + graph_bonus - missing_penalty
        return round(max(0.0, min(1.0, raw)), 2)
