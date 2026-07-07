"""Builds the active RelationExtractor.

Single seam to swap in an LLM-assisted or transformer-based extractor later
without touching the knowledge graph builder or ingestion service.
"""

from app.relation_extraction.base import RelationExtractor
from app.relation_extraction.rule_based import RuleBasedRelationExtractor


def build_relation_extractor() -> RelationExtractor:
    return RuleBasedRelationExtractor()
