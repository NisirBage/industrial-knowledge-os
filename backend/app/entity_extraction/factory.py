"""Builds the (primary, supplement) extractor pair used by EntityExtractionService.

GLiNER is primary when it loads successfully; regex always runs too, since it
catches structured codes (tag numbers, standard names) precisely regardless of
how good the NER model is. If GLiNER can't be loaded (no network/cache), regex
becomes the sole extractor — the pipeline degrades gracefully, it never fails
to start.
"""

import logging

from app.entity_extraction.base import EntityExtractor
from app.entity_extraction.regex_extractor import RegexEntityExtractor

logger = logging.getLogger(__name__)


def build_entity_extractors(gliner_model: str, gliner_threshold: float) -> tuple[EntityExtractor, EntityExtractor | None]:
    regex_extractor = RegexEntityExtractor()
    try:
        from app.entity_extraction.gliner_extractor import GlinerEntityExtractor

        gliner_extractor = GlinerEntityExtractor(gliner_model, threshold=gliner_threshold)
        return gliner_extractor, regex_extractor
    except Exception:
        logger.warning(
            "GLiNER model '%s' unavailable — falling back to regex-only entity extraction.",
            gliner_model,
        )
        return regex_extractor, None
