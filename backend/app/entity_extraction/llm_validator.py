"""Optional LLM validation pass over candidate entities.

Off by default (ENABLE_LLM_ENTITY_VALIDATION=false) since GLiNER+regex is
already precise enough for the MVP and every LLM call adds latency/cost.
When enabled, the LLM is asked to confirm which candidates are genuine
entities in context; parsing is defensive — any malformed/unparseable
response fails open (returns the original candidates unchanged) rather than
dropping extraction results because of an LLM hiccup.
"""

import json
import logging

from app.documents.chunking import Chunk
from app.entity_extraction.schemas import ExtractedEntity
from app.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You validate candidate named entities extracted from an industrial document "
    "chunk. Given the chunk text and a JSON list of candidate entity names, return "
    "a JSON array of the candidate names that are genuinely valid entities in "
    "context (drop false positives). Respond with ONLY a JSON array of strings, "
    "nothing else."
)


class LlmEntityValidator:
    def __init__(self, llm_provider: LLMProvider, enabled: bool):
        self._llm_provider = llm_provider
        self._enabled = enabled

    def validate(self, entities: list[ExtractedEntity], chunk: Chunk) -> list[ExtractedEntity]:
        if not self._enabled or not entities:
            return entities

        candidate_names = [e.canonical_name for e in entities]
        user_prompt = f"Chunk:\n{chunk.text}\n\nCandidates:\n{json.dumps(candidate_names)}"

        try:
            response = self._llm_provider.complete(_SYSTEM_PROMPT, user_prompt)
            confirmed = set(json.loads(response))
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("LLM entity validation response unparseable (%s) — keeping all candidates.", exc)
            return entities

        return [e for e in entities if e.canonical_name in confirmed]
