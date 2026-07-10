import logging

from app.nli.base import NLIDetector
from app.nli.rule_based import RuleBasedNLIDetector

logger = logging.getLogger(__name__)


def build_nli_detector(model_name: str) -> NLIDetector:
    try:
        from app.nli.transformer_nli import TransformerNLIDetector

        return TransformerNLIDetector(model_name)
    except Exception:
        logger.warning(
            "NLI model '%s' unavailable — falling back to RuleBasedNLIDetector "
            "(antonym-vocabulary contradiction detection, no real entailment/neutral distinction).",
            model_name,
        )
        return RuleBasedNLIDetector()
