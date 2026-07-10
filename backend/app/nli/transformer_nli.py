"""Real transformer NLI (ModernBERT/DeBERTa-MNLI-class models) via
sentence-transformers' CrossEncoder — the exact same library and loading
idiom `CrossEncoderReranker` already uses (app/retrieval/reranker.py), so
this stays consistent with the one heavy-model-loading pattern already
established in this codebase rather than introducing a second one.

Default model (`cross-encoder/nli-deberta-v3-base`) is a DeBERTa-v3 model
fine-tuned on SNLI+MultiNLI, satisfying the spec's "DeBERTa MNLI" option
directly; any other sentence-transformers NLI cross-encoder (including a
ModernBERT-based one, once published) is a drop-in via NLI_MODEL config.
"""

from app.nli.models import NLILabel, NLIResult

_LABEL_ALIASES = {
    "entailment": NLILabel.ENTAILMENT,
    "neutral": NLILabel.NEUTRAL,
    "contradiction": NLILabel.CONTRADICTION,
}


class TransformerNLIDetector:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self._id2label = self._resolve_id2label()

    def classify(self, premise: str, hypothesis: str) -> NLIResult:
        scores = self._model.predict([(premise, hypothesis)], apply_softmax=True)[0]
        best_index = int(scores.argmax()) if hasattr(scores, "argmax") else max(range(len(scores)), key=lambda i: scores[i])
        label = self._label_for(best_index)
        return NLIResult(premise, hypothesis, label, float(scores[best_index]))

    def _resolve_id2label(self) -> dict[int, str]:
        try:
            return dict(self._model.model.config.id2label)
        except Exception:
            # Standard label order for sentence-transformers NLI cross-encoders.
            return {0: "contradiction", 1: "entailment", 2: "neutral"}

    def _label_for(self, index: int) -> NLILabel:
        raw = str(self._id2label.get(index, "")).lower()
        for key, label in _LABEL_ALIASES.items():
            if key in raw:
                return label
        return NLILabel.NEUTRAL
