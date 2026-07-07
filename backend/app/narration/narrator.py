"""LLM Narrator: the final stage of the pipeline, and now the LLM's only
job in the whole system. It writes the user-facing answer from the
*structured* Reasoning Graph summary and Recommendation Bundle — not from a
raw evidence dump — so the LLM narrates conclusions the reasoning stages
already reached rather than re-deriving them itself. This is the concrete
realization of "the LLM becomes the final narrator, not the reasoning
engine."
"""

from app.reasoning.graph_models import ReasoningGraph, ReasoningNodeType
from app.recommendation.models import RecommendationBundle
from app.services.llm_provider import LLMProvider

_SYSTEM_PROMPT = (
    "You are the narration layer of an industrial knowledge intelligence system. "
    "Reasoning has already happened — you are given its structured conclusions "
    "(a reasoning summary and a list of recommendations), not raw documents. "
    "Write a clear, concise final answer to the user's question using ONLY the "
    "information below. Do not invent facts, equipment, or recommendations beyond "
    "what is given. If the reasoning summary indicates insufficient evidence, say so."
)


class LLMNarrator:
    def __init__(self, llm_provider: LLMProvider):
        self._llm_provider = llm_provider

    def narrate(self, query: str, reasoning_graph: ReasoningGraph, recommendation_bundle: RecommendationBundle) -> str:
        key_facts = reasoning_graph.nodes_of_type(ReasoningNodeType.FACT)[:5]
        risks = reasoning_graph.nodes_of_type(ReasoningNodeType.RISK)

        lines = [f"Query: {query}", "", f"Reasoning summary: {reasoning_graph.summary}"]

        if key_facts:
            lines.append("\nKey facts:")
            lines += [f"- {fact.description}" for fact in key_facts]

        if risks:
            lines.append("\nIdentified risks:")
            lines += [f"- {risk.description}" for risk in risks]

        if recommendation_bundle.recommendations:
            lines.append("\nRecommendations:")
            lines += [
                f"- [{r.priority.value}/{r.severity.value}] {r.description} — {r.justification}"
                for r in recommendation_bundle.recommendations
            ]
        else:
            lines.append("\nNo recommendations were generated.")

        user_prompt = "\n".join(lines)
        return self._llm_provider.complete(_SYSTEM_PROMPT, user_prompt)
