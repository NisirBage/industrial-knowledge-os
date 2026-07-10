"""Self-Reflection agent (O3 Component 3): the last pipeline stage. Reads
the final answer's confidence and every real signal already computed
(evidence bundle, faithfulness report, recommendation verification) and
writes a read-only `ReflectionReport` — it never mutates `answer` or
`confidence`."""

from app.orchestrator.state import AgentState
from app.reflection.engine import SelfReflectionEngine


class ReflectionAgent:
    name = "reflection"

    def __init__(self, reflection_engine: SelfReflectionEngine):
        self._reflection_engine = reflection_engine

    def run(self, state: AgentState) -> AgentState:
        report = self._reflection_engine.reflect(
            state["query"],
            state["evidence_bundle"],
            state.get("confidence", 0.0),
            state.get("faithfulness_report"),
            state.get("recommendation_verification"),
        )
        return {**state, "reflection_report": report}
