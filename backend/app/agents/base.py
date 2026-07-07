from typing import Protocol

from app.orchestrator.state import AgentState


class Agent(Protocol):
    name: str

    def run(self, state: AgentState) -> AgentState: ...
