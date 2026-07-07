"""Reasoning agent: detects contradictions, builds the per-query Evidence
Graph, and builds the Industrial Reasoning Graph from it — the structural
reasoning step that runs before any LLM call."""

import dataclasses

from app.contradiction.detector import ContradictionDetector
from app.evidence.graph import EvidenceGraphBuilder
from app.orchestrator.state import AgentState
from app.reasoning.reasoning_graph_builder import ReasoningGraphBuilder


class ReasoningAgent:
    name = "reasoning"

    def __init__(
        self,
        contradiction_detector: ContradictionDetector,
        evidence_graph_builder: EvidenceGraphBuilder,
        reasoning_graph_builder: ReasoningGraphBuilder,
    ):
        self._contradiction_detector = contradiction_detector
        self._evidence_graph_builder = evidence_graph_builder
        self._reasoning_graph_builder = reasoning_graph_builder

    def run(self, state: AgentState) -> AgentState:
        bundle = state["evidence_bundle"]
        contradictions = self._contradiction_detector.detect(bundle)
        bundle = dataclasses.replace(bundle, contradictions=contradictions)

        evidence_graph = self._evidence_graph_builder.build(state["query"], bundle)
        reasoning_graph = self._reasoning_graph_builder.build(bundle, contradictions)

        return {
            **state,
            "evidence_bundle": bundle,
            "evidence_graph": evidence_graph,
            "reasoning_graph": reasoning_graph,
            "contradictions": contradictions,
        }
