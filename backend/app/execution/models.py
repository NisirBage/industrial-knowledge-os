"""Unified Execution Trace & Failure Intelligence Core (O3.3 Component 1/2).

Before this milestone, planner/retrieval/ranking/verification/recommendation/
calibration/reflection/benchmark metrics were each represented independently
by their own module. `ExecutionTrace` is the single canonical representation
of one complete reasoning session: every future analysis (Pattern Mining,
Root Cause Graph, Counterfactuals, Optimization Recommendations, Publication
Pipeline) must be built on top of it rather than collecting telemetry itself.

`ExecutionEvent` is immutable (frozen dataclass) — pipeline stages append
events, they never mutate one. Per-subsystem trace slots (`planner_trace`,
`retrieval_trace`, ...) are plain `dict | None`: subsystems that are not
invoked inside the live chat pipeline for a given query (Compression,
Vision, Prediction, Graph Intelligence — the same "honest scope" gap
documented across O1/O2/O3) simply leave their slot `None` rather than
fabricating a placeholder structure.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExecutionEvent:
    event_type: str
    component: str
    timestamp: str
    duration: float
    inputs: dict = field(default_factory=dict)
    outputs: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    confidence: float | None = None
    latency: float = 0.0
    memory_bytes: int | None = None


@dataclass(frozen=True)
class ExecutionTrace:
    execution_id: str
    query: str
    events: list[ExecutionEvent] = field(default_factory=list)

    # Per-subsystem trace slots. Each is a small real-signal dict pulled
    # from that subsystem's own already-computed output — never a
    # duplicated re-derivation of what that subsystem already knows.
    planner_trace: dict | None = None
    retrieval_trace: dict | None = None
    ranking_trace: dict | None = None
    compression_trace: dict | None = None
    claim_extraction_trace: dict | None = None
    verification_trace: dict | None = None
    nli_trace: dict | None = None
    recommendation_trace: dict | None = None
    faithfulness_trace: dict | None = None
    reflection_trace: dict | None = None
    vision_trace: dict | None = None
    prediction_trace: dict | None = None
    graph_trace: dict | None = None
    cache_trace: dict | None = None

    latencies: dict[str, float] = field(default_factory=dict)
    memory: dict[str, int] = field(default_factory=dict)
    confidence: float = 0.0
    timestamps: dict[str, str] = field(default_factory=dict)

    def event_count(self) -> int:
        return len(self.events)

    def events_for_component(self, component: str) -> list[ExecutionEvent]:
        return [e for e in self.events if e.component == component]

    def components(self) -> list[str]:
        """Component names in first-seen (i.e. execution) order."""
        seen: list[str] = []
        for event in self.events:
            if event.component not in seen:
                seen.append(event.component)
        return seen
