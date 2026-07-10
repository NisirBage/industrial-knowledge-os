"""O3.3 (Unified Execution Trace & Failure Intelligence Core): API response
shapes for ExecutionTrace, its events, reconstructed timelines, and
FailureEvents — the Research Dashboard's execution-visualization data
surface."""

from pydantic import BaseModel


class ExecutionEventOut(BaseModel):
    event_type: str
    component: str
    timestamp: str
    duration: float
    inputs: dict
    outputs: dict
    metadata: dict
    confidence: float | None = None
    latency: float
    memory_bytes: int | None = None


class ExecutionTraceOut(BaseModel):
    execution_id: str
    query: str
    events: list[ExecutionEventOut]
    latencies: dict[str, float]
    memory: dict[str, int]
    confidence: float
    timestamps: dict[str, str]
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


class TimelineSpanOut(BaseModel):
    component: str
    started_at: str
    finished_at: str
    duration_seconds: float
    confidence: float | None = None
    memory_bytes: int | None = None


class ExecutionDAGOut(BaseModel):
    nodes: list[str]
    edges: list[list[str]]


class ReconstructedTimelineOut(BaseModel):
    spans: list[TimelineSpanOut]
    component_order: list[str]
    parallel_branches: list[list[str]]
    cache_hits: list[str]
    confidence_evolution: list[list]
    dag: ExecutionDAGOut


class FailureEventOut(BaseModel):
    id: str
    execution_id: str
    component: str
    failure_type: str
    severity: str
    confidence: float
    root_cause: str
    evidence_ids: list[str]
    claim_ids: list[str]
    graph_nodes: list[str]
    metadata: dict


class ExecutionStatisticsOut(BaseModel):
    trace_count: int
    event_count: int
    average_confidence: float
    average_latency_by_component: dict[str, float]
