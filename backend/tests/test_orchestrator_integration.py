"""End-to-end test of the M3 pipeline: planner -> retrieval -> knowledge_graph
-> evidence_fusion -> reasoning -> recommendation -> verifier -> report,
using fakes for the retriever/graph repository/LLM so it runs fast and
deterministically without any real model or database.
"""

from app.agents.claim_extraction_agent import ClaimExtractionAgent
from app.agents.claim_verification_agent import ClaimVerificationAgent
from app.agents.confidence_propagation_agent import ConfidencePropagationAgent
from app.agents.evidence_fusion_agent import EvidenceFusionAgent
from app.agents.faithfulness_agent import FaithfulnessAgent
from app.agents.knowledge_graph_agent import KnowledgeGraphAgent
from app.agents.planner import PlannerAgent
from app.agents.reasoning_agent import ReasoningAgent
from app.agents.recommendation_agent import RecommendationAgent
from app.agents.recommendation_verification_agent import RecommendationVerificationAgent
from app.agents.reflection_agent import ReflectionAgent
from app.agents.report_agent import ReportAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.verifier_agent import VerifierAgent
from app.claims.extraction import ClaimExtractionEngine
from app.claims.propagation import GraphDecayConfidencePropagator
from app.claims.verification import ClaimVerificationEngine
from app.contradiction.detector import ContradictionDetector
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evidence.fusion import EvidenceFusionEngine
from app.evidence.graph import EvidenceGraphBuilder
from app.faithfulness.scorer import FaithfulnessScorer
from app.narration.narrator import LLMNarrator
from app.nli.rule_based import RuleBasedNLIDetector
from app.orchestrator.graph import build_orchestrator
from app.planning.executor import TaskGraphExecutor
from app.planning.planner import TaskGraphPlanner
from app.reasoning.reasoning_graph_builder import ReasoningGraphBuilder
from app.recommendation.engine import RecommendationEngine
from app.recommendation.verification import RecommendationVerifier
from app.reflection.engine import SelfReflectionEngine
from app.relation_extraction.rule_based import RuleBasedRelationExtractor
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionStage
from app.services.llm_provider import MockLLMProvider


class _FakeRetriever:
    def retrieve(self, query: str, top_k: int = 8):
        return [
            Evidence(
                document_id="doc-1",
                page_number=3,
                chunk_id="c1",
                text="Pump P-101 experienced a bearing failure in March 2024 due to inadequate lubrication.",
                confidence=0.8,
                source="fusion",
            )
        ]


class _FakeGraphRepository:
    def find_entity(self, canonical_name: str, label=None):
        if canonical_name == "P-101":
            return {"id": "node-p101", "type": "Pump", "canonical_name": "P-101", "aliases": []}
        return None

    def documents_mentioning(self, canonical_name: str):
        return [{"document_node_id": "doc-node-1", "document_id": "doc-1", "filename": "manual.pdf"}]

    def find_neighbors(self, entity_id: str, relation_type=None, depth: int = 1):
        if relation_type is not None and relation_type.value == "USES_STANDARD":
            return []
        return [{"id": "node-failure1", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]

    def maintenance_history(self, canonical_name: str):
        return [{"id": "node-maint1", "type": "Inspection", "canonical_name": "Quarterly Inspection", "relation_type": "INSPECTED_BY"}]

    def incident_history(self, canonical_name: str):
        return [{"id": "node-incident1", "type": "Incident", "canonical_name": "Bearing Failure", "relation_type": "FAILED_DUE_TO"}]


class _PassthroughReranker:
    def score(self, query: str, texts: list[str]):
        return [0.0] * len(texts)


def _build_test_orchestrator():
    llm_provider = MockLLMProvider()
    entity_extraction_service = EntityExtractionService(RegexEntityExtractor())
    graph_repository = _FakeGraphRepository()
    retriever = _FakeRetriever()

    task_graph_planner = TaskGraphPlanner()
    task_graph_executor = TaskGraphExecutor(retriever, graph_repository)
    graph_expansion = GraphExpansionStage(entity_extraction_service, graph_repository)

    nli_detector = RuleBasedNLIDetector()

    return build_orchestrator(
        planner_agent=PlannerAgent(entity_extraction_service, task_graph_planner),
        retrieval_agent=RetrievalAgent(task_graph_executor),
        knowledge_graph_agent=KnowledgeGraphAgent(graph_expansion),
        evidence_fusion_agent=EvidenceFusionAgent(EvidenceFusionEngine()),
        reasoning_agent=ReasoningAgent(ContradictionDetector(), EvidenceGraphBuilder(), ReasoningGraphBuilder()),
        claim_extraction_agent=ClaimExtractionAgent(
            ClaimExtractionEngine(entity_extraction_service, RuleBasedRelationExtractor())
        ),
        claim_verification_agent=ClaimVerificationAgent(ClaimVerificationEngine(graph_repository)),
        confidence_propagation_agent=ConfidencePropagationAgent(GraphDecayConfidencePropagator()),
        recommendation_agent=RecommendationAgent(RecommendationEngine(llm_provider)),
        verifier_agent=VerifierAgent(),
        recommendation_verification_agent=RecommendationVerificationAgent(RecommendationVerifier()),
        report_agent=ReportAgent(LLMNarrator(llm_provider)),
        faithfulness_agent=FaithfulnessAgent(FaithfulnessScorer(nli_detector)),
        reflection_agent=ReflectionAgent(SelfReflectionEngine()),
    )


def test_full_pipeline_produces_grounded_answer_with_recommendation():
    orchestrator = _build_test_orchestrator()

    final_state = orchestrator.invoke({"query": "Why has Pump P-101 repeatedly failed?"})

    from app.reasoning.graph_models import ReasoningNodeType

    assert final_state["task_graph"].all_completed()
    assert final_state["evidence_bundle"].evidence
    assert final_state["reasoning_graph"].nodes_of_type(ReasoningNodeType.RISK)
    assert final_state["recommendation_bundle"].recommendations
    assert final_state["answer"]
    assert final_state["citations"]


def test_pipeline_tracks_latency_for_every_stage():
    orchestrator = _build_test_orchestrator()

    final_state = orchestrator.invoke({"query": "Why has Pump P-101 repeatedly failed?"})

    expected_stages = {
        "planning", "retrieval", "knowledge_graph", "evidence_fusion",
        "reasoning", "claim_extraction", "claim_verification", "confidence_propagation",
        "recommendation", "verification", "recommendation_verification", "narration", "faithfulness",
        "reflection",
    }
    assert expected_stages.issubset(final_state["latencies"].keys())


def test_recommendation_is_traceable_to_evidence_in_bundle():
    orchestrator = _build_test_orchestrator()
    final_state = orchestrator.invoke({"query": "Why has Pump P-101 repeatedly failed?"})

    known_chunk_ids = {e.chunk_id for e in final_state["evidence_bundle"].all_evidence()}
    for recommendation in final_state["recommendation_bundle"].recommendations:
        assert recommendation.supporting_evidence
        for chunk_id in recommendation.supporting_evidence:
            assert chunk_id in known_chunk_ids


def test_pipeline_emits_started_and_finished_execution_events_per_stage():
    orchestrator = _build_test_orchestrator()

    final_state = orchestrator.invoke({"query": "Why has Pump P-101 repeatedly failed?"})

    events = final_state["execution_events"]
    assert events
    event_types = {e.event_type for e in events}
    assert "planning_started" in event_types
    assert "planning_finished" in event_types
    assert "reflection_finished" in event_types

    planner_finished = next(e for e in events if e.event_type == "planning_finished")
    assert planner_finished.component == "planning"
    assert planner_finished.inputs["query"] == "Why has Pump P-101 repeatedly failed?"
    assert planner_finished.latency >= 0.0


def test_query_with_no_resolvable_entity_still_produces_an_answer():
    orchestrator = _build_test_orchestrator()

    final_state = orchestrator.invoke({"query": "What is the plant safety policy?"})

    assert final_state["task_graph"].nodes[0].type.value == "general_retrieval"
    assert final_state["answer"]
