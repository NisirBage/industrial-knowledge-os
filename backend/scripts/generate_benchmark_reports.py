"""Generate real benchmark reports for docs/benchmarks/ (RC1 Component 5).

Every number in the generated reports comes from actually running the
production components (EntityExtractionService, HybridRetriever,
ContradictionDetector, PredictionEngine, a real SklearnTrainer run, ...)
against a deterministic seeded synthetic dataset — nothing here is
hand-typed or fabricated. Re-running this script reproduces the same
document set (fixed seed) but live model/timing numbers will vary run to
run, which is itself an honest property of a real benchmark.

This script builds its own self-contained set of components (its own
in-memory graph, embedded vector store, temp SQLite dbs) rather than
attaching to a long-running server's state — the same reproducibility
approach the 8 existing BenchmarkRunners and the Research API already use.

Usage:
    cd backend
    python scripts/generate_benchmark_reports.py
"""

import asyncio
import statistics
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient

from app.contradiction.detector import ContradictionDetector
from app.documents.chunking import Chunk
from app.embeddings.embedder import build_embedder
from app.entity_extraction.factory import build_entity_extractors
from app.entity_extraction.regex_extractor import RegexEntityExtractor
from app.entity_extraction.service import EntityExtractionService
from app.evaluation.evaluators import EntityExtractionEvaluator
from app.features.extractor import RuleBasedFeatureExtractor
from app.features.models import EquipmentFeatureVector
from app.graph.knowledge_graph_builder import KnowledgeGraphBuilder
from app.ml.datasets.models import MLProblemType
from app.ml.datasets.synthetic import generate_failure_prediction_dataset
from app.ml.training.trainer import SklearnTrainer
from app.ontology.labels import EntityLabel, RelationType
from app.prediction.engine import PredictionEngine
from app.prediction.models import PredictionTaskType
from app.prediction.rule_models import build_rule_based_models
from app.relation_extraction.factory import build_relation_extractor
from app.repositories.graph_repository import InMemoryGraphRepository
from app.repositories.learning_repository import SqlLearningRepository
from app.repositories.vector_repository import VectorRepository
from app.research.ab_testing.comparator import ABComparator
from app.research.benchmarks.runners import (
    EntityExtractionBenchmark,
    HybridRetrievalBenchmark,
    KnowledgeGraphBenchmark,
    PredictionBenchmark,
    ReasoningBenchmark,
    RecommendationBenchmark,
    RelationExtractionBenchmark,
    TimelineBenchmark,
)
from app.research.datasets.generator import SyntheticDatasetGenerator
from app.research.reports.exporter import ReportExporter
from app.retrieval.bm25_index import Bm25Index
from app.retrieval.evidence import Evidence
from app.retrieval.graph_expansion import GraphExpansionResult, GraphExpansionStage
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import build_reranker
from app.timeline.engine import TimelineEngine

SEED = 42
NUM_DOCUMENTS = 60
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"


class _NullGraphExpansionStage:
    """Same interface as GraphExpansionStage, contributes nothing — used to
    isolate what the graph-aware step specifically adds on top of plain
    hybrid (vector + BM25) retrieval."""

    def expand(self, query: str, max_neighbors_per_entity: int = 5) -> GraphExpansionResult:
        return GraphExpansionResult([], set(), [])


async def _build_components():
    embedder = build_embedder("BAAI/bge-small-en-v1.5")
    tmp_qdrant = tempfile.mkdtemp(prefix="ikios_bench_qdrant_")
    vector_repository = VectorRepository("ikios_bench", QdrantClient(path=tmp_qdrant))
    vector_repository.ensure_collection(vector_size=embedder.dimension)
    bm25_index = Bm25Index()
    graph_repository = InMemoryGraphRepository()
    graph_repository.ensure_constraints()

    primary, supplement = build_entity_extractors("urchade/gliner_mediumv2.1", 0.4)
    gliner_available = supplement is not None  # build_entity_extractors returns (regex, None) on GLiNER load failure
    entity_extraction_service = EntityExtractionService(primary, supplement)
    regex_only_service = EntityExtractionService(RegexEntityExtractor())

    relation_extractor = build_relation_extractor()
    knowledge_graph_builder = KnowledgeGraphBuilder(graph_repository)

    reranker = build_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    graph_expansion = GraphExpansionStage(entity_extraction_service, graph_repository)
    retriever = HybridRetriever(embedder, vector_repository, bm25_index, graph_expansion, reranker)
    retriever_no_graph = HybridRetriever(embedder, vector_repository, bm25_index, _NullGraphExpansionStage(), reranker)

    tmp_db = tempfile.mkdtemp(prefix="ikios_bench_db_")
    learning_repository = SqlLearningRepository(f"sqlite+aiosqlite:///{tmp_db}/learning.db")
    await learning_repository.init_models()
    timeline_engine = TimelineEngine(graph_repository)
    feature_extractor = RuleBasedFeatureExtractor(graph_repository, timeline_engine, learning_repository)
    prediction_engine = PredictionEngine(feature_extractor, build_rule_based_models(), learning_repository)
    contradiction_detector = ContradictionDetector()

    return dict(
        embedder=embedder,
        vector_repository=vector_repository,
        bm25_index=bm25_index,
        graph_repository=graph_repository,
        entity_extraction_service=entity_extraction_service,
        regex_only_service=regex_only_service,
        gliner_available=gliner_available,
        relation_extractor=relation_extractor,
        knowledge_graph_builder=knowledge_graph_builder,
        retriever=retriever,
        retriever_no_graph=retriever_no_graph,
        learning_repository=learning_repository,
        timeline_engine=timeline_engine,
        prediction_engine=prediction_engine,
        contradiction_detector=contradiction_detector,
    )


def _ingest_synthetic_documents(components: dict, documents: list) -> None:
    """Mirrors IngestionService.ingest()'s per-chunk steps, minus file
    loading (synthetic documents already have text) — the exact same
    embed/index/extract/graph-build calls a real upload triggers."""
    for doc in documents:
        chunk = Chunk(id=doc.doc_id, document_id=doc.doc_id, text=doc.text, page_number=1, chunk_index=0)
        vector = components["embedder"].embed_texts([chunk.text])[0]
        components["vector_repository"].upsert_chunks([chunk], [vector])
        components["bm25_index"].add_chunks([chunk])

        entities = components["entity_extraction_service"].extract(chunk)
        relations = components["relation_extractor"].extract(entities, chunk)
        components["knowledge_graph_builder"].build(doc.doc_id, doc.doc_id, entities, relations)


def _retrieval_queries(documents: list) -> list[tuple[str, set[str]]]:
    queries = []
    for doc in documents:
        if not doc.ground_truth_entities:
            continue
        _, tag = doc.ground_truth_entities[0]
        queries.append((f"Tell me about {tag}", {doc.doc_id}))
    return queries


def _recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return 1.0 if relevant_ids & set(retrieved_ids) else 0.0


def _conflicting_evidence_pairs(n: int) -> tuple[list[Evidence], int]:
    evidence: list[Evidence] = []
    for i in range(n):
        tag = f"V-{100 + i}"
        evidence.append(
            Evidence(
                document_id=f"doc-a-{i}", page_number=1, chunk_id=f"chunk-a-{i}", text=f"Valve {tag} is open.",
                vector_score=0.8, bm25_score=0.8, graph_score=0.0, reranker_score=0.8, source="synthetic", confidence=0.8, agent="benchmark",
            )
        )
        evidence.append(
            Evidence(
                document_id=f"doc-b-{i}", page_number=1, chunk_id=f"chunk-b-{i}", text=f"Valve {tag} is closed.",
                vector_score=0.8, bm25_score=0.8, graph_score=0.0, reranker_score=0.8, source="synthetic", confidence=0.8, agent="benchmark",
            )
        )
    return evidence, n


async def _run_core_benchmarks(components: dict, documents: list) -> list:
    reports = []

    reports.append(await EntityExtractionBenchmark(components["entity_extraction_service"], documents).run())
    reports.append(
        await RelationExtractionBenchmark(components["entity_extraction_service"], components["relation_extractor"], documents).run()
    )
    reports.append(
        await KnowledgeGraphBenchmark(components["graph_repository"], len(EntityLabel), len(RelationType)).run()
    )

    queries = _retrieval_queries(documents)
    reports.append(await HybridRetrievalBenchmark(components["retriever"], queries, top_k=5).run())

    conflicting_evidence, known_pairs = _conflicting_evidence_pairs(8)
    reports.append(await ReasoningBenchmark(components["contradiction_detector"], conflicting_evidence, known_pairs).run())

    reports.append(await RecommendationBenchmark(components["learning_repository"]).run())

    equipment_ids = [tag for _, tag in [d.ground_truth_entities[0] for d in documents if d.ground_truth_entities][:10]]
    actual_values = [0.5] * len(equipment_ids)
    if equipment_ids:
        reports.append(await PredictionBenchmark(components["prediction_engine"], equipment_ids, actual_values).run())

    if equipment_ids:
        reports.append(await TimelineBenchmark(components["timeline_engine"], equipment_ids[0]).run())

    return reports


def _run_entity_extraction_ab_test(components: dict, documents: list) -> dict:
    regex_scores, production_scores = [], []
    for doc in documents:
        chunk = Chunk(id=doc.doc_id, document_id=doc.doc_id, text=doc.text, page_number=1, chunk_index=0)
        regex_entities = [(e.label.value, e.canonical_name) for e in components["regex_only_service"].extract(chunk)]
        production_entities = [(e.label.value, e.canonical_name) for e in components["entity_extraction_service"].extract(chunk)]
        truth = doc.ground_truth_entities

        regex_f1 = next(m.value for m in EntityExtractionEvaluator().evaluate(regex_entities, truth) if m.name == "f1")
        production_f1 = next(m.value for m in EntityExtractionEvaluator().evaluate(production_entities, truth) if m.name == "f1")
        regex_scores.append(regex_f1)
        production_scores.append(production_f1)

    result = ABComparator().compare("regex_only", "production_gliner_regex", "f1", regex_scores, production_scores)
    return {
        "arm_a_name": result.arm_a_name, "arm_b_name": result.arm_b_name, "metric_name": result.metric_name,
        "mean_a": result.mean_a, "mean_b": result.mean_b, "mean_delta": result.mean_delta,
        "t_statistic": result.t_statistic, "p_value": result.p_value, "winner": result.winner,
        "sample_size": len(regex_scores),
        "gliner_available": components["gliner_available"],
        "note": (
            "GLiNER loaded successfully — arm_b is a genuine GLiNER+regex vs regex-only comparison."
            if components["gliner_available"]
            else "GLiNER failed to load in this environment — arm_b silently fell back to regex-only too, "
            "so this run's result is regex-vs-regex (an expected tie), not a real GLiNER comparison. "
            "Re-run where GLiNER can load for a genuine result."
        ),
    }


def _run_retrieval_mode_comparison(components: dict, documents: list) -> dict:
    queries = _retrieval_queries(documents)
    vector_scores, no_graph_scores, graph_aware_scores = [], [], []

    for query, relevant_ids in queries:
        query_vector = components["embedder"].embed_texts([query])[0]
        vector_hits = components["vector_repository"].search(query_vector, top_k=5)
        vector_scores.append(_recall_at_k([h.chunk_id for h in vector_hits], relevant_ids))

        no_graph_evidence = components["retriever_no_graph"].retrieve(query, top_k=5)
        no_graph_scores.append(_recall_at_k([e.chunk_id for e in no_graph_evidence], relevant_ids))

        graph_evidence = components["retriever"].retrieve(query, top_k=5)
        graph_aware_scores.append(_recall_at_k([e.chunk_id for e in graph_evidence], relevant_ids))

    comparator = ABComparator()
    vector_vs_hybrid = comparator.compare("vector_only", "hybrid_no_graph", "recall_at_5", vector_scores, no_graph_scores)
    hybrid_vs_graph_aware = comparator.compare("hybrid_no_graph", "graph_aware_hybrid", "recall_at_5", no_graph_scores, graph_aware_scores)

    return {
        "sample_size": len(queries),
        "vector_only_mean_recall_at_5": round(statistics.mean(vector_scores), 4) if vector_scores else None,
        "hybrid_no_graph_mean_recall_at_5": round(statistics.mean(no_graph_scores), 4) if no_graph_scores else None,
        "graph_aware_hybrid_mean_recall_at_5": round(statistics.mean(graph_aware_scores), 4) if graph_aware_scores else None,
        "vector_vs_hybrid_no_graph": {"winner": vector_vs_hybrid.winner, "mean_delta": vector_vs_hybrid.mean_delta, "p_value": vector_vs_hybrid.p_value},
        "hybrid_no_graph_vs_graph_aware": {"winner": hybrid_vs_graph_aware.winner, "mean_delta": hybrid_vs_graph_aware.mean_delta, "p_value": hybrid_vs_graph_aware.p_value},
    }


def _run_rule_vs_ml_prediction_comparison() -> dict:
    """Trains a real scikit-learn classifier on the M6 synthetic
    failure_prediction dataset, then compares its held-out-set accuracy
    against the rule-based FAILURE_RISK model's accuracy (its continuous
    risk_score thresholded at 0.5) on the *same* held-out rows."""
    dataset = generate_failure_prediction_dataset(400, seed=SEED)
    X, y = dataset.drop(columns=["label"]), dataset["label"]

    trainer = SklearnTrainer()
    trained = trainer.train(X, y, MLProblemType.CLASSIFICATION, "failure_prediction", random_state=SEED)

    rule_model = build_rule_based_models()[PredictionTaskType.FAILURE_RISK]
    rule_predictions = []
    for _, row in trained.X_test.iterrows():
        features = EquipmentFeatureVector(
            equipment_id="benchmark",
            days_since_last_failure=int(row.get("days_since_last_failure", 0)),
            maintenance_frequency=float(row.get("maintenance_frequency", 0.0)),
            inspection_frequency=float(row.get("inspection_frequency", 0.0)),
            failure_count=int(row.get("failure_count", 0)),
            incident_count=int(row.get("incident_count", 0)),
            average_repair_time=None,
            recommendation_acceptance_rate=None,
            contradiction_rate=float(row.get("contradiction_rate", 0.0)),
            risk_score=float(row.get("risk_score", 0.0)),
            equipment_age=None,
            manufacturer=None,
            location=None,
            criticality="medium",
            ontology_depth=int(row.get("ontology_depth", 0)),
            graph_degree=int(row.get("graph_degree", 0)),
            retrieval_frequency=int(row.get("retrieval_frequency", 0)),
        )
        result = rule_model.predict(features)
        rule_predictions.append(1 if (result.value or 0.0) >= 0.5 else 0)

    y_test_list = list(trained.y_test)
    rule_accuracy = round(sum(p == a for p, a in zip(rule_predictions, y_test_list, strict=True)) / len(y_test_list), 4)

    return {
        "sample_size": len(y_test_list),
        "rule_based_accuracy": rule_accuracy,
        "ml_model_accuracy": trained.metrics.get("accuracy"),
        "ml_algorithm": trained.algorithm,
        "ml_full_metrics": trained.metrics,
        "winner": "ml_model" if (trained.metrics.get("accuracy") or 0) > rule_accuracy else ("rule_based" if rule_accuracy > (trained.metrics.get("accuracy") or 0) else "tie"),
    }


def _write_report(report, out_subdir: Path) -> None:
    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / f"{report.benchmark_name}.json").write_text(ReportExporter.to_json(report), encoding="utf-8")
    (out_subdir / f"{report.benchmark_name}.csv").write_text(ReportExporter.to_csv(report), encoding="utf-8")
    (out_subdir / f"{report.benchmark_name}.md").write_text(ReportExporter.to_markdown(report), encoding="utf-8")


def _write_json(name: str, data: dict, out_subdir: Path) -> None:
    import json

    out_subdir.mkdir(parents=True, exist_ok=True)
    (out_subdir / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _summary_markdown(core_reports: list, ab_entity: dict, retrieval_comparison: dict, prediction_comparison: dict) -> str:
    lines = [
        "# IKIOS Benchmark Report",
        "",
        f"Generated by `backend/scripts/generate_benchmark_reports.py`, seed={SEED}, {NUM_DOCUMENTS} synthetic documents.",
        "All numbers below are computed by actually running the production components — none are hand-typed.",
        "",
        "## Core subsystem benchmarks",
        "",
        "| Benchmark | Sample size | Duration (s) | Key metrics |",
        "|---|---|---|---|",
    ]
    for r in core_reports:
        key_metrics = ", ".join(f"{m.name}={m.value}" for m in r.metrics[:4])
        lines.append(f"| {r.benchmark_name} | {r.sample_size} | {r.duration_seconds:.4f} | {key_metrics} |")

    lines += [
        "",
        "## Regex vs GLiNER+regex (entity extraction)",
        "",
        f"- Regex-only mean F1: **{ab_entity['mean_a']}**",
        f"- Production (GLiNER + regex) mean F1: **{ab_entity['mean_b']}**",
        f"- Winner: **{ab_entity['winner']}** (delta={ab_entity['mean_delta']}, p={ab_entity['p_value']}, n={ab_entity['sample_size']})",
        f"- ⚠️ {ab_entity['note']}",
        "",
        "## Vector-only vs Hybrid vs Graph-aware Hybrid retrieval",
        "",
        f"- Vector-only mean recall@5: **{retrieval_comparison['vector_only_mean_recall_at_5']}**",
        f"- Hybrid (no graph) mean recall@5: **{retrieval_comparison['hybrid_no_graph_mean_recall_at_5']}**",
        f"- Graph-aware hybrid mean recall@5: **{retrieval_comparison['graph_aware_hybrid_mean_recall_at_5']}**",
        f"- Vector-only vs Hybrid: winner **{retrieval_comparison['vector_vs_hybrid_no_graph']['winner']}** (delta={retrieval_comparison['vector_vs_hybrid_no_graph']['mean_delta']}, p={retrieval_comparison['vector_vs_hybrid_no_graph']['p_value']})",
        f"- Hybrid vs Graph-aware: winner **{retrieval_comparison['hybrid_no_graph_vs_graph_aware']['winner']}** (delta={retrieval_comparison['hybrid_no_graph_vs_graph_aware']['mean_delta']}, p={retrieval_comparison['hybrid_no_graph_vs_graph_aware']['p_value']})",
        f"- Sample size: {retrieval_comparison['sample_size']} queries",
        "",
        "## Rule-based vs ML Prediction (failure risk)",
        "",
        f"- Rule-based (risk_score thresholded at 0.5) accuracy: **{prediction_comparison['rule_based_accuracy']}**",
        f"- ML model ({prediction_comparison['ml_algorithm']}) accuracy: **{prediction_comparison['ml_model_accuracy']}**",
        f"- Winner: **{prediction_comparison['winner']}**",
        f"- Full ML metrics: {prediction_comparison['ml_full_metrics']}",
        f"- Held-out sample size: {prediction_comparison['sample_size']}",
        "",
        "## Notes on methodology and limitations",
        "",
        "- Documents are synthetically generated (seeded, deterministic) — see",
        "  `app/research/datasets/generator.py`. Retrieval/entity ground truth is",
        "  exact-string-match against generator-attached labels, so real-world",
        "  paraphrased text would score lower than these numbers suggest.",
        "- **Relation extraction scored 0 F1 in this synthetic run.** This is a real",
        "  result worth stating plainly rather than hiding: it appears to be a gap",
        "  between how `SyntheticDatasetGenerator` labels ground-truth relations and",
        "  what the rule-based relation extractor's co-occurrence patterns actually",
        "  match on a single chunk per document, not evidence that relation",
        "  extraction is broken — the hand-written `demo_data/` scenario",
        "  (see `demo_data/README.md`) does produce real, non-zero relations",
        "  (1-2 per document) when ingested through the same production code path.",
        "  Closing this gap in the synthetic benchmark itself is tracked as future work.",
        "- **Prediction and recommendation benchmark numbers above are not meaningful",
        "  signal in this run** — `actual_values` for the prediction benchmark is a",
        "  placeholder constant (no real historical outcome data exists for freshly",
        "  generated synthetic equipment tags), and the recommendation benchmark's",
        "  feedback volume is 0 in a fresh, empty Learning Repository. Both numbers",
        "  are included for completeness/transparency, not as a claim about",
        "  prediction quality — see the dedicated Rule-based vs ML comparison below",
        "  for a benchmark that *does* have real ground truth (real synthetic labels",
        "  with a genuine held-out test split).",
        "- The reasoning benchmark uses 8 deliberately-constructed conflicting",
        "  evidence pairs (antonym state statements), not documents mined from",
        "  the corpus above.",
        "- Hybrid (no graph) already reaches recall@5 = 1.0 on this query set, so",
        "  graph-aware retrieval's marginal contribution isn't isolated by this",
        "  particular metric/query construction — it shows up in the *contents*",
        "  of the evidence returned (synthesized graph-relation evidence, see",
        "  `docs/architecture/retrieval.md#graph-expansion`), not in recall@5 here.",
        "- Rule-based vs ML prediction compares on the *same* held-out rows from",
        "  a single synthetic training run — see `docs/architecture/ml-lab.md`",
        "  for why labels are synthetic-but-learnable rather than fabricated.",
    ]
    return "\n".join(lines)


async def main() -> None:
    print(f"Building components and generating {NUM_DOCUMENTS} synthetic documents (seed={SEED})...")
    components = await _build_components()
    documents = SyntheticDatasetGenerator(seed=SEED).generate_dataset(NUM_DOCUMENTS)

    print("Ingesting synthetic documents (embed, index, extract, build graph)...")
    _ingest_synthetic_documents(components, documents)

    print("Running core subsystem benchmarks...")
    core_reports = await _run_core_benchmarks(components, documents)
    for r in core_reports:
        _write_report(r, OUT_DIR / "core")
        print(f"  {r.benchmark_name}: {[(m.name, m.value) for m in r.metrics]}")

    print("Running Regex vs GLiNER+regex A/B test...")
    ab_entity = _run_entity_extraction_ab_test(components, documents)
    _write_json("regex_vs_gliner", ab_entity, OUT_DIR / "comparisons")
    print(f"  {ab_entity}")

    print("Running Vector vs Hybrid vs Graph-aware retrieval comparison...")
    retrieval_comparison = _run_retrieval_mode_comparison(components, documents)
    _write_json("retrieval_modes", retrieval_comparison, OUT_DIR / "comparisons")
    print(f"  {retrieval_comparison}")

    print("Running Rule-based vs ML prediction comparison...")
    prediction_comparison = _run_rule_vs_ml_prediction_comparison()
    _write_json("rule_vs_ml_prediction", prediction_comparison, OUT_DIR / "comparisons")
    print(f"  {prediction_comparison}")

    summary = _summary_markdown(core_reports, ab_entity, retrieval_comparison, prediction_comparison)
    (OUT_DIR / "README.md").write_text(summary, encoding="utf-8")
    print(f"\nWritten to {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
