from app.research.experiments.manager import ExperimentManager
from app.research.experiments.models import ExperimentStatus
from app.research.experiments.repository import SqlExperimentTracker


async def _manager(tmp_path) -> ExperimentManager:
    db_path = tmp_path / "experiments_test.db"
    tracker = SqlExperimentTracker(f"sqlite+aiosqlite:///{db_path}")
    await tracker.init_models()
    return ExperimentManager(tracker)


async def test_start_creates_running_experiment(tmp_path):
    manager = await _manager(tmp_path)

    experiment = await manager.start(
        name="test_experiment",
        description="a test",
        configuration={"seed": "42"},
        model_versions=[],
        retriever_versions=[],
        graph_versions=[],
        reranker="none",
        ontology_version="1.0",
        dataset="synthetic",
        random_seed=42,
    )

    assert experiment.status == ExperimentStatus.RUNNING
    assert experiment.random_seed == 42
    assert experiment.duration is None


async def test_complete_sets_status_and_metrics(tmp_path):
    manager = await _manager(tmp_path)
    experiment = await manager.start("t", "d", {}, [], [], [], "none", "1.0", "synthetic")

    completed = await manager.complete(experiment.experiment_id, {"f1": 0.9})

    assert completed.status == ExperimentStatus.COMPLETED
    assert completed.metrics == {"f1": 0.9}
    assert completed.duration is not None
    assert completed.duration >= 0


async def test_fail_sets_status_and_error(tmp_path):
    manager = await _manager(tmp_path)
    experiment = await manager.start("t", "d", {}, [], [], [], "none", "1.0", "synthetic")

    failed = await manager.fail(experiment.experiment_id, "boom")

    assert failed.status == ExperimentStatus.FAILED
    assert failed.error == "boom"


async def test_get_returns_none_for_unknown_id(tmp_path):
    manager = await _manager(tmp_path)
    assert await manager.get("does-not-exist") is None


async def test_get_returns_persisted_experiment(tmp_path):
    manager = await _manager(tmp_path)
    experiment = await manager.start("t", "d", {}, [], [], [], "none", "1.0", "synthetic")

    fetched = await manager.get(experiment.experiment_id)

    assert fetched is not None
    assert fetched.experiment_id == experiment.experiment_id


async def test_list_returns_all_experiments_newest_first(tmp_path):
    manager = await _manager(tmp_path)
    first = await manager.start("first", "d", {}, [], [], [], "none", "1.0", "synthetic")
    await manager.complete(first.experiment_id, {})
    await manager.start("second", "d", {}, [], [], [], "none", "1.0", "synthetic")

    experiments = await manager.list()

    assert len(experiments) == 2
    names = {e.name for e in experiments}
    assert names == {"first", "second"}


async def test_complete_raises_for_unknown_experiment(tmp_path):
    manager = await _manager(tmp_path)
    try:
        await manager.complete("unknown-id", {})
        assert False, "expected KeyError"
    except KeyError:
        pass


async def test_experiment_versions_round_trip_through_storage(tmp_path):
    from app.research.model_registry import ComponentKind, entity_extractor_version, retriever_version

    manager = await _manager(tmp_path)
    experiment = await manager.start(
        "t", "d", {},
        model_versions=[entity_extractor_version("gliner", "v1", threshold="0.4")],
        retriever_versions=[retriever_version("hybrid", "v1")],
        graph_versions=[],
        reranker="ms-marco",
        ontology_version="1.0",
        dataset="synthetic",
    )

    fetched = await manager.get(experiment.experiment_id)

    assert fetched.model_versions[0].name == "gliner"
    assert fetched.model_versions[0].config == {"threshold": "0.4"}
    assert fetched.retriever_versions[0].name == "hybrid"
    # Regression check: ComponentKind is a str subclass, so json.dumps
    # accepts it silently without erroring — easy for the reload path to
    # silently degrade it back to a plain string instead of the enum.
    assert fetched.model_versions[0].kind == ComponentKind.ENTITY_EXTRACTOR
    assert isinstance(fetched.model_versions[0].kind, ComponentKind)
    assert fetched.retriever_versions[0].kind == ComponentKind.RETRIEVER
