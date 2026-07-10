from app.error_analysis.models import FailureSeverity
from app.failure_intelligence.models import FailurePattern
from app.failure_intelligence.repository import SqlFailurePatternRepository


async def _repository(tmp_path) -> SqlFailurePatternRepository:
    db_path = tmp_path / "failure_pattern_test.db"
    repository = SqlFailurePatternRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


def _pattern(pattern_id: str = "p1") -> FailurePattern:
    return FailurePattern(
        pattern_id=pattern_id, category="retrieval_failure", support=3, frequency=0.5,
        severity=FailureSeverity.HIGH, affected_components=["HybridRetriever"],
        example_execution_ids=["e1", "e2"], confidence=0.8, description="d",
    )


async def test_save_and_get_round_trips_a_pattern(tmp_path):
    repository = await _repository(tmp_path)
    pattern_id = await repository.save(_pattern())
    assert pattern_id == "p1"

    fetched = await repository.get("p1")
    assert fetched is not None
    assert fetched.category == "retrieval_failure"
    assert fetched.severity == FailureSeverity.HIGH
    assert fetched.affected_components == ["HybridRetriever"]


async def test_get_returns_none_for_unknown_pattern_id(tmp_path):
    repository = await _repository(tmp_path)
    assert await repository.get("nonexistent") is None


async def test_save_all_persists_multiple_patterns(tmp_path):
    repository = await _repository(tmp_path)
    ids = await repository.save_all([_pattern("p1"), _pattern("p2")])
    assert ids == ["p1", "p2"]

    all_patterns = await repository.list_all()
    assert len(all_patterns) == 2


async def test_list_all_on_empty_repository_returns_empty_list(tmp_path):
    repository = await _repository(tmp_path)
    assert await repository.list_all() == []
