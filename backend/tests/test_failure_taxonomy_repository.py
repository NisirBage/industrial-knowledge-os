from app.error_analysis.models import DetectedFailure, FailureCategory, FailureSeverity
from app.error_analysis.repository import SqlFailureTaxonomyRepository


async def _repository(tmp_path) -> SqlFailureTaxonomyRepository:
    db_path = tmp_path / "failure_taxonomy_test.db"
    repository = SqlFailureTaxonomyRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


def _failure(category: FailureCategory = FailureCategory.RETRIEVAL_FAILURE, severity: FailureSeverity = FailureSeverity.CRITICAL) -> DetectedFailure:
    return DetectedFailure(
        category=category,
        root_cause="No evidence retrieved.",
        affected_components=["HybridRetriever"],
        confidence=0.9,
        suggested_improvement="Check ingestion coverage.",
        severity=severity,
    )


async def test_record_and_list_round_trips_a_failure(tmp_path):
    repository = await _repository(tmp_path)

    record_id = await repository.record("Why did P-101 fail?", _failure())
    assert record_id

    failures = await repository.list_by_category()
    assert len(failures) == 1
    assert failures[0].category == FailureCategory.RETRIEVAL_FAILURE
    assert failures[0].affected_components == ["HybridRetriever"]
    assert failures[0].severity == FailureSeverity.CRITICAL
    assert failures[0].frequency == 1


async def test_list_by_category_filters(tmp_path):
    repository = await _repository(tmp_path)
    await repository.record("q1", _failure(FailureCategory.RETRIEVAL_FAILURE))
    await repository.record("q2", _failure(FailureCategory.NLI_FAILURE))

    retrieval_only = await repository.list_by_category(FailureCategory.RETRIEVAL_FAILURE)
    assert len(retrieval_only) == 1
    assert retrieval_only[0].category == FailureCategory.RETRIEVAL_FAILURE


async def test_list_by_category_with_no_filter_returns_all(tmp_path):
    repository = await _repository(tmp_path)
    await repository.record("q1", _failure(FailureCategory.RETRIEVAL_FAILURE))
    await repository.record("q2", _failure(FailureCategory.NLI_FAILURE))

    all_failures = await repository.list_by_category()
    assert len(all_failures) == 2


async def test_category_counts_aggregates_correctly(tmp_path):
    repository = await _repository(tmp_path)
    await repository.record("q1", _failure(FailureCategory.RETRIEVAL_FAILURE))
    await repository.record("q2", _failure(FailureCategory.RETRIEVAL_FAILURE))
    await repository.record("q3", _failure(FailureCategory.NLI_FAILURE))

    counts = await repository.category_counts()
    assert counts["retrieval_failure"] == 2
    assert counts["nli_failure"] == 1


async def test_empty_repository_returns_no_failures_and_empty_counts(tmp_path):
    repository = await _repository(tmp_path)

    assert await repository.list_by_category() == []
    assert await repository.category_counts() == {}
