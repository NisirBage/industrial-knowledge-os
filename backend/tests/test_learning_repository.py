from app.repositories.learning_repository import SqlLearningRepository


async def _repo(tmp_path):
    db_path = tmp_path / "learning_test.db"
    repository = SqlLearningRepository(f"sqlite+aiosqlite:///{db_path}")
    await repository.init_models()
    return repository


async def test_record_and_average_prediction_confidence(tmp_path):
    repo = await _repo(tmp_path)

    await repo.record_prediction("failure_risk", "P-101", 0.5, 0.6, "explanation", "rule_based_v1")
    await repo.record_prediction("failure_risk", "P-101", 0.7, 0.8, "explanation", "rule_based_v1")

    assert await repo.average_prediction_confidence() == 0.7


async def test_average_prediction_confidence_none_when_no_predictions(tmp_path):
    repo = await _repo(tmp_path)
    assert await repo.average_prediction_confidence() is None


async def test_latest_prediction_confidence_returns_most_recent_for_that_equipment(tmp_path):
    repo = await _repo(tmp_path)

    await repo.record_prediction("failure_risk", "P-101", 0.5, 0.6, "explanation", "rule_based_v1")
    await repo.record_prediction("failure_risk", "P-101", 0.7, 0.9, "explanation", "rule_based_v1")
    await repo.record_prediction("failure_risk", "V-201", 0.3, 0.2, "explanation", "rule_based_v1")

    assert await repo.latest_prediction_confidence("P-101") == 0.9
    assert await repo.latest_prediction_confidence("V-201") == 0.2


async def test_latest_prediction_confidence_none_for_unknown_equipment(tmp_path):
    repo = await _repo(tmp_path)

    assert await repo.latest_prediction_confidence("does-not-exist") is None


async def test_record_outcome_is_independently_stored(tmp_path):
    repo = await _repo(tmp_path)
    prediction_id = await repo.record_prediction("failure_risk", "P-101", 0.5, 0.6, "explanation", "rule_based_v1")

    outcome_id = await repo.record_outcome(prediction_id, actual_value=1.0, notes="failed as predicted")
    assert outcome_id


async def test_recommendation_acceptance_rate_computed_from_feedback(tmp_path):
    repo = await _repo(tmp_path)

    await repo.record_feedback("recommendation_accepted", "rec-1", None, None, equipment_id="P-101")
    await repo.record_feedback("recommendation_accepted", "rec-2", None, None, equipment_id="P-101")
    await repo.record_feedback("recommendation_rejected", "rec-3", None, None, equipment_id="P-101")

    rate = await repo.recommendation_acceptance_rate("P-101")
    assert rate == round(2 / 3, 4)


async def test_recommendation_acceptance_rate_none_without_feedback(tmp_path):
    repo = await _repo(tmp_path)
    assert await repo.recommendation_acceptance_rate("P-999") is None


async def test_recommendation_acceptance_rate_scoped_per_equipment(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_feedback("recommendation_accepted", "rec-1", None, None, equipment_id="P-101")
    await repo.record_feedback("recommendation_rejected", "rec-2", None, None, equipment_id="V-2201A")

    assert await repo.recommendation_acceptance_rate("P-101") == 1.0
    assert await repo.recommendation_acceptance_rate("V-2201A") == 0.0


async def test_contradiction_rate_uses_interaction_count_as_denominator(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_interaction("P-101", "why did P-101 fail?")
    await repo.record_interaction("P-101", "is P-101 due for inspection?")
    await repo.record_contradiction("P-101", "conflicting valve state", "high")

    assert await repo.contradiction_rate("P-101") == 0.5


async def test_contradiction_rate_zero_with_no_data(tmp_path):
    repo = await _repo(tmp_path)
    assert await repo.contradiction_rate("P-999") == 0.0


async def test_retrieval_frequency_counts_interactions(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_interaction("P-101", "q1")
    await repo.record_interaction("P-101", "q2")
    await repo.record_interaction("V-2201A", "q3")

    assert await repo.retrieval_frequency("P-101") == 2
    assert await repo.retrieval_frequency("V-2201A") == 1


async def test_feedback_volume_counts_all_feedback(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_feedback("recommendation_accepted", "rec-1", None, None)
    await repo.record_feedback("entity_correction", "entity-1", "wrong label", "Pump")

    assert await repo.feedback_volume() == 2


async def test_average_model_accuracy_filters_by_metric_name(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_model_performance("failure_risk", "rule_based_v1", "accuracy", 0.8)
    await repo.record_model_performance("failure_risk", "rule_based_v1", "accuracy", 0.6)
    await repo.record_model_performance("failure_risk", "rule_based_v1", "prediction_latency_seconds", 0.001)

    assert await repo.average_model_accuracy("accuracy") == 0.7
    assert await repo.average_model_accuracy("nonexistent_metric") is None


async def test_confidence_calibration_buckets_groups_by_decile(tmp_path):
    repo = await _repo(tmp_path)
    prediction_id = await repo.record_prediction("failure_risk", "P-101", 0.5, 0.75, "explanation", "rule_based_v1")
    await repo.record_outcome(prediction_id, actual_value=0.55)

    buckets = await repo.confidence_calibration_buckets(threshold=0.2)

    assert len(buckets) == 1
    assert buckets[0]["confidence_bucket"] == "70-80%"
    assert buckets[0]["sample_count"] == 1
    assert buckets[0]["actual_accuracy"] == 1.0


async def test_confidence_calibration_buckets_reports_none_accuracy_without_outcomes(tmp_path):
    repo = await _repo(tmp_path)
    await repo.record_prediction("failure_risk", "P-101", 0.5, 0.4, "explanation", "rule_based_v1")

    buckets = await repo.confidence_calibration_buckets()

    assert buckets[0]["actual_accuracy"] is None
    assert buckets[0]["sample_count"] == 1


async def test_confidence_calibration_buckets_flags_inaccurate_predictions(tmp_path):
    repo = await _repo(tmp_path)
    prediction_id = await repo.record_prediction("failure_risk", "P-101", 0.9, 0.9, "explanation", "rule_based_v1")
    await repo.record_outcome(prediction_id, actual_value=0.1)

    buckets = await repo.confidence_calibration_buckets(threshold=0.2)

    assert buckets[0]["actual_accuracy"] == 0.0
