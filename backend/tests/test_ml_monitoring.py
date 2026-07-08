from app.ml.monitoring.monitor import ModelMonitor, population_stability_index


def test_psi_is_near_zero_for_identical_distributions():
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10
    assert population_stability_index(values, values) < 0.05


def test_psi_is_significant_for_a_large_distribution_shift():
    baseline = [0.1 + 0.01 * i for i in range(100)]
    recent = [0.9 + 0.01 * i for i in range(100)]

    assert population_stability_index(baseline, recent) > 0.25


def test_psi_returns_zero_for_empty_inputs():
    assert population_stability_index([], [1.0, 2.0]) == 0.0
    assert population_stability_index([1.0, 2.0], []) == 0.0


def test_psi_returns_zero_when_baseline_has_too_few_distinct_values():
    assert population_stability_index([0.5] * 20, [0.5, 0.6, 0.7]) == 0.0


def test_drift_severity_thresholds():
    assert ModelMonitor.drift_severity(0.05) == "none"
    assert ModelMonitor.drift_severity(0.15) == "moderate"
    assert ModelMonitor.drift_severity(0.3) == "significant"
    assert ModelMonitor.drift_severity(0.099) == "none"
    assert ModelMonitor.drift_severity(0.1) == "moderate"
    assert ModelMonitor.drift_severity(0.25) == "significant"


class _FakeLearningRepository:
    def __init__(self, accuracy=None, acceptance_rate=None):
        self._accuracy = accuracy
        self._acceptance_rate = acceptance_rate

    async def average_model_accuracy(self, metric_name: str):
        return self._accuracy

    async def recommendation_acceptance_rate(self, equipment_id: str):
        return self._acceptance_rate


def test_prediction_drift_delegates_to_population_stability_index():
    monitor = ModelMonitor(_FakeLearningRepository())
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5] * 10
    recent = [0.6, 0.7, 0.8, 0.9, 1.0] * 10

    assert monitor.prediction_drift(baseline, recent) == population_stability_index(baseline, recent)


def test_feature_drift_delegates_to_population_stability_index():
    monitor = ModelMonitor(_FakeLearningRepository())
    baseline = [1, 2, 3, 4, 5] * 10
    recent = [1, 2, 3, 4, 5] * 10

    assert monitor.feature_drift(baseline, recent) == population_stability_index(baseline, recent)


async def test_concept_drift_reads_average_model_accuracy():
    monitor = ModelMonitor(_FakeLearningRepository(accuracy=0.82))
    assert await monitor.concept_drift("accuracy") == 0.82


async def test_concept_drift_returns_none_without_recorded_metrics():
    monitor = ModelMonitor(_FakeLearningRepository(accuracy=None))
    assert await monitor.concept_drift("accuracy") is None


async def test_recommendation_acceptance_trend_reads_learning_repository():
    monitor = ModelMonitor(_FakeLearningRepository(acceptance_rate=0.65))
    assert await monitor.recommendation_acceptance_trend("P-101") == 0.65
