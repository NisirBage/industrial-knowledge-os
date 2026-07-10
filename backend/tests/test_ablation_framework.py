from app.research.ablation.models import AblationAxis, AblationConfiguration
from app.research.ablation.runner import (
    AblationRunner,
    generate_full_factorial_configurations,
    generate_one_factor_at_a_time_configurations,
)


def test_ablation_configuration_defaults_to_everything_enabled():
    config = AblationConfiguration()
    assert config.is_enabled(AblationAxis.NLI)
    assert config.is_enabled(AblationAxis.CACHING)


def test_ablation_configuration_is_enabled_respects_explicit_set():
    config = AblationConfiguration(enabled_axes=frozenset({AblationAxis.NLI}))
    assert config.is_enabled(AblationAxis.NLI)
    assert not config.is_enabled(AblationAxis.CACHING)


def test_one_factor_at_a_time_generates_one_configuration_per_axis():
    configurations = generate_one_factor_at_a_time_configurations()
    assert len(configurations) == len(list(AblationAxis))
    # each configuration disables exactly one axis relative to "everything on"
    for config in configurations:
        assert len(config.enabled_axes) == len(list(AblationAxis)) - 1


def test_one_factor_at_a_time_accepts_a_custom_axis_subset():
    configurations = generate_one_factor_at_a_time_configurations([AblationAxis.NLI, AblationAxis.CACHING])
    assert len(configurations) == 2


def test_full_factorial_generates_real_power_of_two_combinations():
    configurations = generate_full_factorial_configurations([AblationAxis.NLI, AblationAxis.CACHING])
    assert len(configurations) == 4  # 2**2
    # every combination must be distinct
    assert len({config.enabled_axes for config in configurations}) == 4


def test_full_factorial_raises_above_the_axis_cap():
    import pytest

    with pytest.raises(ValueError):
        generate_full_factorial_configurations(list(AblationAxis))  # 16 axes, way above the cap


def test_full_factorial_other_axes_stay_enabled_in_every_combination():
    configurations = generate_full_factorial_configurations([AblationAxis.NLI])
    for config in configurations:
        assert config.is_enabled(AblationAxis.CACHING)  # not part of the swept subset -> always on


async def test_runner_run_axis_raises_for_unregistered_axis(monkeypatch):
    import pytest
    from app.research.ablation import runner as runner_module

    monkeypatch.setattr(runner_module, "PROBES", {})
    with pytest.raises(ValueError):
        await AblationRunner().run_axis(AblationAxis.NLI)


async def test_runner_run_axis_planner():
    result = await AblationRunner().run_axis(AblationAxis.PLANNER)
    assert result.axis == AblationAxis.PLANNER
    assert result.metrics_enabled[0].value > result.metrics_disabled[0].value  # more task nodes with a resolved entity


async def test_runner_run_axis_caching():
    result = await AblationRunner().run_axis(AblationAxis.CACHING)
    assert result.metrics_enabled[0].value == 1.0  # repeated key -> perfect hit ratio
    assert result.metrics_disabled[0].value == 0.0  # distinct keys -> no hits


async def test_runner_run_axis_claim_extraction():
    result = await AblationRunner().run_axis(AblationAxis.CLAIM_EXTRACTION)
    assert result.metrics_enabled[0].value > 0
    assert result.metrics_disabled[0].value == 0.0


async def test_runner_run_axis_entity_resolution():
    result = await AblationRunner().run_axis(AblationAxis.ENTITY_RESOLUTION)
    assert result.metrics_enabled[0].value == 1.0  # P-102 merges into known P-101 at the default threshold
    assert result.metrics_disabled[0].value == 0.0  # near-impossible threshold blocks the merge


async def test_runner_run_axis_recommendation_verification():
    result = await AblationRunner().run_axis(AblationAxis.RECOMMENDATION_VERIFICATION)
    assert result.metrics_enabled[0].value == 0.0  # real thresholds reject the low-confidence recommendation
    assert result.metrics_disabled[0].value == 1.0  # permissive thresholds let it through


async def test_runner_run_axis_faithfulness_scoring():
    result = await AblationRunner().run_axis(AblationAxis.FAITHFULNESS_SCORING)
    assert result.metrics_enabled[0].value < result.metrics_disabled[0].value  # real scoring catches the hallucination; blind trust doesn't


async def test_runner_run_axis_confidence_propagation():
    result = await AblationRunner().run_axis(AblationAxis.CONFIDENCE_PROPAGATION)
    assert result.metrics_enabled[0].value >= result.metrics_disabled[0].value  # propagation from a strong neighbor should not lower confidence


async def test_runner_run_axis_ml_prediction():
    result = await AblationRunner().run_axis(AblationAxis.ML_PREDICTION)
    assert 0.0 <= result.metrics_enabled[0].value <= 1.0
    assert 0.0 <= result.metrics_disabled[0].value <= 1.0


async def test_runner_run_axis_explainability():
    result = await AblationRunner().run_axis(AblationAxis.EXPLAINABILITY)
    assert result.metrics_enabled[0].value > 0
    assert result.metrics_disabled[0].value == 0.0


async def test_runner_run_axis_learning_to_rank():
    result = await AblationRunner().run_axis(AblationAxis.LEARNING_TO_RANK)
    assert result.axis == AblationAxis.LEARNING_TO_RANK
    assert result.metrics_enabled[0].name == "mrr"


async def test_runner_run_axis_context_compression():
    result = await AblationRunner().run_axis(AblationAxis.CONTEXT_COMPRESSION)
    assert result.metrics_enabled[0].value < 1.0
    assert result.metrics_disabled[0].value == 1.0


async def test_runner_run_axis_claim_verification():
    result = await AblationRunner().run_axis(AblationAxis.CLAIM_VERIFICATION)
    assert result.axis == AblationAxis.CLAIM_VERIFICATION


async def test_runner_run_axis_nli():
    result = await AblationRunner().run_axis(AblationAxis.NLI)
    assert result.metrics_enabled[0].value >= result.metrics_disabled[0].value


async def test_runner_run_axis_parallel_retrieval():
    result = await AblationRunner().run_axis(AblationAxis.PARALLEL_RETRIEVAL)
    assert result.duration_enabled_seconds < result.duration_disabled_seconds  # parallel fetch beats sequential


async def test_runner_run_axis_graph_intelligence():
    result = await AblationRunner().run_axis(AblationAxis.GRAPH_INTELLIGENCE)
    assert result.axis == AblationAxis.GRAPH_INTELLIGENCE
    assert result.metrics_disabled[0].value == 0.0


async def test_runner_run_axis_vision():
    result = await AblationRunner().run_axis(AblationAxis.VISION)
    assert result.axis == AblationAxis.VISION
    assert 0.0 <= result.metrics_enabled[0].value <= 1.0
