import pytest

from app.research.ab_testing.comparator import ABComparator


def test_compare_identifies_winner_by_mean():
    result = ABComparator().compare("a", "b", "f1", [0.5, 0.6, 0.55], [0.8, 0.85, 0.82])
    assert result.winner == "b"
    assert result.mean_b > result.mean_a


def test_compare_tie_when_means_equal():
    result = ABComparator().compare("a", "b", "f1", [0.5, 0.5], [0.5, 0.5])
    assert result.winner == "tie"
    assert result.p_value == 1.0


def test_compare_raises_on_unequal_length_lists():
    with pytest.raises(ValueError):
        ABComparator().compare("a", "b", "f1", [0.5], [0.5, 0.6])


def test_compare_raises_on_empty_lists():
    with pytest.raises(ValueError):
        ABComparator().compare("a", "b", "f1", [], [])


def test_compare_mean_delta_is_a_minus_b():
    result = ABComparator().compare("a", "b", "f1", [0.9, 0.9], [0.5, 0.5])
    assert result.mean_delta == pytest.approx(0.4)


def test_compare_large_consistent_difference_yields_low_p_value():
    arm_a = [0.9] * 10
    arm_b = [0.1] * 10
    result = ABComparator().compare("a", "b", "f1", arm_a, arm_b)
    assert result.p_value < 0.05


def test_compare_identical_arms_yields_p_value_one():
    scores = [0.7, 0.6, 0.8, 0.5]
    result = ABComparator().compare("a", "b", "f1", scores, list(scores))
    assert result.p_value == 1.0
    assert result.t_statistic == 0.0


def test_compare_preserves_raw_scores_in_result():
    result = ABComparator().compare("a", "b", "f1", [0.1, 0.2], [0.3, 0.4])
    assert result.arm_a_scores == [0.1, 0.2]
    assert result.arm_b_scores == [0.3, 0.4]
