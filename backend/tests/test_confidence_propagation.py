import pytest

from app.claims.models import Claim, ClaimGraph, ClaimRelation, ClaimType, ClaimVerificationStatus
from app.claims.propagation import (
    BayesianConfidencePropagator,
    GraphDecayConfidencePropagator,
    PropagationConfig,
)


def _claim(
    claim_id: str,
    confidence: float = 0.5,
    status: ClaimVerificationStatus = ClaimVerificationStatus.UNKNOWN,
    support_count: int = 0,
    contradiction_count: int = 0,
) -> Claim:
    return Claim(
        id=claim_id,
        claim_type=ClaimType.FACT,
        text=claim_id,
        subject=claim_id,
        predicate="is_a",
        confidence=confidence,
        verification_status=status,
        support_count=support_count,
        contradiction_count=contradiction_count,
    )


def test_isolated_claim_propagated_confidence_equals_local_confidence():
    claim = _claim("c1", confidence=0.5, status=ClaimVerificationStatus.SUPPORTED, support_count=2)
    graph = ClaimGraph(claims=[claim], relations=[])
    propagator = GraphDecayConfidencePropagator()

    result = propagator.propagate(graph)
    updated = result.claim_by_id("c1")

    # local = 0.5 * 1.15 (SUPPORTED) + min(0.15, 2*0.03) = 0.575 + 0.06 = 0.635
    assert updated.propagated_confidence == pytest.approx(0.635, abs=1e-4)


def test_contradicted_claim_has_lower_local_confidence_than_supported():
    supported = _claim("s1", confidence=0.6, status=ClaimVerificationStatus.SUPPORTED)
    contradicted = _claim("c1", confidence=0.6, status=ClaimVerificationStatus.CONTRADICTED, contradiction_count=1)
    graph = ClaimGraph(claims=[supported, contradicted], relations=[])

    result = GraphDecayConfidencePropagator().propagate(graph)

    assert result.claim_by_id("c1").propagated_confidence < result.claim_by_id("s1").propagated_confidence


def test_connected_claim_confidence_rises_from_a_well_supported_neighbor():
    strong = _claim("strong", confidence=0.9, status=ClaimVerificationStatus.SUPPORTED, support_count=3)
    weak = _claim("weak", confidence=0.2, status=ClaimVerificationStatus.UNKNOWN)
    graph = ClaimGraph(
        claims=[strong, weak],
        relations=[ClaimRelation(id="r1", source_claim_id="strong", target_claim_id="weak", relation_type="RELATED_TO")],
    )

    baseline = GraphDecayConfidencePropagator()._local_confidence(weak)

    propagated = GraphDecayConfidencePropagator().propagate(graph)
    weak_result = propagated.claim_by_id("weak")

    assert weak_result.propagated_confidence > baseline


def test_disconnected_claims_do_not_influence_each_other():
    a = _claim("a", confidence=0.9, status=ClaimVerificationStatus.SUPPORTED, support_count=3)
    b = _claim("b", confidence=0.1, status=ClaimVerificationStatus.UNKNOWN)
    graph = ClaimGraph(claims=[a, b], relations=[])

    result = GraphDecayConfidencePropagator().propagate(graph)
    b_result = result.claim_by_id("b")
    expected_local = GraphDecayConfidencePropagator()._local_confidence(b)

    assert b_result.propagated_confidence == pytest.approx(expected_local, abs=1e-4)


def test_propagated_confidence_respects_configured_bounds():
    claim = _claim("c1", confidence=1.0, status=ClaimVerificationStatus.SUPPORTED, support_count=10)
    graph = ClaimGraph(claims=[claim], relations=[])
    config = PropagationConfig(max_confidence=0.8)

    result = GraphDecayConfidencePropagator().propagate(graph, config)
    assert result.claim_by_id("c1").propagated_confidence <= 0.8


def test_zero_decay_factor_produces_no_propagation_beyond_local():
    a = _claim("a", confidence=0.9, status=ClaimVerificationStatus.SUPPORTED, support_count=3)
    b = _claim("b", confidence=0.1, status=ClaimVerificationStatus.UNKNOWN)
    graph = ClaimGraph(
        claims=[a, b],
        relations=[ClaimRelation(id="r1", source_claim_id="a", target_claim_id="b", relation_type="RELATED_TO")],
    )
    config = PropagationConfig(decay_factor=0.0)

    result = GraphDecayConfidencePropagator().propagate(graph, config)
    b_result = result.claim_by_id("b")
    expected_local = GraphDecayConfidencePropagator()._local_confidence(b)

    assert b_result.propagated_confidence == pytest.approx(expected_local, abs=1e-4)


def test_bayesian_propagator_is_a_documented_unimplemented_extension_point():
    with pytest.raises(NotImplementedError):
        BayesianConfidencePropagator().propagate(ClaimGraph(claims=[], relations=[]))


def test_multi_hop_propagation_reaches_second_degree_neighbor():
    a = _claim("a", confidence=0.9, status=ClaimVerificationStatus.SUPPORTED, support_count=3)
    b = _claim("b", confidence=0.5, status=ClaimVerificationStatus.UNKNOWN)
    c = _claim("c", confidence=0.1, status=ClaimVerificationStatus.UNKNOWN)
    graph = ClaimGraph(
        claims=[a, b, c],
        relations=[
            ClaimRelation(id="r1", source_claim_id="a", target_claim_id="b", relation_type="RELATED_TO"),
            ClaimRelation(id="r2", source_claim_id="b", target_claim_id="c", relation_type="RELATED_TO"),
        ],
    )
    baseline_c = GraphDecayConfidencePropagator()._local_confidence(c)

    result = GraphDecayConfidencePropagator().propagate(graph, PropagationConfig(max_hops=3))

    assert result.claim_by_id("c").propagated_confidence > baseline_c


def test_max_hops_limits_how_far_a_single_origins_influence_travels():
    """Every claim spreads its own local confidence outward independently
    (not a single BFS from one source), so "c" still receives a 1-hop bump
    from "b" even with max_hops=1 — what max_hops=1 actually prevents is
    "a"'s influence reaching 2 hops to "c". Comparing against max_hops=3
    (which lets "a" reach "c" too) isolates that specific effect."""
    def _graph() -> ClaimGraph:
        return ClaimGraph(
            claims=[
                _claim("a", confidence=0.9, status=ClaimVerificationStatus.SUPPORTED, support_count=3),
                _claim("b", confidence=0.5, status=ClaimVerificationStatus.UNKNOWN),
                _claim("c", confidence=0.1, status=ClaimVerificationStatus.UNKNOWN),
            ],
            relations=[
                ClaimRelation(id="r1", source_claim_id="a", target_claim_id="b", relation_type="RELATED_TO"),
                ClaimRelation(id="r2", source_claim_id="b", target_claim_id="c", relation_type="RELATED_TO"),
            ],
        )

    result_one_hop = GraphDecayConfidencePropagator().propagate(_graph(), PropagationConfig(max_hops=1))
    result_three_hops = GraphDecayConfidencePropagator().propagate(_graph(), PropagationConfig(max_hops=3))

    assert result_one_hop.claim_by_id("c").propagated_confidence < result_three_hops.claim_by_id("c").propagated_confidence


def test_propagate_returns_new_graph_original_unmodified():
    claim = _claim("c1", confidence=0.5)
    graph = ClaimGraph(claims=[claim], relations=[])

    result = GraphDecayConfidencePropagator().propagate(graph)

    assert result is not graph
    assert graph.claims[0].propagated_confidence is None
    assert result.claims[0].propagated_confidence is not None
