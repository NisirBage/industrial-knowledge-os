from app.schemas.chat import ClaimOut, FaithfulnessOut, RecommendationOut, SentenceAttributionOut, SentenceFaithfulnessOut


def _recommendation_kwargs() -> dict:
    return dict(
        id="r1", description="d", priority="high", severity="high", estimated_impact="i",
        required_personnel=[], required_equipment=[], required_documents=[], estimated_downtime="TBD",
        confidence=0.8, justification="j", supporting_evidence=["c1"],
    )


def test_recommendation_out_defaults_verification_fields_to_unknown_and_empty():
    recommendation = RecommendationOut(**_recommendation_kwargs())
    assert recommendation.verification_status == "unknown"
    assert recommendation.verification_reasons == []


def test_recommendation_out_accepts_explicit_verification_fields():
    recommendation = RecommendationOut(**_recommendation_kwargs(), verification_status="verified", verification_reasons=["ok"])
    assert recommendation.verification_status == "verified"
    assert recommendation.verification_reasons == ["ok"]


def test_claim_out_defaults_optional_fields():
    claim = ClaimOut(id="c1", claim_type="fact", text="t", subject="s", predicate="p", confidence=0.5, verification_status="unknown")
    assert claim.object == ""
    assert claim.propagated_confidence is None
    assert claim.source_evidence_ids == []
    assert claim.support_count == 0
    assert claim.contradiction_count == 0


def test_sentence_faithfulness_out_shape():
    sentence = SentenceFaithfulnessOut(sentence="a", entailed=True, best_claim_id="c1", score=0.9)
    assert sentence.sentence == "a"
    assert sentence.entailed is True


def test_sentence_faithfulness_out_best_claim_id_defaults_to_none():
    sentence = SentenceFaithfulnessOut(sentence="a", entailed=False, score=0.0)
    assert sentence.best_claim_id is None


def test_faithfulness_out_requires_all_core_fields():
    faithfulness = FaithfulnessOut(
        overall_faithfulness=0.8, sentence_faithfulness=[], recommendation_faithfulness=1.0,
        graph_faithfulness=1.0, claim_coverage=1.0, unsupported_claim_count=0,
        hallucinated_claim_count=0, missing_evidence_count=0, contradiction_rate=0.0, summary="s",
    )
    assert faithfulness.overall_faithfulness == 0.8
    assert faithfulness.summary == "s"


def test_claim_out_accepts_full_set_of_optional_fields():
    claim = ClaimOut(
        id="c1", claim_type="risk", text="t", subject="s", predicate="p", object="pump",
        confidence=0.7, propagated_confidence=0.8, source_evidence_ids=["c1", "c2"],
        verification_status="supported", verification_notes="notes", support_count=2, contradiction_count=1,
    )
    assert claim.object == "pump"
    assert claim.propagated_confidence == 0.8
    assert claim.source_evidence_ids == ["c1", "c2"]
    assert claim.support_count == 2
    assert claim.contradiction_count == 1


def test_faithfulness_out_accepts_populated_sentence_faithfulness_list():
    faithfulness = FaithfulnessOut(
        overall_faithfulness=0.5, sentence_faithfulness=[SentenceFaithfulnessOut(sentence="a", entailed=True, score=0.9)],
        recommendation_faithfulness=0.5, graph_faithfulness=0.5, claim_coverage=0.5,
        unsupported_claim_count=1, hallucinated_claim_count=1, missing_evidence_count=1, contradiction_rate=0.1, summary="s",
    )
    assert len(faithfulness.sentence_faithfulness) == 1
    assert faithfulness.sentence_faithfulness[0].sentence == "a"


def test_sentence_attribution_out_defaults():
    attribution = SentenceAttributionOut(sentence="a")
    assert attribution.supporting_claim_ids == []
    assert attribution.supporting_evidence_ids == []
    assert attribution.graph_node_ids == []
    assert attribution.graph_edge_descriptions == []
    assert attribution.confidence == 0.0
    assert attribution.reasoning_path == ""
