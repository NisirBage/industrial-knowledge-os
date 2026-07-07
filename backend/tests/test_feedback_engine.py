from app.feedback.engine import FeedbackEngine
from app.feedback.models import FeedbackKind


class _RecordingLearningRepository:
    def __init__(self):
        self.records: list[dict] = []

    async def record_feedback(self, kind, target_id, reason, corrected_value, equipment_id=None) -> str:
        self.records.append(
            {"kind": kind, "target_id": target_id, "reason": reason, "corrected_value": corrected_value, "equipment_id": equipment_id}
        )
        return f"fb-{len(self.records)}"


async def test_accept_recommendation_records_correct_kind():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.accept_recommendation("rec-1", equipment_id="P-101", reason="looks right")

    assert feedback.kind == FeedbackKind.RECOMMENDATION_ACCEPTED
    assert repo.records[0]["kind"] == "recommendation_accepted"
    assert repo.records[0]["equipment_id"] == "P-101"


async def test_reject_recommendation_records_correct_kind():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.reject_recommendation("rec-1", reason="not applicable")

    assert feedback.kind == FeedbackKind.RECOMMENDATION_REJECTED
    assert repo.records[0]["reason"] == "not applicable"


async def test_modify_recommendation_carries_corrected_value():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.modify_recommendation("rec-1", corrected_value="Use torque wrench instead")

    assert feedback.corrected_value == "Use torque wrench instead"
    assert feedback.kind == FeedbackKind.RECOMMENDATION_MODIFIED


async def test_correct_entity_records_entity_correction():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.correct_entity("entity-1", corrected_value="Valve", reason="mislabeled as Pump")

    assert feedback.kind == FeedbackKind.ENTITY_CORRECTION
    assert repo.records[0]["target_id"] == "entity-1"


async def test_correct_relation_records_relation_correction():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.correct_relation("relation-1", corrected_value="CONNECTED_TO")

    assert feedback.kind == FeedbackKind.RELATION_CORRECTION


async def test_correct_recommendation_records_recommendation_correction():
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    feedback = await engine.correct_recommendation("rec-1", corrected_value="Schedule for next week, not today")

    assert feedback.kind == FeedbackKind.RECOMMENDATION_CORRECTION


async def test_feedback_engine_never_triggers_retraining_side_effects():
    """No retraining hook exists to call — this test documents that
    FeedbackEngine's public surface is exactly the six recording methods."""
    repo = _RecordingLearningRepository()
    engine = FeedbackEngine(repo)

    public_methods = {name for name in dir(engine) if not name.startswith("_")}
    assert public_methods == {
        "accept_recommendation",
        "reject_recommendation",
        "modify_recommendation",
        "correct_entity",
        "correct_relation",
        "correct_recommendation",
    }
