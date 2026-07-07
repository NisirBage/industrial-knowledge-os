"""Feedback Engine (M4 Component 5): human-in-the-loop recording only.

Every method here writes one structured Feedback record to the Learning
Repository and returns it — nothing here triggers retraining, re-scoring, or
any other automated reaction, per the spec's explicit instruction. A future
training pipeline would read from LearningRepository on its own schedule;
FeedbackEngine's only job is making sure that pipeline has clean, structured
data to read.
"""

from typing import Protocol

from app.feedback.models import Feedback, FeedbackKind
from app.repositories.learning_repository import LearningRepositoryProtocol


class FeedbackProcessor(Protocol):
    async def accept_recommendation(self, recommendation_id: str, equipment_id: str | None = None, reason: str | None = None) -> Feedback: ...
    async def reject_recommendation(self, recommendation_id: str, equipment_id: str | None = None, reason: str | None = None) -> Feedback: ...
    async def modify_recommendation(self, recommendation_id: str, corrected_value: str, equipment_id: str | None = None, reason: str | None = None) -> Feedback: ...
    async def correct_entity(self, entity_id: str, corrected_value: str, reason: str | None = None) -> Feedback: ...
    async def correct_relation(self, relation_id: str, corrected_value: str, reason: str | None = None) -> Feedback: ...
    async def correct_recommendation(self, recommendation_id: str, corrected_value: str, reason: str | None = None) -> Feedback: ...


class FeedbackEngine:
    def __init__(self, learning_repository: LearningRepositoryProtocol):
        self._learning_repository = learning_repository

    async def accept_recommendation(self, recommendation_id: str, equipment_id: str | None = None, reason: str | None = None) -> Feedback:
        return await self._record(FeedbackKind.RECOMMENDATION_ACCEPTED, recommendation_id, equipment_id, reason, None)

    async def reject_recommendation(self, recommendation_id: str, equipment_id: str | None = None, reason: str | None = None) -> Feedback:
        return await self._record(FeedbackKind.RECOMMENDATION_REJECTED, recommendation_id, equipment_id, reason, None)

    async def modify_recommendation(
        self, recommendation_id: str, corrected_value: str, equipment_id: str | None = None, reason: str | None = None
    ) -> Feedback:
        return await self._record(FeedbackKind.RECOMMENDATION_MODIFIED, recommendation_id, equipment_id, reason, corrected_value)

    async def correct_entity(self, entity_id: str, corrected_value: str, reason: str | None = None) -> Feedback:
        return await self._record(FeedbackKind.ENTITY_CORRECTION, entity_id, None, reason, corrected_value)

    async def correct_relation(self, relation_id: str, corrected_value: str, reason: str | None = None) -> Feedback:
        return await self._record(FeedbackKind.RELATION_CORRECTION, relation_id, None, reason, corrected_value)

    async def correct_recommendation(self, recommendation_id: str, corrected_value: str, reason: str | None = None) -> Feedback:
        return await self._record(FeedbackKind.RECOMMENDATION_CORRECTION, recommendation_id, None, reason, corrected_value)

    async def _record(
        self, kind: FeedbackKind, target_id: str, equipment_id: str | None, reason: str | None, corrected_value: str | None
    ) -> Feedback:
        record_id = await self._learning_repository.record_feedback(
            kind=kind.value, target_id=target_id, reason=reason, corrected_value=corrected_value, equipment_id=equipment_id
        )
        return Feedback(id=record_id, kind=kind, target_id=target_id, equipment_id=equipment_id, reason=reason, corrected_value=corrected_value)
