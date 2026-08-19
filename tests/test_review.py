import json

import pytest

from transcripts_ai.memory import CampaignMemory
from transcripts_ai.providers import FakeProvider
from transcripts_ai.review import ReviewCoordinator
from transcripts_ai.schemas import (
    EntityKind,
    EpistemicStatus,
    ReviewAction,
    ReviewItem,
)


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "m.sqlite")
    yield mem
    mem.close()


@pytest.fixture
def item(memory):
    review = ReviewItem(
        campaign_id="camp-a", session_id="s1", item_type="spelling",
        subject="Aragon", reason="close to Daragon",
        evidence=["Aragon, it is your turn."],
        suggestions=[{"canonical": "Daragon", "score": 0.9}],
        confidence=0.8,
    )
    memory.enqueue_review(review, actor="engine")
    return review


class TestSixActions:
    def test_correct_learns_alias_and_feedback(self, memory, item):
        coordinator = ReviewCoordinator(memory)
        coordinator.correct(item, canonical="Daragon", actor="human:1")
        assert memory.pending_reviews("camp-a") == []
        alias = memory.resolve_alias("camp-a", "Aragon")
        assert alias.canonical == "Daragon" and alias.human_approved
        feedback = memory.feedback_for("camp-a", "correction_accepted", "Aragon->Daragon")
        assert feedback and feedback[0]["accepted"]

    def test_alias_action(self, memory, item):
        coordinator = ReviewCoordinator(memory)
        coordinator.alias(item, canonical="Daragon", actor="human:1", reason="nickname")
        assert memory.resolve_alias("camp-a", "Aragon").reason == "nickname"

    def test_new_entity_is_canon(self, memory, item):
        coordinator = ReviewCoordinator(memory)
        entity = coordinator.new_entity(item, kind=EntityKind.NPC, actor="human:1")
        assert entity.status is EpistemicStatus.CONFIRMED_CANON
        assert memory.find_entity("camp-a", "Aragon") is not None

    def test_save_for_review_keeps_pending(self, memory, item):
        ReviewCoordinator(memory).save_for_review(item, actor="human:1")
        assert len(memory.pending_reviews("camp-a")) == 1

    def test_dont_know_yet_records_reason(self, memory, item):
        ReviewCoordinator(memory).dont_know_yet(
            item, reason="dm_has_not_said", actor="human:1"
        )
        assert memory.pending_reviews("camp-a") == []
        feedback = memory.feedback_for("camp-a", "deferred", "Aragon")
        assert json.loads(feedback[0]["detail_json"])["reason"] == "dm_has_not_said"

    def test_engine_cannot_decide(self, memory, item):
        coordinator = ReviewCoordinator(memory)
        with pytest.raises(PermissionError):
            coordinator.correct(item, canonical="Daragon", actor="engine")

    def test_reject_suggestion_blocks_reproposal(self, memory, item):
        ReviewCoordinator(memory).reject_suggestion(
            item, canonical="Daragon", actor="human:1"
        )
        assert memory.was_rejected_before(
            "camp-a", "correction_rejected", "Aragon->Daragon"
        )


class TestLetAIPick:
    def test_valid_pick_is_advisory_only(self, memory, item):
        reviewer = FakeProvider([json.dumps(
            {"action": "correct", "choice": "Daragon", "confidence": 0.9,
             "reason": "speaker addresses Daragon on their turn"}
        )])
        recommendation = ReviewCoordinator(memory, reviewer=reviewer).let_ai_pick(item)
        assert recommendation.action is ReviewAction.CORRECT
        assert recommendation.choice == "Daragon"
        assert recommendation.applied is False
        # Nothing changed in memory: still pending, no alias.
        assert len(memory.pending_reviews("camp-a")) == 1
        assert memory.resolve_alias("camp-a", "Aragon") is None

    def test_pick_outside_offered_names_rejected_then_fails_safe(self, memory, item):
        reviewer = FakeProvider([
            json.dumps({"action": "correct", "choice": "Gandalf",
                        "confidence": 0.9, "reason": "x"}),
            json.dumps({"action": "correct", "choice": "Sauron",
                        "confidence": 0.9, "reason": "x"}),
        ])
        recommendation = ReviewCoordinator(memory, reviewer=reviewer).let_ai_pick(item)
        assert recommendation.action is ReviewAction.DONT_KNOW_YET
        assert recommendation.confidence == 0.0

    def test_no_reviewer_falls_back_to_native_learner(self, memory, item):
        recommendation = ReviewCoordinator(memory).let_ai_pick(item)
        # Self-contained advisory pick from the learner — still never applied.
        assert recommendation.applied is False
        assert recommendation.choice == "Daragon"
        assert len(memory.pending_reviews("camp-a")) == 1
