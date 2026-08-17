"""CLI-level tests for the review workflow the runbook relies on."""
import pytest

from transcripts_ai.cli import main
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.schemas import ReviewItem


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "m.sqlite"
    memory = CampaignMemory(path)
    memory.enqueue_review(
        ReviewItem(
            campaign_id="camp", session_id="ingest", item_type="spelling",
            subject="Gomra", reason="variant of Ghomra",
            evidence=["Gomra said hi"],
            suggestions=[{"canonical": "Ghomra", "score": 0.9, "reason": "phonetic"}],
        ),
        actor="engine",
    )
    memory.enqueue_review(
        ReviewItem(
            campaign_id="camp", session_id="ingest", item_type="entity",
            subject="AIDS", reason="detected repeatedly", evidence=[],
        ),
        actor="engine",
    )
    memory.close()
    return str(path)


def run(db, *argv):
    return main(["--db", db, *argv])


class TestResolveCommand:
    def test_correct_by_number(self, db, capsys):
        assert run(db, "resolve", "--campaign", "camp", "--item", "1",
                   "--action", "correct", "--canonical", "Ghomra",
                   "--user", "neill") == 0
        memory = CampaignMemory(db)
        try:
            assert memory.resolve_alias("camp", "Gomra").canonical == "Ghomra"
            assert len(memory.pending_reviews("camp")) == 1
        finally:
            memory.close()

    def test_not_entity_suppresses_future_proposals(self, db):
        memory = CampaignMemory(db)
        items = memory.pending_reviews("camp")
        aids = next(i for i in items if i.subject == "AIDS")
        memory.close()
        assert run(db, "resolve", "--campaign", "camp", "--item", aids.item_id,
                   "--action", "not-entity", "--user", "neill") == 0
        memory = CampaignMemory(db)
        try:
            rows = memory.feedback_for("camp", "entity_misclassified", "AIDS")
            assert rows and rows[-1]["accepted"]
        finally:
            memory.close()

    def test_new_entity_kind_validation(self, db):
        assert run(db, "resolve", "--campaign", "camp", "--item", "2",
                   "--action", "new", "--kind", "nonsense") == 2

    def test_missing_item(self, db):
        assert run(db, "resolve", "--campaign", "camp", "--item", "zzz",
                   "--action", "defer") == 1

    def test_correct_requires_canonical(self, db):
        assert run(db, "resolve", "--campaign", "camp", "--item", "1",
                   "--action", "correct") == 2

    def test_reject_others(self, db):
        assert run(db, "resolve", "--campaign", "camp", "--item", "1",
                   "--action", "correct", "--canonical", "Gomph",
                   "--reject-others", "--user", "neill") == 0
        memory = CampaignMemory(db)
        try:
            # The non-chosen suggestion (Ghomra) was recorded as rejected.
            assert memory.was_rejected_before("camp", "correction_rejected",
                                              "Gomra->Ghomra")
        finally:
            memory.close()


class TestEntitiesCommand:
    def test_lists_entities_and_aliases(self, db, capsys):
        run(db, "resolve", "--campaign", "camp", "--item", "1",
            "--action", "alias", "--canonical", "Ghomra", "--user", "neill")
        assert run(db, "entities", "--campaign", "camp") == 0
        out = capsys.readouterr().out
        assert "Gomra" in out and "Ghomra" in out


class TestReviewsCommand:
    def test_listing_shows_ids_and_hint(self, db, capsys):
        assert run(db, "reviews", "--campaign", "camp") == 0
        out = capsys.readouterr().out
        assert "Gomra" in out and "resolve with:" in out
