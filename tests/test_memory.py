import pytest

from transcripts_ai.memory import CampaignMemory, MemoryError_
from transcripts_ai.schemas import (
    AliasRecord,
    ChangeType,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    ReviewAction,
    ReviewItem,
    text_sha256,
)


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "memory.sqlite")
    yield mem
    mem.close()


def make_fact(statement, *, campaign="camp-a", session="2026-08-01", lines=(1, 1),
              status=EpistemicStatus.STRONGLY_SUPPORTED, entities=(), confidence=0.9):
    return Fact(
        statement=statement,
        category=FactCategory.STORY_EVENT,
        change_type=ChangeType.MENTIONED,
        entities=list(entities),
        status=status,
        confidence=confidence,
        provenance=Provenance(
            campaign_id=campaign,
            session_id=session,
            source_path="Transcript Mapped/x.md",
            source_hash=text_sha256("src"),
            line_start=lines[0],
            line_end=lines[1],
            quote=statement,
        ),
    )


class TestCampaignIsolation:
    def test_facts_do_not_leak_between_campaigns(self, memory):
        memory.remember_fact(make_fact("A thing happened", campaign="camp-a"), actor="engine")
        memory.remember_fact(make_fact("Other thing", campaign="camp-b"), actor="engine")
        assert len(memory.facts_for_session("camp-a", "2026-08-01")) == 1
        assert memory.search_facts("camp-a", "Other") == []

    def test_entities_and_aliases_scoped(self, memory):
        memory.upsert_entity(
            EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp-a"),
            actor="engine",
        )
        assert memory.find_entity("camp-b", "Daragon") is None
        memory.add_alias(
            AliasRecord(campaign_id="camp-a", observed="Cloud",
                        canonical="Black Storm Cloud", entity_id=None,
                        approved_by="human:1"),
            actor="human:1",
        )
        assert memory.resolve_alias("camp-b", "Cloud") is None
        assert memory.resolve_alias("camp-a", "Cloud").canonical == "Black Storm Cloud"


class TestEpistemicSafety:
    def test_engine_cannot_store_canon(self, memory):
        with pytest.raises(MemoryError_):
            memory.remember_fact(
                make_fact("X", status=EpistemicStatus.CONFIRMED_CANON), actor="engine"
            )

    def test_human_promotion_path(self, memory):
        fact = memory.remember_fact(make_fact("The king died"), actor="engine")
        with pytest.raises(MemoryError_):
            memory.promote_fact("camp-a", fact.fact_id, actor="engine")
        promoted = memory.promote_fact("camp-a", fact.fact_id, actor="human:1")
        assert promoted.status is EpistemicStatus.CONFIRMED_CANON

    def test_entity_status_never_downgrades(self, memory):
        strong = EntityRecord(
            name="Daragon", kind=EntityKind.NPC, campaign_id="camp-a",
            status=EpistemicStatus.CONFIRMED_CANON,
        )
        memory.upsert_entity(strong, actor="human:1")
        weak = EntityRecord(
            name="Daragon", kind=EntityKind.NPC, campaign_id="camp-a",
            status=EpistemicStatus.PLAYER_ASSUMPTION,
        )
        result = memory.upsert_entity(weak, actor="engine")
        assert result.status is EpistemicStatus.CONFIRMED_CANON


class TestContradictions:
    def test_contradiction_preserves_both_facts(self, memory):
        old = memory.remember_fact(make_fact("The mayor is alive"), actor="engine")
        new = memory.remember_fact(make_fact("The mayor is dead"), actor="engine")
        memory.record_contradiction(
            "camp-a", old.fact_id, new.fact_id, note="death scene", actor="engine"
        )
        assert memory.get_fact("camp-a", old.fact_id).status is EpistemicStatus.CONFLICTING
        assert memory.get_fact("camp-a", new.fact_id) is not None
        assert len(memory.open_contradictions("camp-a")) == 1

    def test_resolution_requires_human_and_sets_states(self, memory):
        old = memory.remember_fact(make_fact("The mayor is alive"), actor="engine")
        new = memory.remember_fact(make_fact("The mayor is dead"), actor="engine")
        cid = memory.record_contradiction(
            "camp-a", old.fact_id, new.fact_id, note="", actor="engine"
        )
        with pytest.raises(MemoryError_):
            memory.resolve_contradiction("camp-a", cid, actor="engine", resolution="x")
        memory.resolve_contradiction(
            "camp-a", cid, actor="human:1", resolution="mayor died in session 12",
            winning_fact_id=new.fact_id,
        )
        assert memory.get_fact("camp-a", new.fact_id).status is EpistemicStatus.CONFIRMED_CANON
        assert memory.get_fact("camp-a", old.fact_id).status is EpistemicStatus.RETCONNED
        assert memory.open_contradictions("camp-a") == []


class TestSearchAndLinks:
    def test_entity_linked_facts(self, memory):
        memory.remember_fact(
            make_fact("Daragon gave the party a map", entities=["Daragon"]), actor="engine"
        )
        facts = memory.facts_for_entity("camp-a", "daragon")
        assert len(facts) == 1

    def test_fts_search(self, memory):
        memory.remember_fact(make_fact("The party looted a Moon Sickle"), actor="engine")
        assert memory.search_facts("camp-a", "sickle")
        assert memory.search_facts("camp-a", '"moon sickle"')

    def test_malformed_fts_query_falls_back(self, memory):
        memory.remember_fact(make_fact("Edge case AND OR NOT"), actor="engine")
        # An unbalanced quote is invalid FTS syntax; lexical fallback must not raise.
        assert memory.search_facts("camp-a", 'edge "') == []


class TestReviewQueueAndFeedback:
    def test_review_lifecycle(self, memory):
        item = ReviewItem(
            campaign_id="camp-a", session_id="s1", item_type="spelling",
            subject="Aragon", reason="probable transcription of Daragon",
            evidence=["Aragon said hello"], confidence=0.6,
        )
        memory.enqueue_review(item, actor="engine")
        assert len(memory.pending_reviews("camp-a")) == 1
        resolved = memory.resolve_review(
            "camp-a", item.item_id, action=ReviewAction.CORRECT,
            actor="human:1", detail={"canonical": "Daragon"},
        )
        assert resolved.resolved
        assert memory.pending_reviews("camp-a") == []

    def test_save_for_review_stays_pending(self, memory):
        item = ReviewItem(
            campaign_id="camp-a", session_id="s1", item_type="entity",
            subject="Mysterious Order", reason="unclear", evidence=[],
        )
        memory.enqueue_review(item, actor="engine")
        memory.resolve_review(
            "camp-a", item.item_id, action=ReviewAction.SAVE_FOR_REVIEW, actor="human:1"
        )
        assert len(memory.pending_reviews("camp-a")) == 1

    def test_rejected_corrections_not_reproposed(self, memory):
        memory.record_feedback(
            "camp-a", "s1", kind="correction_rejected", subject="Aragon->Daragon",
            accepted=False, actor="human:1",
        )
        assert memory.was_rejected_before("camp-a", "correction_rejected", "aragon->daragon")
        memory.record_feedback(
            "camp-a", "s2", kind="correction_rejected", subject="Aragon->Daragon",
            accepted=True, actor="human:1",
        )
        assert not memory.was_rejected_before("camp-a", "correction_rejected", "Aragon->Daragon")

    def test_unknown_feedback_kind_rejected(self, memory):
        with pytest.raises(MemoryError_):
            memory.record_feedback(
                "camp-a", "s1", kind="nonsense", subject="x", accepted=True, actor="human:1"
            )

    def test_verified_dataset_exports_human_rows_only(self, memory):
        memory.record_feedback("camp-a", "s1", kind="alias_added", subject="Cloud",
                               accepted=True, actor="human:1")
        memory.record_feedback("camp-a", "s1", kind="alias_added", subject="Bot",
                               accepted=True, actor="engine")
        data = memory.export_verified_dataset("camp-a")
        assert [d["subject"] for d in data] == ["Cloud"]


class TestReversibility:
    def test_forget_fact_is_audited(self, memory):
        fact = memory.remember_fact(make_fact("Wrong thing"), actor="engine")
        assert memory.forget_fact("camp-a", fact.fact_id, actor="human:1", reason="incorrect")
        assert memory.get_fact("camp-a", fact.fact_id) is None
        assert memory.search_facts("camp-a", "Wrong") == []
        actions = [e["action"] for e in memory.audit_entries("camp-a")]
        assert "forget_fact" in actions and "remember_fact" in actions


class TestFileIndex:
    def test_incremental_scan(self, memory, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "a.md").write_text("alpha", encoding="utf-8")
        (vault / "b.md").write_text("beta", encoding="utf-8")

        changed, deleted = memory.changed_files("camp-a", vault)
        assert {e.path for e in changed} == {"a.md", "b.md"} and deleted == []
        for entry in changed:
            memory.mark_file_indexed("camp-a", entry)

        # Nothing changed -> nothing reported, nothing read.
        changed, deleted = memory.changed_files("camp-a", vault)
        assert changed == [] and deleted == []

        (vault / "a.md").write_text("alpha v2", encoding="utf-8")
        (vault / "b.md").unlink()
        changed, deleted = memory.changed_files("camp-a", vault)
        assert [e.path for e in changed] == ["a.md"]
        assert deleted == ["b.md"]

    def test_touch_without_change_restamps_quietly(self, memory, tmp_path):
        import os
        vault = tmp_path / "vault"
        vault.mkdir()
        f = vault / "a.md"
        f.write_text("alpha", encoding="utf-8")
        changed, _ = memory.changed_files("camp-a", vault)
        memory.mark_file_indexed("camp-a", changed[0])
        os.utime(f, ns=(1, 1))  # mtime changes, content identical
        changed, deleted = memory.changed_files("camp-a", vault)
        assert changed == [] and deleted == []


class TestSummaries:
    def test_previous_summaries_ordering(self, memory):
        memory.remember_summary("camp-a", "2026-07-01", "s1", "h1", actor="engine")
        memory.remember_summary("camp-a", "2026-07-15", "s2", "h2", actor="engine")
        memory.remember_summary("camp-a", "2026-08-01", "s3", "h3", actor="engine")
        prev = memory.previous_summaries("camp-a", before_session="2026-08-01")
        assert [p[0] for p in prev] == ["2026-07-15", "2026-07-01"]
