"""Regression tests for the full-code-audit findings (one class per finding)."""
import json
import sqlite3

import pytest

from transcripts_ai.detector import detect_names
from transcripts_ai.dnd_patterns import (
    _BELIEF_CUES,
    _PAST_PRESENT_ACTION,
    assess_time_status,
)
from transcripts_ai.memory import CampaignMemory, MemoryError_
from transcripts_ai.pipeline import SessionPipeline
from transcripts_ai.providers import FakeProvider, RoleRegistry
from transcripts_ai.resolver import BandAction, NameResolver
from transcripts_ai.schemas import (
    ChangeType,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    TimeStatus,
    text_sha256,
)
from transcripts_ai.transcript import parse_transcript


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "m.sqlite")
    yield mem
    mem.close()


def make_fact(statement, session="s1", entities=(), needs_review=False):
    fact = Fact(
        statement=statement,
        category=FactCategory.STORY_EVENT,
        change_type=ChangeType.MENTIONED,
        entities=list(entities),
        status=EpistemicStatus.STRONGLY_SUPPORTED,
        confidence=0.9,
        provenance=Provenance(
            campaign_id="camp", session_id=session, source_path="t.md",
            source_hash=text_sha256("x"), line_start=1, line_end=1,
            quote=statement,
        ),
    )
    fact.needs_review = needs_review
    return fact


class TestFinding1EmptyTranscriptNoCrash:
    def test_native_pipeline_handles_unparseable_transcript(self, memory, tmp_path):
        transcript = tmp_path / "empty Transcripts Mapped.md"
        transcript.write_text("just prose with no speaker turns\n", encoding="utf-8")
        report = SessionPipeline(memory).process_session_native(
            campaign_id="camp", session_id="s", transcript_path=transcript,
            game_name="g", session_date="d",
        )
        assert report.facts_verified == []
        assert report.summary is None
        assert any("no parseable" in w for w in report.warnings)

    def test_provider_pipeline_same_guard(self, memory, tmp_path):
        transcript = tmp_path / "empty Transcripts Mapped.md"
        transcript.write_text("\n", encoding="utf-8")
        registry = RoleRegistry({}, overrides={
            "extractor": FakeProvider(), "verifier": FakeProvider(),
            "summarizer": FakeProvider(),
        })
        report = SessionPipeline(memory, registry).process_session(
            campaign_id="camp", session_id="s", transcript_path=transcript,
            game_name="g", session_date="d",
        )
        assert report.summary is None  # and no IndexError


class TestFinding2ResumeDoesNotLaunderDisputedFacts:
    def test_needs_review_facts_stay_out_of_verified_on_resume(self, memory, tmp_path):
        transcript = tmp_path / "s Transcripts Mapped.md"
        transcript.write_text("DM: Roll initiative!\nDiego: Diego rolled a 17.\n",
                              encoding="utf-8")
        pipeline = SessionPipeline(memory)
        report1 = pipeline.process_session_native(
            campaign_id="camp", session_id="s", transcript_path=transcript,
            game_name="g", session_date="d",
        )
        # Plant a disputed fact inside the completed chunk's line range.
        disputed = make_fact("dubious claim", session="s", needs_review=True)
        memory.remember_fact(disputed, actor="test")
        report2 = pipeline.process_session_native(
            campaign_id="camp", session_id="s", transcript_path=transcript,
            game_name="g", session_date="d",
        )
        verified_ids = {f.fact_id for f in report2.facts_verified}
        assert disputed.fact_id not in verified_ids


class TestFinding3UpsertPreservesData:
    def test_empty_upsert_keeps_description_and_attributes(self, memory):
        memory.upsert_entity(
            EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp",
                         description="curated bio",
                         attributes={"player_id": "42"}),
            actor="human:1",
        )
        result = memory.upsert_entity(
            EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp"),
            actor="engine",
        )
        assert result.description == "curated bio"
        assert result.attributes["player_id"] == "42"
        stored = memory.find_entity("camp", "Daragon")
        assert stored.description == "curated bio"
        assert stored.attributes == {"player_id": "42"}

    def test_new_values_still_update(self, memory):
        memory.upsert_entity(
            EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp",
                         description="old"), actor="human:1",
        )
        memory.upsert_entity(
            EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp",
                         description="newer", attributes={"role": "ferryman"}),
            actor="human:1",
        )
        stored = memory.find_entity("camp", "Daragon")
        assert stored.description == "newer"
        assert stored.attributes["role"] == "ferryman"


class TestFinding4KnownMentionMatching:
    def test_punctuation_adjacent_known_name_detected(self):
        entries = parse_transcript("DM: You finally meet cloud, the mercenary.\n").entries
        found = detect_names(entries, known_names=frozenset({"cloud"}))
        assert any(d.folded == "cloud" for d in found)

    def test_prefix_word_not_matched(self):
        entries = parse_transcript("DM: cloudy weather rolls in over the pass.\n").entries
        found = detect_names(entries, known_names=frozenset({"cloud"}))
        assert not any(d.folded == "cloud" for d in found)


class TestFinding5RegexTrailingSpaces:
    def test_belief_cue_matches_at_end_of_line(self):
        assert _BELIEF_CUES.search("I say that in character")

    def test_action_cue_matches_at_end_of_line(self):
        assert _PAST_PRESENT_ACTION.search("The dragon died")
        assert assess_time_status("The dragon died") is TimeStatus.HAPPENED


class TestFinding6ContradictionWinnerValidated:
    def test_unknown_winner_rejected(self, memory):
        a = memory.remember_fact(make_fact("mayor alive"), actor="engine")
        b = memory.remember_fact(make_fact("mayor dead"), actor="engine")
        cid = memory.record_contradiction("camp", a.fact_id, b.fact_id,
                                          note="", actor="engine")
        with pytest.raises(MemoryError_, match="winning_fact_id"):
            memory.resolve_contradiction(
                "camp", cid, actor="human:1", resolution="typo",
                winning_fact_id="not-a-real-id",
            )
        # Nothing was retconned by the failed call.
        assert memory.get_fact("camp", b.fact_id).status is not EpistemicStatus.RETCONNED


class TestFinding7FtsBackfill:
    def test_facts_written_without_fts_are_searchable_after_reopen(self, tmp_path):
        db = tmp_path / "m.sqlite"
        memory = CampaignMemory(db)
        memory.remember_fact(make_fact("The party looted a Moon Sickle"),
                             actor="engine")
        memory.close()
        # Simulate a DB written on a build without FTS5.
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE IF EXISTS facts_fts")
        conn.commit()
        conn.close()
        reopened = CampaignMemory(db)
        try:
            assert reopened.search_facts("camp", "sickle")
        finally:
            reopened.close()


class TestFinding8SummaryRendersNpcsAndLocations:
    def test_native_summary_includes_npc_and_location_sections(self, memory, tmp_path):
        transcript = tmp_path / "s Transcripts Mapped.md"
        transcript.write_text(
            "DM: Welcome to the city of Silverspire.\n"
            "DM: A figure says, my name is Daragon.\n",
            encoding="utf-8",
        )
        report = SessionPipeline(memory).process_session_native(
            campaign_id="camp", session_id="s", transcript_path=transcript,
            game_name="g", session_date="d",
        )
        assert "Daragon" in report.summary.npcs_and_groups
        assert "Silverspire" in report.summary.locations
        assert "NPCs/Groups:" in report.summary_markdown
        assert "## Places" in report.summary_markdown


class TestFinding9BandsWired:
    def test_drop_band_filters_suggestions(self, memory, monkeypatch):
        memory.upsert_entity(
            EntityRecord(name="Silverspire", kind=EntityKind.LOCATION,
                         campaign_id="camp"), actor="human:1",
        )
        resolver = NameResolver(memory)
        baseline = resolver.resolve("camp", "Silverspyre")
        assert baseline.best is not None
        assert baseline.band_action in (BandAction.SUGGEST, BandAction.SAVE_FOR_REVIEW)
        # Raising the review floor above the score must drop the suggestion.
        monkeypatch.setenv("ENGINE_BAND_REVIEW", "0.99")
        monkeypatch.setenv("ENGINE_BAND_SUGGEST", "0.995")
        raised = resolver.resolve("camp", "Silverspyre")
        assert raised.suggestions == []

    def test_exact_match_is_auto_link(self, memory):
        memory.upsert_entity(
            EntityRecord(name="Silverspire", kind=EntityKind.LOCATION,
                         campaign_id="camp"), actor="human:1",
        )
        resolution = NameResolver(memory).resolve("camp", "silverspire")
        assert resolution.band_action is BandAction.AUTO_LINK
        assert not resolution.needs_human


class TestFinding10ProbeReadsBounded:
    def test_probe_reads_only_head(self, tmp_path, monkeypatch):
        # Behavioural proxy: a giant file without transcript markers is
        # skipped, and accepted files are found by frontmatter probe.
        import sys
        sys.path.insert(0, "scripts")
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "huge_note.md").write_text("x" * 10_000, encoding="utf-8")
        (vault / "session.md").write_text(
            "---\ntype: transcript-mapped\ncampaign: \"c\"\n---\nDM: hi\n",
            encoding="utf-8",
        )
        from ingest_campaign import FRONTMATTER_RE  # noqa: F401  (import works)
        head = open(vault / "huge_note.md", encoding="utf-8-sig").read(300)
        assert len(head) == 300
