"""Tests for the preflight-review fixes: Mapped-only enforcement, mapping
authority, exact-casing preservation, evidence-true initiative, native
Let-AI-Pick."""
import json

import pytest

from transcripts_ai.cli import main
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.native_extractor import _known_name_normaliser, extract_native_facts
from transcripts_ai.review import ReviewCoordinator
from transcripts_ai.schemas import ReviewAction, ReviewItem
from transcripts_ai.transcript import parse_transcript


class TestMappedOnlyGuard:
    def test_process_refuses_unmapped_filenames(self, tmp_path, capsys):
        transcript = tmp_path / "20260809_unmapped.md"
        transcript.write_text("Unknown: hello\n", encoding="utf-8")
        code = main([
            "--db", str(tmp_path / "m.sqlite"), "process",
            "--campaign", "c", "--session", "s", "--game", "g",
            "--transcript", str(transcript),
        ])
        assert code == 3
        assert "REFUSED" in capsys.readouterr().out


class TestMappingWiring:
    def make_mapping(self, tmp_path, campaign="camp"):
        path = tmp_path / "mapping.json"
        path.write_text(json.dumps({
            "campaign": campaign,
            "dm_labels": ["Alex"],
            "players": [
                {"player_id": "1", "player_label": "diego_discord",
                 "character_name": "Nwen'sua"},
            ],
        }), encoding="utf-8")
        return path

    def test_process_uses_mapping(self, tmp_path):
        transcript = tmp_path / "2026-08-09 - Game - Transcripts Mapped.md"
        transcript.write_text("Nwen'sua: I open the door.\nAlex: it creaks.\n",
                              encoding="utf-8")
        db = tmp_path / "m.sqlite"
        code = main([
            "--db", str(db), "process", "--campaign", "camp", "--session", "s",
            "--game", "g", "--transcript", str(transcript),
            "--mapping", str(self.make_mapping(tmp_path)),
        ])
        assert code == 0
        memory = CampaignMemory(db)
        try:
            pc = memory.find_entity("camp", "Nwen'sua")
            assert pc is not None and pc.name == "Nwen'sua"  # exact spelling kept
        finally:
            memory.close()

    def test_campaign_mismatch_rejected(self, tmp_path):
        transcript = tmp_path / "x Transcripts Mapped.md"
        transcript.write_text("A: hi\n", encoding="utf-8")
        with pytest.raises(SystemExit):
            main([
                "--db", str(tmp_path / "m.sqlite"), "process", "--campaign",
                "other", "--session", "s", "--game", "g",
                "--transcript", str(transcript),
                "--mapping", str(self.make_mapping(tmp_path, campaign="camp")),
            ])


class TestExactCasing:
    def test_normaliser_preserves_stored_spelling(self):
        normalise = _known_name_normaliser(frozenset({"Nwen'sua", "Vel'Nadar"}))
        assert normalise("we met nwen'sua and VEL'NADAR") == "we met Nwen'sua and Vel'Nadar"

    def test_lowercase_stored_names_left_alone(self):
        normalise = _known_name_normaliser(frozenset({"weirdname"}))
        assert normalise("Weirdname spoke") == "Weirdname spoke"


class TestInitiativeEvidence:
    def test_quote_is_from_transcript(self):
        text = ("DM: Roll initiative!\n"
                "Diego: Diego rolled a 17.\n")
        parsed = parse_transcript(text)
        facts = extract_native_facts(
            parsed.entries, campaign_id="c", session_id="s",
            source_path="t.md", source_hash=parsed.source_hash,
        )
        fact = next(f for f in facts if f.relationship == "initiative_order")
        assert "Diego rolled a 17." in fact.provenance.quote
        assert fact.provenance.line_start == 2


class TestNativeLetAIPick:
    def test_advisory_pick_with_candidates(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            item = ReviewItem(
                campaign_id="c", session_id="s", item_type="spelling",
                subject="Gamera", reason="variant", evidence=[],
                suggestions=[{"canonical": "Ghomra", "score": 0.8},
                             {"canonical": "Oxwater", "score": 0.5}],
            )
            memory.enqueue_review(item, actor="engine")
            recommendation = ReviewCoordinator(memory).let_ai_pick(item)
            assert recommendation.applied is False
            assert recommendation.choice == "Ghomra"
            assert "learner" in recommendation.reason
            # advisory only: nothing changed
            assert len(memory.pending_reviews("c")) == 1
        finally:
            memory.close()

    def test_no_candidates_says_needs_judgement(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            item = ReviewItem(
                campaign_id="c", session_id="s", item_type="entity",
                subject="Zephyrblade", reason="new?", evidence=[],
            )
            recommendation = ReviewCoordinator(memory).let_ai_pick(item)
            assert recommendation.action is ReviewAction.DONT_KNOW_YET
        finally:
            memory.close()
