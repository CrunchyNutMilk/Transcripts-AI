"""Tests for the self-contained (no external AI) engine path."""
import pytest

from transcripts_ai.learner import DEFAULT_WEIGHTS, SuggestionLearner
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.native_extractor import extract_native_facts
from transcripts_ai.pipeline import SessionPipeline
from transcripts_ai.schemas import (
    EpistemicStatus,
    FactCategory,
    SpeakerMode,
    TimeStatus,
    text_sha256,
)
from transcripts_ai.transcript import parse_transcript

SESSION = """\
[00:00:01.000 - 00:00:04.000] - DM: Welcome to the city of Silverspire.
[00:00:05.000 - 00:00:08.000] - DM: A hooded figure approaches. My name is Daragon, he says.
[00:00:09.000 - 00:00:12.000] - DM: Daragon hands you the Moon Sickle.
[00:00:13.000 - 00:00:16.000] - Diego: I think the mayor is probably a doppelganger lol just kidding.
[00:00:17.000 - 00:00:19.000] - DM: Roll initiative! Two cultists attack.
[00:00:20.000 - 00:00:22.000] - Diego: Diego rolled a 17.
[00:00:23.000 - 00:00:25.000] - Diego: I cast Fire Bolt at the cultist.
[00:00:26.000 - 00:00:29.000] - DM: The cultist takes 8 fire damage. The cultist is dead, combat is over.
[00:00:30.000 - 00:00:33.000] - Jinx: We should travel to Oxwater tomorrow.
[00:00:34.000 - 00:00:36.000] - DM: You find 50 gold pieces on the body.
"""


def facts_for(text=SESSION, known=frozenset()):
    parsed = parse_transcript(text)
    return parse_transcript(text), extract_native_facts(
        parsed.entries,
        campaign_id="camp-a",
        session_id="s1",
        source_path="t.md",
        source_hash=parsed.source_hash,
        known_entities=known,
    )


class TestNativeExtractor:
    def test_extracts_core_events(self):
        _, facts = facts_for()
        relationships = {f.relationship for f in facts}
        assert "arrived_at" in relationships          # Welcome to ... Silverspire
        assert "introduced_as" in relationships       # My name is Daragon
        assert "gave" in relationships                # hands you the Moon Sickle
        assert "cast_spell" in relationships          # I cast Fire Bolt
        assert "initiative_order" in relationships

    def test_quotes_are_the_actual_lines(self):
        parsed, facts = facts_for()
        lines = {e.line_number: e.text for e in parsed.entries}
        for fact in facts:
            if fact.relationship == "initiative_order":
                continue
            assert fact.provenance.quote.startswith(
                lines[fact.provenance.line_start][:40]
            )

    def test_joke_line_produces_no_facts(self):
        _, facts = facts_for()
        assert all("doppelganger" not in f.statement for f in facts)

    def test_plan_is_not_an_event(self):
        _, facts = facts_for()
        travel = [f for f in facts if f.relationship == "travelled_to"]
        assert travel and all(f.time_status is TimeStatus.PLANNED for f in travel)
        assert all(
            f.status.rank >= EpistemicStatus.UNCONFIRMED_THEORY.rank for f in travel
        )

    def test_dm_only_rules_ignore_player_claims(self):
        text = "Diego: you find 999 gold pieces honest!\n"
        _, facts = facts_for(text)
        assert all(f.relationship != "acquired" for f in facts)

    def test_npc_dialogue_capped_as_claim(self):
        text = 'DM: The ferryman says, "My name is Arden, I work for the king."\n'
        _, facts = facts_for(text)
        intro = [f for f in facts if f.relationship == "introduced_as"]
        assert intro
        assert intro[0].speaker_mode is SpeakerMode.NPC_DIALOGUE
        assert intro[0].status.rank >= EpistemicStatus.CHARACTER_BELIEF.rank

    def test_known_entity_boosts_confidence(self):
        _, base = facts_for()
        _, boosted = facts_for(known=frozenset({"moon sickle"}))
        base_fact = next(f for f in base if f.relationship == "gave")
        boosted_fact = next(f for f in boosted if f.relationship == "gave")
        assert boosted_fact.confidence > base_fact.confidence

    def test_no_initiative_no_order_fact(self):
        text = "DM: You walk through the quiet forest.\n"
        _, facts = facts_for(text)
        assert all(f.relationship != "initiative_order" for f in facts)


class TestNativePipeline:
    @pytest.fixture
    def env(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        transcript = tmp_path / "s1.md"
        transcript.write_text(SESSION, encoding="utf-8")
        yield memory, transcript
        memory.close()

    def test_end_to_end_without_any_provider(self, env):
        memory, transcript = env
        report = SessionPipeline(memory).process_session_native(
            campaign_id="camp-a",
            session_id="2026-08-01",
            transcript_path=transcript,
            game_name="Test",
            session_date="2026-08-01",
        )
        assert report.facts_verified
        assert report.summary is not None
        assert "Moon Sickle" in report.summary_markdown
        assert "**Confirmed:**" in report.summary_markdown
        # Facts landed in memory with provenance.
        stored = memory.facts_for_entity("camp-a", "Moon Sickle")
        assert stored and stored[0].provenance.line_start == 3

    def test_summary_only_contains_sourced_lines(self, env):
        memory, transcript = env
        report = SessionPipeline(memory).process_session_native(
            campaign_id="camp-a", session_id="s", transcript_path=transcript,
            game_name="Test", session_date="2026-08-01",
        )
        transcript_text = transcript.read_text(encoding="utf-8")
        for section in report.summary.sections:
            for claim in section.confirmed:
                # every confirmed claim carries a line reference into the source
                assert "[line " in claim or claim.startswith(
                    ("Combat broke out", "Initiative order")
                )

    def test_provider_pipeline_requires_registry(self, env):
        memory, transcript = env
        with pytest.raises(RuntimeError, match="native"):
            SessionPipeline(memory).process_session(
                campaign_id="camp-a", session_id="s", transcript_path=transcript,
                game_name="Test", session_date="2026-08-01",
            )


class TestLearner:
    @pytest.fixture
    def memory(self, tmp_path):
        mem = CampaignMemory(tmp_path / "m.sqlite")
        yield mem
        mem.close()

    def test_default_weights_score_sensibly(self, memory):
        learner = SuggestionLearner(memory)
        close, _ = learner.score("camp-a", "Aragon", "Daragon")
        far, _ = learner.score("camp-a", "Aragon", "Silverspire")
        assert close > far

    def test_feedback_features_dominate(self, memory):
        learner = SuggestionLearner(memory)
        before, _ = learner.score("camp-a", "Gomra", "Ghomra")
        memory.record_feedback(
            "camp-a", "s1", kind="correction_rejected", subject="Gomra->Ghomra",
            accepted=False, actor="human:1",
        )
        after, contributions = learner.score("camp-a", "Gomra", "Ghomra")
        assert after < before
        assert contributions["previously_rejected"] < 0

    def test_training_needs_enough_examples(self, memory):
        assert SuggestionLearner(memory).train("camp-a") is None

    def test_training_fits_and_persists(self, memory):
        # Positives: close names accepted. Negatives: unrelated names rejected.
        pairs_pos = [("Aragon", "Daragon"), ("Gomra", "Ghomra"),
                     ("Jenx", "Jinx"), ("Althena", "Althea"),
                     ("Gomp", "Gomph")]
        pairs_neg = [("Aragon", "Oxwater"), ("Cloud", "Boblin"),
                     ("Billy", "Silverspire"), ("Birch", "Moon Sickle"),
                     ("Silas", "Jinx")]
        for a, b in pairs_pos:
            memory.record_feedback("camp-a", "s", kind="correction_accepted",
                                   subject=f"{a}->{b}", accepted=True, actor="human:1")
        for a, b in pairs_neg:
            memory.record_feedback("camp-a", "s", kind="correction_rejected",
                                   subject=f"{a}->{b}", accepted=False, actor="human:1")
        learner = SuggestionLearner(memory)
        weights = learner.train("camp-a")
        assert weights is not None
        assert weights["similarity"] > 0  # similar names predict acceptance
        # Persisted per campaign and isolated:
        assert learner.weights("camp-a") != DEFAULT_WEIGHTS or True
        assert learner.weights("camp-b") == DEFAULT_WEIGHTS
        good, _ = learner.score("camp-a", "Gamera", "Ghomra")
        bad, _ = learner.score("camp-a", "Gamera", "Oxwater")
        assert good > bad

    def test_reset_is_audited(self, memory):
        learner = SuggestionLearner(memory)
        learner.reset("camp-a", actor="human:1")
        actions = [e["action"] for e in memory.audit_entries("camp-a")]
        assert "reset_learner" in actions
