"""End-to-end pipeline tests with scripted fake providers."""
import json

import pytest

from transcripts_ai.memory import CampaignMemory
from transcripts_ai.pipeline import SessionPipeline
from transcripts_ai.providers import FakeProvider, RoleRegistry
from transcripts_ai.schemas import EpistemicStatus

TRANSCRIPT = """\
[00:00:01.000 - 00:00:04.000] - DM: Welcome to the city of Silverspire.
[00:00:05.000 - 00:00:09.000] - DM: A hooded figure says, my name is Daragon.
[00:00:10.000 - 00:00:14.000] - Diego: I think Daragon is secretly the mayor, lol just kidding.
[00:00:15.000 - 00:00:19.000] - DM: Daragon hands you the Moon Sickle.
"""


def extraction_response():
    return json.dumps(
        {
            "facts": [
                {
                    "statement": "The party arrived at Silverspire",
                    "category": "location",
                    "change_type": "introduced",
                    "entities": ["Silverspire"],
                    "status": "strongly_supported",
                    "confidence": 0.95,
                    "quote": "Welcome to the city of Silverspire.",
                    "line_start": 1,
                    "line_end": 1,
                    "speaker": "DM",
                    "importance": "medium",
                },
                {
                    "statement": "Daragon gave the party the Moon Sickle",
                    "category": "loot",
                    "change_type": "awarded",
                    "entities": ["Daragon", "Moon Sickle"],
                    "status": "strongly_supported",
                    "confidence": 0.9,
                    "quote": "Daragon hands you the Moon Sickle.",
                    "line_start": 4,
                    "line_end": 4,
                    "speaker": "DM",
                    "importance": "high",
                },
                {
                    "statement": "Daragon is secretly the mayor",
                    "category": "npc",
                    "change_type": "mentioned",
                    "entities": ["Daragon"],
                    "status": "strongly_supported",  # model overclaims; joke line
                    "confidence": 0.8,
                    "quote": "I think Daragon is secretly the mayor, lol just kidding.",
                    "line_start": 3,
                    "line_end": 3,
                    "speaker": "Diego",
                    "importance": "low",
                },
                {
                    "statement": "A dragon attacked the market",
                    "category": "combat",
                    "change_type": "introduced",
                    "entities": ["dragon"],
                    "status": "strongly_supported",
                    "confidence": 0.9,
                    "quote": "A dragon attacked the market square yesterday.",  # invented
                    "line_start": 2,
                    "line_end": 2,
                    "speaker": "DM",
                    "importance": "high",
                },
            ]
        }
    )


def verification_response(fact_ids):
    verdicts = []
    for fact_id, statement in fact_ids:
        verdict = "supported"
        reason = "quote supports statement"
        if "mayor" in statement:
            verdict = "uncertain"
            reason = "joking tone; not an in-world confirmation"
        verdicts.append(
            {"fact_id": fact_id, "verdict": verdict, "confidence": 0.9, "reason": reason}
        )
    return json.dumps({"verdicts": verdicts})


def summary_response():
    sections = {}
    for title in (
        "Key Events", "Important Dialogue & Revealed Information", "Decisions Made",
        "Plans", "Risks & Unresolved Questions", "Quests", "Loot", "Combat",
        "Vault Update Suggestions",
    ):
        sections[title] = {"confirmed": [], "uncertain": []}
    sections["Key Events"]["confirmed"] = [
        "The party arrived at Silverspire",
        "Zorblax the Destroyer appeared",  # hallucinated name -> must be caught
    ]
    sections["Loot"]["confirmed"] = ["Daragon gave the party the Moon Sickle"]
    return json.dumps(
        {
            "party": ["Diego"],
            "npcs_and_groups": ["Daragon"],
            "locations": ["Silverspire"],
            "sections": sections,
        }
    )


@pytest.fixture
def env(tmp_path):
    memory = CampaignMemory(tmp_path / "m.sqlite")
    transcript = tmp_path / "2026-08-01 - Test - Transcripts Mapped.md"
    transcript.write_text(TRANSCRIPT, encoding="utf-8")
    yield memory, transcript
    memory.close()


class _DynamicVerifier(FakeProvider):
    """Builds a valid verdict list from whatever fact listing it receives."""

    def complete(self, *, system, user, max_tokens=2000, temperature=0.0):
        self.calls.append({"system": system, "user": user})
        import re
        pairs = re.findall(r"fact_id (\w+): \"(.*?)\"", user)
        from transcripts_ai.providers import ProviderResponse
        return ProviderResponse(
            text=verification_response(pairs), model=self.model, provider="fake"
        )


def run_pipeline(memory, transcript, extractor_responses=None, summary_responses=None):
    extractor = FakeProvider(extractor_responses or [extraction_response()])
    verifier = _DynamicVerifier()
    summarizer = FakeProvider(summary_responses or [summary_response()])
    registry = RoleRegistry(
        {}, overrides={
            "extractor": extractor, "verifier": verifier, "summarizer": summarizer,
        },
    )
    pipeline = SessionPipeline(memory, registry)
    report = pipeline.process_session(
        campaign_id="camp-a",
        session_id="2026-08-01",
        transcript_path=transcript,
        game_name="Test Game",
        session_date="2026-08-01",
    )
    return report, extractor, verifier, summarizer


class TestEndToEnd:
    def test_invented_quote_rejected(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(memory, transcript)
        assert report.facts_rejected == 1  # the dragon fact with a fake quote
        statements = [f.statement for f in report.facts_verified]
        assert "A dragon attacked the market" not in statements

    def test_joke_capped_at_table_talk_and_disputed(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(memory, transcript)
        mayor = next(
            f for f in report.facts_disputed if "mayor" in f.statement
        )
        # Deterministic ceiling: joke line can never be strongly supported.
        assert mayor.status.rank >= EpistemicStatus.TABLE_TALK.rank
        assert mayor.needs_review

    def test_verified_facts_reach_memory_with_provenance(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(memory, transcript)
        facts = memory.facts_for_entity("camp-a", "Moon Sickle")
        assert len(facts) == 1
        fact = facts[0]
        assert fact.provenance.quote == "Daragon hands you the Moon Sickle."
        assert fact.provenance.line_start == 4
        assert fact.provenance.context_manifest_hash

    def test_summary_hallucination_gate(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(memory, transcript)
        assert report.summary is not None
        key_events = report.summary.section("Key Events")
        assert "The party arrived at Silverspire" in key_events.confirmed
        assert all("Zorblax" not in c for c in key_events.confirmed)
        assert any("Zorblax" in u for u in key_events.uncertain)
        # And it produced a review item for the hallucinated claim.
        assert any(
            "Zorblax" in i.reason or "Zorblax" in i.subject
            for i in report.review_items
        )

    def test_summary_markdown_separates_confidence(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(memory, transcript)
        assert "**Confirmed:**" in report.summary_markdown
        assert "**Uncertain / unconfirmed:**" in report.summary_markdown
        assert "Initiative order: not stated in transcript" in report.summary_markdown

    def test_resume_skips_done_chunks(self, env):
        memory, transcript = env
        report1, extractor1, *_ = run_pipeline(memory, transcript)
        assert report1.chunks_processed == 1
        report2, extractor2, *_ = run_pipeline(memory, transcript)
        assert report2.chunks_processed == 0
        assert report2.chunks_skipped_resume == 1
        assert extractor2.calls == []  # no paid extraction on resume

    def test_edited_transcript_reprocesses(self, env):
        memory, transcript = env
        run_pipeline(memory, transcript)
        transcript.write_text(
            TRANSCRIPT + "[00:00:20.000 - 00:00:22.000] - DM: You leave the city.\n",
            encoding="utf-8",
        )
        report, extractor, *_ = run_pipeline(memory, transcript)
        assert report.chunks_processed == 1
        assert extractor.calls  # re-extracted because content hash changed

    def test_source_file_never_modified(self, env):
        memory, transcript = env
        before = transcript.read_bytes()
        run_pipeline(memory, transcript)
        assert transcript.read_bytes() == before


class TestFailureRouting:
    def test_invalid_extractor_output_goes_to_review(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(
            memory, transcript,
            extractor_responses=["garbage", "still garbage"],
        )
        assert report.facts_verified == []
        pending = memory.pending_reviews("camp-a")
        assert any("extraction failed validation" in i.subject for i in pending)

    def test_invalid_summary_output_goes_to_review(self, env):
        memory, transcript = env
        report, *_ = run_pipeline(
            memory, transcript,
            summary_responses=["nope", "still nope"],
        )
        assert report.summary is None
        assert any("summary" in w for w in report.warnings)
        pending = memory.pending_reviews("camp-a")
        assert any(i.item_type == "summary_claim" for i in pending)
