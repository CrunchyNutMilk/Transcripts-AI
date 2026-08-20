"""Layer 5: the time-travel eval and the post-game quiz generator."""
import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.quiz import generate_quiz, render_quiz_markdown
from transcripts_ai.lab.timetravel import (
    render_report,
    run_time_travel,
    session_order_key,
)
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.pipeline import SessionPipeline
from transcripts_ai.schemas import EntityKind, EntityRecord

CAMPAIGN = "heckuva"


def write_session(tmp_path, date, lines):
    path = tmp_path / f"{date}__Test__Transcript_Mapped.md"
    path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return path


class TestTimeTravel:
    def test_ordering_by_embedded_date(self):
        names = ["b-20260110__x.md", "a-20251201__x.md", "c-20260105__x.md"]
        assert [session_order_key(n)[0] for n in sorted(names, key=session_order_key)] == \
            ["20251201", "20260105", "20260110"]

    def test_returning_npc_is_recognised_in_later_sessions(self, tmp_path):
        s1 = write_session(tmp_path, "20260101", [
            "DM: You finally meet Valdrek the lich, and Valdrek laughs.",
            "Jinx: I do not trust Valdrek at all.",
        ])
        s2 = write_session(tmp_path, "20260108", [
            "DM: Valdrek returns, angrier than before, Valdrek is furious.",
            "Jinx: I knew Valdrek would come back.",
        ])
        s3 = write_session(tmp_path, "20260115", [
            "DM: The shadow of Valdrec looms again, Valdrec speaks.",
            "Jinx: Valdrec never stays gone.",
        ])
        steps = run_time_travel([s3, s1, s2], campaign_id=CAMPAIGN,
                                db_path=tmp_path / "work.sqlite",
                                on_progress=lambda *_: None)
        assert [s.session_id for s in steps] == \
            ["2026-01-01", "2026-01-08", "2026-01-15"]
        # session 1: cold memory knows nothing
        assert steps[0].recognised_share == 0.0 or steps[0].candidates == 0
        # session 2: exact return -> recognised with certainty
        assert steps[1].auto_linked >= 1
        assert steps[1].recognised_share == 1.0
        # session 3: misspelled return (Valdrec) -> recognised or suggested
        assert steps[2].candidates >= 1
        assert steps[2].recognised_share > 0.0

    def test_report_renders_curve(self, tmp_path):
        s1 = write_session(tmp_path, "20260101", ["DM: You meet Valdrek."])
        s2 = write_session(tmp_path, "20260108", ["DM: Valdrek returns."])
        steps = run_time_travel([s1, s2], campaign_id=CAMPAIGN,
                                db_path=tmp_path / "w.sqlite",
                                on_progress=lambda *_: None)
        report = render_report(steps, campaign_id=CAMPAIGN)
        assert "Time-travel test" in report and "2026-01-08" in report

    def test_cli_requires_two_transcripts(self, tmp_path):
        one = write_session(tmp_path, "20260101", ["DM: hi"])
        assert lab_main(["time-travel", "--campaign", CAMPAIGN,
                         "--db", str(tmp_path / "w.sqlite"),
                         "--out", str(tmp_path / "r.md"),
                         "--transcripts", str(one)]) == 2


class TestQuiz:
    def _seeded(self, tmp_path):
        transcript = tmp_path / "s1 Mapped.md"
        transcript.write_text(
            "DM: you find a necklace of prayer beads in the chest.\n\n"
            "DM: Gomph takes 24 damage from the blast.\n",
            encoding="utf-8")
        memory = CampaignMemory(tmp_path / "m.sqlite")
        memory.upsert_entity(
            EntityRecord(name="Ghomra", kind=EntityKind.NPC,
                         campaign_id=CAMPAIGN), actor="human:harry")
        SessionPipeline(memory).process_session_native(
            campaign_id=CAMPAIGN, session_id="s1",
            transcript_path=transcript, game_name="G", session_date="s1")
        return memory

    def test_facts_become_questions_with_answers(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            items = generate_quiz(memory, CAMPAIGN, "s1")
            questions = {i.question.question for i in items}
            assert any("What item did the party obtain" in q for q in questions)
            loot = next(i for i in items
                        if "item did the party" in i.question.question)
            assert loot.question.expected == ["Necklace of Prayer Beads"]
            assert loot.source_line is not None
        finally:
            memory.close()

    def test_trick_questions_are_baited_refusals(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            items = generate_quiz(memory, CAMPAIGN, "s1")
            tricks = [i for i in items if i.question.qtype == "trick"]
            assert tricks, "quiz must contain no-invention bait"
            assert all(i.question.expected == [] for i in tricks)
            # a bait question about an entity the session never killed/awarded
            assert any("die this session" in i.question.question
                       or "obtained the" in i.question.question
                       for i in tricks)
        finally:
            memory.close()

    def test_markdown_hides_answers_until_key(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            items = generate_quiz(memory, CAMPAIGN, "s1")
        finally:
            memory.close()
        text = render_quiz_markdown(items, campaign_id=CAMPAIGN,
                                    session_id="s1")
        body, key = text.split("## Answer key", 1)
        assert "Necklace of Prayer Beads" not in body
        assert "Necklace of Prayer Beads" in key

    def test_deterministic_ids(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            first = [i.question.question_id
                     for i in generate_quiz(memory, CAMPAIGN, "s1")]
            second = [i.question.question_id
                      for i in generate_quiz(memory, CAMPAIGN, "s1")]
            assert first == second
        finally:
            memory.close()

    def test_cli_end_to_end(self, tmp_path, capsys):
        memory = self._seeded(tmp_path)
        db = memory.db_path
        memory.close()
        bank = tmp_path / "bank.jsonl"
        md = tmp_path / "quiz.md"
        assert lab_main(["quiz", "--db", db, "--campaign", CAMPAIGN,
                         "--session", "s1", "--out-bank", str(bank),
                         "--out-md", str(md)]) == 0
        assert bank.exists() and "recap trivia" in md.read_text(encoding="utf-8")

    def test_cli_empty_session_fails(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        memory.close()
        assert lab_main(["quiz", "--db", str(tmp_path / "m.sqlite"),
                         "--campaign", CAMPAIGN, "--session", "nope",
                         "--out-md", str(tmp_path / "q.md")]) == 1
