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


class TestAuditHardeningLayer5:
    """Regressions from the Layer 5 adversarial review."""

    def test_epoch_and_uuid_prefixes_never_outrank_real_dates(self):
        from transcripts_ai.lab.timetravel import session_date_of
        assert session_date_of("craig-1699843200-session_20231113.txt") == "20231113"
        assert session_date_of("12345678-90ab_20220505.md") == "20220505"
        assert session_date_of("recording-000000012-20240105.md") == "20240105"
        assert session_date_of("no-date-here.md") is None
        names = ["craig-1699843200-x_20231113.md", "a_20240105.md",
                 "b_20220505.md"]
        ordered = sorted(names, key=session_order_key)
        assert ordered == ["b_20220505.md", "craig-1699843200-x_20231113.md",
                           "a_20240105.md"]

    def test_zero_candidates_renders_na_never_100(self, tmp_path):
        from transcripts_ai.lab.timetravel import StepMetrics, render_report
        quiet = StepMetrics(session_id="s", candidates=0, auto_linked=0,
                            suggested=0, unknown=0, known_mentions=9)
        assert quiet.recognised_share is None
        report = render_report([quiet] * 5, campaign_id=CAMPAIGN)
        assert "n/a" in report and "100%" not in report

    def test_existing_db_is_refused(self, tmp_path):
        db = tmp_path / "real_campaign.sqlite"
        CampaignMemory(db).close()          # simulate a precious existing DB
        s1 = write_session(tmp_path, "20260101", ["DM: hello Valdrek."])
        s2 = write_session(tmp_path, "20260108", ["DM: Valdrek returns."])
        with pytest.raises(ValueError, match="FRESH"):
            run_time_travel([s1, s2], campaign_id=CAMPAIGN, db_path=db,
                            on_progress=lambda *_: None)

    def test_oracle_writes_never_masquerade_as_human(self, tmp_path):
        """Invariant: eval simulation must not contaminate human-only
        consumers (questions_from_feedback, dpo_pairs, learner)."""
        from transcripts_ai.lab.scorecard import questions_from_feedback
        s1 = write_session(tmp_path, "20260101", [
            "DM: You meet Valdrek the lich, and Valdrek laughs."])
        s2 = write_session(tmp_path, "20260108", [
            "DM: Valdrec returns, and Valdrec is furious."])
        db = tmp_path / "work.sqlite"
        run_time_travel([s1, s2], campaign_id=CAMPAIGN, db_path=db,
                        on_progress=lambda *_: None)
        memory = CampaignMemory(db)
        try:
            for row in memory.all_feedback(CAMPAIGN):
                assert not str(row["actor"]).startswith("human:"), row
            for alias in memory.aliases(CAMPAIGN):
                assert not alias.approved_by.startswith("human:"), alias
            assert questions_from_feedback(memory, CAMPAIGN) == []
        finally:
            memory.close()

    def test_cli_directory_walk_is_recursive_and_date_filtered(
            self, tmp_path, capsys):
        nested = tmp_path / "01 - Transcript Mapped" / "2026-01-01 - Session"
        nested.mkdir(parents=True)
        (nested / "20260101 - Transcript Mapped.md").write_text(
            "DM: hi Valdrek\n", encoding="utf-8")
        (tmp_path / "01 - Transcript Mapped" / "Index.md").write_text(
            "not a session", encoding="utf-8")
        assert lab_main(["time-travel", "--campaign", CAMPAIGN,
                         "--db", str(tmp_path / "w.sqlite"),
                         "--out", str(tmp_path / "r.md"),
                         "--transcripts", str(tmp_path)]) == 2
        out = capsys.readouterr().out
        assert "skipped 1 .md file(s)" in out      # Index.md filtered
        assert "at least two" in out               # only one dated file found


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
            # a bait question about an entity the record never killed/awarded
            assert any("die?" in i.question.question
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

    def test_damage_answer_is_the_adjacent_number_only(self, tmp_path):
        transcript = tmp_path / "s1 Mapped.md"
        transcript.write_text(
            "DM: Gomph rolled a 12 and Gomph takes 24 damage from the blast.\n",
            encoding="utf-8")
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            SessionPipeline(memory).process_session_native(
                campaign_id=CAMPAIGN, session_id="s1",
                transcript_path=transcript, game_name="G", session_date="s1")
            items = generate_quiz(memory, CAMPAIGN, "s1")
            damage = [i for i in items if "damage" in i.question.question]
            if damage:                       # never '1224'
                assert damage[0].question.expected == ["24"]
            for item in items:
                assert "1224" not in item.question.expected
        finally:
            memory.close()

    def test_dead_in_earlier_session_is_never_bait(self, tmp_path):
        from transcripts_ai.schemas import (ChangeType, EntityRecord, Fact,
                                            FactCategory, EpistemicStatus,
                                            Provenance, text_sha256)
        memory = self._seeded(tmp_path)
        try:
            memory.upsert_entity(
                EntityRecord(name="Boblin", kind=EntityKind.NPC,
                             campaign_id=CAMPAIGN), actor="human:harry")
            memory.remember_fact(Fact(
                statement="Boblin was killed by the Elite Honor Guard",
                category=FactCategory.DEATH_OR_STATUS,
                change_type=ChangeType.UPDATED,
                entities=["Boblin"],
                status=EpistemicStatus.STRONGLY_SUPPORTED,
                confidence=0.9,
                provenance=Provenance(
                    campaign_id=CAMPAIGN, session_id="s0",
                    source_path="s0.md", source_hash=text_sha256("x"),
                    line_start=1, line_end=1,
                    quote="Boblin was killed")), actor="engine")
            items = generate_quiz(memory, CAMPAIGN, "s1")
            assert not [i for i in items
                        if "Boblin" in i.question.question], \
                "a character with a recorded death must never be bait"
        finally:
            memory.close()

    def test_duplicate_question_texts_merge_answers(self, tmp_path):
        transcript = tmp_path / "s1 Mapped.md"
        transcript.write_text(
            "DM: you find a necklace of prayer beads in the chest.\n\n"
            "DM: you find a wand of fear under the throne.\n",
            encoding="utf-8")
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            SessionPipeline(memory).process_session_native(
                campaign_id=CAMPAIGN, session_id="s1",
                transcript_path=transcript, game_name="G", session_date="s1")
            items = generate_quiz(memory, CAMPAIGN, "s1")
            loot_questions = [i for i in items
                              if "item did the party obtain" in i.question.question]
            assert len(loot_questions) == 1        # merged, not contradictory
            expected = {e.casefold() for e in loot_questions[0].question.expected}
            assert {"necklace of prayer beads", "wand of fear"} <= expected
        finally:
            memory.close()

    def test_bait_never_names_a_real_answer(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            items = generate_quiz(memory, CAMPAIGN, "s1")
            real_answers = {a.casefold() for i in items
                            if i.question.qtype != "trick"
                            for a in i.question.expected}
            for item in items:
                if item.question.qtype == "trick":
                    for answer in real_answers:
                        assert answer not in item.question.question.casefold()
        finally:
            memory.close()

    def test_count_one_keeps_a_real_question(self, tmp_path):
        memory = self._seeded(tmp_path)
        try:
            items = generate_quiz(memory, CAMPAIGN, "s1", count=1)
            assert items and items[0].question.qtype != "trick"
        finally:
            memory.close()

    def test_cli_template_miss_message(self, tmp_path, capsys):
        # a session with facts that fit no quiz template
        transcript = tmp_path / "s1 Mapped.md"
        transcript.write_text(
            "DM: The party took a long rest by the fire.\n", encoding="utf-8")
        memory = CampaignMemory(tmp_path / "m.sqlite")
        SessionPipeline(memory).process_session_native(
            campaign_id=CAMPAIGN, session_id="s1",
            transcript_path=transcript, game_name="G", session_date="s1")
        memory.close()
        code = lab_main(["quiz", "--db", str(tmp_path / "m.sqlite"),
                         "--campaign", CAMPAIGN, "--session", "s1",
                         "--out-md", str(tmp_path / "q.md")])
        out = capsys.readouterr().out
        if code == 1:
            assert ("none fit a quiz template" in out
                    or "no verified facts" in out)

    def test_cli_empty_session_fails(self, tmp_path):
        memory = CampaignMemory(tmp_path / "m.sqlite")
        memory.close()
        assert lab_main(["quiz", "--db", str(tmp_path / "m.sqlite"),
                         "--campaign", CAMPAIGN, "--session", "nope",
                         "--out-md", str(tmp_path / "q.md")]) == 1
