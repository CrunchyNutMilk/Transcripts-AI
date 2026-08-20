"""Layer 2b: the overnight loop — checkpointing, resume, outputs, CLI."""
import json

import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.overnight import (
    OvernightState,
    read_results,
    result_from_json,
    result_to_json,
    run_overnight,
)
from transcripts_ai.lab.panel import Opinion, PanelResult
from transcripts_ai.lab.records import read_records
from transcripts_ai.lab.scorecard import Question, write_bank
from transcripts_ai.lab.teachers import discover_teachers
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.schemas import AliasRecord, EntityKind, EntityRecord, ReviewItem

CAMPAIGN = "heckuva"


def make_item(subject, *, item_type="entity", session="s1", evidence=None):
    return ReviewItem(
        campaign_id=CAMPAIGN, session_id=session, item_type=item_type,
        subject=subject, reason="engine was not confident",
        evidence=[f"[00:01] the party met {subject} at the gate."]
        if evidence is None else evidence,
        suggestions=[{"canonical": "Ghomra", "score": 0.85}],
    )


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "memory.sqlite")
    ghomra = mem.upsert_entity(
        EntityRecord(name="Ghomra", kind=EntityKind.NPC, campaign_id=CAMPAIGN),
        actor="human:harry",
    )
    mem.add_alias(
        AliasRecord(campaign_id=CAMPAIGN, observed="Gomra", canonical="Ghomra",
                    entity_id=ghomra.entity_id, approved_by="human:harry"),
        actor="human:harry",
    )
    yield mem
    mem.close()


def accepting_teachers():
    teachers, _ = discover_teachers(
        {"TEACHERS": "fake:accept:Ghomra,fake:accept:Ghomra"})
    return teachers


class TestResultSerialization:
    def test_round_trip(self):
        result = PanelResult(
            item=make_item("Gomra"),
            opinions=[Opinion(teacher="engine", verdict="accept",
                              choice="Ghomra", reason="alias")],
            decision="bank_accept", agreed_choice="Ghomra", gate_note="ok",
        )
        again = result_from_json(result_to_json(result))
        assert again.decision == "bank_accept"
        assert again.item.item_id == result.item.item_id
        assert again.opinions[0].choice == "Ghomra"


class TestOvernightRun:
    def test_full_night_banks_and_reports(self, memory, tmp_path):
        for subject in ("Gomra", "Gamra"):
            memory.enqueue_review(make_item(subject), actor="engine")
        out = tmp_path / "night1"
        report = run_overnight(memory, CAMPAIGN, accepting_teachers(),
                               out_dir=out, on_progress=lambda *_: None)
        assert report.processed_tonight == 2
        assert report.bank_accept == 2 and report.queued == 0
        banked = read_records(out / "banked_records.jsonl")
        assert len(banked) == 2
        assert all(r.actor.startswith("panel:") for r in banked)
        report_md = (out / "morning_report.md").read_text(encoding="utf-8")
        assert "Banked automatically" in report_md
        # memory untouched: both items still pending for the human
        assert len(memory.pending_reviews(CAMPAIGN)) == 2

    def test_crash_resume_never_repays(self, memory, tmp_path):
        for subject in ("Gomra", "Gamra", "Goomra"):
            memory.enqueue_review(make_item(subject), actor="engine")
        out = tmp_path / "night"
        # First run dies after one item (max_items simulates the crash point).
        run_overnight(memory, CAMPAIGN, accepting_teachers(), out_dir=out,
                      max_items=1, on_progress=lambda *_: None)
        state = OvernightState.load(out, CAMPAIGN)
        assert len(state.processed) == 1

        # Resume: fresh teachers count their own calls this run.
        teachers = accepting_teachers()
        report = run_overnight(memory, CAMPAIGN, teachers, out_dir=out,
                               on_progress=lambda *_: None)
        assert report.skipped_resumed == 1
        assert report.processed_tonight == 2       # only the unpaid items
        assert teachers[0].provider.calls == 2     # never re-asked item 1
        # Outputs cover the WHOLE night, first run included, exactly once.
        assert len(read_results(out / "results.jsonl")) == 3
        assert len(read_records(out / "banked_records.jsonl")) == 3

    def test_state_refuses_wrong_campaign(self, memory, tmp_path):
        out = tmp_path / "night"
        memory.enqueue_review(make_item("Gomra"), actor="engine")
        run_overnight(memory, CAMPAIGN, accepting_teachers(), out_dir=out,
                      on_progress=lambda *_: None)
        with pytest.raises(ValueError, match="separate --out-dir"):
            OvernightState.load(out, "other-campaign")

    def test_spend_cap_reported(self, memory, tmp_path):
        for index in range(4):
            memory.enqueue_review(make_item(f"Name{index}"), actor="engine")
        report = run_overnight(memory, CAMPAIGN, accepting_teachers(),
                               out_dir=tmp_path / "n", max_items=3,
                               on_progress=lambda *_: None)
        assert report.processed_tonight == 3
        assert report.remaining_pending == 1

    def test_disagreements_fill_the_morning_queue(self, memory, tmp_path):
        memory.enqueue_review(make_item("Xelzor"), actor="engine")
        teachers, _ = discover_teachers(
            {"TEACHERS": "fake:accept:Xelzor,fake:reject"})
        out = tmp_path / "n"
        report = run_overnight(memory, CAMPAIGN, teachers, out_dir=out,
                               on_progress=lambda *_: None)
        assert report.queued == 1 and report.bank_accept == 0
        rows = [json.loads(l) for l in
                (out / "morning_queue.jsonl").read_text().splitlines()]
        assert {o["teacher"] for o in rows[0]["opinions"]} >= \
            {"engine", "fake-accept", "fake-reject"}
        report_md = (out / "morning_report.md").read_text(encoding="utf-8")
        assert "Your morning questions" in report_md and "Xelzor" in report_md

    def test_nightly_scorecard_appends_history(self, memory, tmp_path):
        from transcripts_ai.lab.answerers import memory_answerer
        memory.enqueue_review(make_item("Gomra"), actor="engine")
        bank = tmp_path / "bank.jsonl"
        write_bank(bank, [Question(
            question_id="q1", campaign_id=CAMPAIGN, qtype="alias",
            question='Which known name does "Gomra" refer to?',
            expected=["Ghomra"])])
        out = tmp_path / "n"
        answerer = memory_answerer(memory, CAMPAIGN)
        first = run_overnight(memory, CAMPAIGN, accepting_teachers(),
                              out_dir=out, bank_path=str(bank),
                              answerer=answerer, on_progress=lambda *_: None)
        assert "1/1" in first.scorecard_line
        second = run_overnight(memory, CAMPAIGN, accepting_teachers(),
                               out_dir=out, bank_path=str(bank),
                               answerer=answerer, on_progress=lambda *_: None)
        assert "previous run" in second.scorecard_line
        history = (out / "scorecard_history.jsonl").read_text().splitlines()
        assert len(history) == 2


class TestOvernightCli:
    def test_end_to_end(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "memory.sqlite"
        memory = CampaignMemory(db)
        memory.upsert_entity(
            EntityRecord(name="Ghomra", kind=EntityKind.NPC,
                         campaign_id=CAMPAIGN), actor="human:harry")
        memory.add_alias(
            AliasRecord(campaign_id=CAMPAIGN, observed="Gomra",
                        canonical="Ghomra", entity_id=None,
                        approved_by="human:harry"), actor="human:harry")
        memory.enqueue_review(make_item("Gomra"), actor="engine")
        memory.enqueue_review(make_item("Xelzor"), actor="engine")
        memory.close()
        monkeypatch.setenv("TEACHERS", "fake:accept:Ghomra,fake:accept:Ghomra")
        out = tmp_path / "night"
        assert lab_main(["overnight", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out-dir", str(out)]) == 0
        printed = capsys.readouterr().out
        assert "asked 2 item(s)" in printed
        assert "morning report ->" in printed
        assert (out / "morning_report.md").exists()
        assert (out / "banked_records.jsonl").exists()
        assert (out / "morning_queue.jsonl").exists()

    def test_no_teachers_fails(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TEACHERS", raising=False)
        assert lab_main(["overnight", "--db", str(tmp_path / "m.sqlite"),
                         "--campaign", CAMPAIGN,
                         "--out-dir", str(tmp_path / "n")]) == 1
