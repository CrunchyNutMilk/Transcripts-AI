"""Layer 2a: the teacher panel — opinions, gate, aggregation, outputs, CLI."""
import json

import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.answerers import teacher_answerer
from transcripts_ai.lab.panel import (
    Opinion,
    build_panel_prompt,
    decide,
    engine_opinion,
    evidence_gate,
    queue_from_panel,
    records_from_panel,
    run_panel,
    summarize,
    teacher_opinion,
)
from transcripts_ai.lab.records import read_records
from transcripts_ai.lab.scorecard import NOT_IN_RECORD, Question, write_bank
from transcripts_ai.lab.teachers import Teacher, discover_teachers
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.providers import FakeProvider
from transcripts_ai.schemas import AliasRecord, EntityKind, EntityRecord, ReviewItem

CAMPAIGN = "heckuva"


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


def make_item(subject="Gomra", *, item_type="spelling", evidence=None,
              suggestions=None, session="s1"):
    return ReviewItem(
        campaign_id=CAMPAIGN, session_id=session, item_type=item_type,
        subject=subject, reason="engine was not confident",
        evidence=["[00:12:01] Gomra swings the axe."] if evidence is None else evidence,
        suggestions=suggestions or [{"canonical": "Ghomra", "score": 0.85}],
    )


def fake_teacher(*responses, name="fake"):
    return Teacher(name=name, provider=FakeProvider(list(responses)))


def verdict(v, choice="", reason="r"):
    return json.dumps({"verdict": v, "choice": choice, "reason": reason})


class TestEngineOpinion:
    def test_approved_alias_is_an_accept(self, memory):
        opinion = engine_opinion(memory, CAMPAIGN, make_item("Gomra"))
        assert opinion.verdict == "accept" and opinion.choice == "Ghomra"
        assert opinion.teacher == "engine"

    def test_unknown_subject_is_a_reject(self, memory):
        opinion = engine_opinion(memory, CAMPAIGN, make_item("xqzzt"))
        assert opinion.verdict == "reject"


class TestTeacherOpinion:
    def test_valid_verdict_parsed(self, memory):
        teacher = fake_teacher(verdict("accept", "Ghomra"))
        opinion = teacher_opinion(teacher, "prompt")
        assert (opinion.verdict, opinion.choice) == ("accept", "Ghomra")

    def test_garbage_becomes_abstain_after_retry(self):
        teacher = fake_teacher("not json", "still not json")
        opinion = teacher_opinion(teacher, "prompt")
        assert opinion.verdict == "abstain"
        assert "ValidationFailed" in opinion.reason
        assert teacher.provider.calls and len(teacher.provider.calls) == 2

    def test_corrective_retry_can_recover(self):
        teacher = fake_teacher("nonsense", verdict("reject"))
        assert teacher_opinion(teacher, "prompt").verdict == "reject"

    def test_provider_error_becomes_abstain(self):
        teacher = fake_teacher()   # empty queue -> ProviderError
        opinion = teacher_opinion(teacher, "prompt")
        assert opinion.verdict == "abstain" and "ProviderError" in opinion.reason

    def test_bad_verdict_value_rejected_then_abstain(self):
        teacher = fake_teacher(verdict("definitely"), verdict("perhaps"))
        assert teacher_opinion(teacher, "prompt").verdict == "abstain"


class TestEvidenceGate:
    def test_known_entity_passes(self, memory):
        ok, _ = evidence_gate(memory, CAMPAIGN, make_item(), "Ghomra")
        assert ok

    def test_approved_alias_passes(self, memory):
        ok, note = evidence_gate(memory, CAMPAIGN, make_item(), "Gomra")
        assert ok and "Ghomra" in note

    def test_invented_name_fails(self, memory):
        ok, note = evidence_gate(memory, CAMPAIGN, make_item(), "Sauron")
        assert not ok and "not a known entity" in note

    def test_no_evidence_fails(self, memory):
        ok, note = evidence_gate(memory, CAMPAIGN, make_item(evidence=[]), "Ghomra")
        assert not ok and "no evidence" in note


class TestDecide:
    def _opinions(self, *verdicts_and_choices):
        return [Opinion(teacher=f"t{i}", verdict=v, choice=c)
                for i, (v, c) in enumerate(verdicts_and_choices)]

    def test_unanimous_accept_through_gate_banks(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("accept", "Ghomra"), ("accept", "ghomra "), ("accept", "Ghomra")))
        assert result.decision == "bank_accept"
        assert result.agreed_choice == "Ghomra"

    def test_agreement_on_invented_name_is_blocked_by_gate(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("accept", "Sauron"), ("accept", "Sauron"), ("accept", "Sauron")))
        assert result.decision == "queue"
        assert "gate" in result.gate_note

    def test_unanimous_reject_banks_hard_negative(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("reject", ""), ("reject", ""), ("reject", "")))
        assert result.decision == "bank_reject"

    def test_split_panel_queues(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("accept", "Ghomra"), ("reject", ""), ("accept", "Ghomra")))
        assert result.decision == "queue"

    def test_any_uncertainty_queues(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("accept", "Ghomra"), ("uncertain", "Ghomra"), ("accept", "Ghomra")))
        assert result.decision == "queue"

    def test_accepts_disagreeing_on_choice_queue(self, memory):
        item = make_item(suggestions=[{"canonical": "Ghomra", "score": 0.9},
                                      {"canonical": "Gomrad", "score": 0.8}])
        result = decide(memory, CAMPAIGN, item, self._opinions(
            ("accept", "Ghomra"), ("accept", "Gomrad")))
        assert result.decision == "queue"

    def test_single_vote_never_banks(self, memory):
        result = decide(memory, CAMPAIGN, make_item(), self._opinions(
            ("accept", "Ghomra"), ("abstain", ""), ("abstain", "")))
        assert result.decision == "queue"


class TestPanelOutputs:
    def _results(self, memory):
        items = [make_item("Gomra"), make_item("gonf", item_type="entity")]
        teachers = [
            fake_teacher(verdict("accept", "Ghomra"), verdict("accept", "Ghomra"),
                         name="gpt"),
            fake_teacher(verdict("accept", "Ghomra"), verdict("reject"),
                         name="claude"),
        ]
        return run_panel(memory, CAMPAIGN, items, teachers), teachers

    def test_run_panel_banks_and_queues(self, memory):
        results, _ = self._results(memory)
        assert [r.decision for r in results] == ["bank_accept", "queue"]

    def test_panel_is_read_only_on_memory(self, memory):
        item = make_item("Gomra")
        memory.enqueue_review(item, actor="engine")
        before_feedback = memory.all_feedback(CAMPAIGN)
        run_panel(memory, CAMPAIGN, [item],
                  [fake_teacher(verdict("accept", "Ghomra"))])
        still_pending = memory.pending_reviews(CAMPAIGN)
        assert [i.item_id for i in still_pending] == [item.item_id]
        assert memory.all_feedback(CAMPAIGN) == before_feedback

    def test_banked_records_are_never_human(self, memory):
        results, _ = self._results(memory)
        records = records_from_panel(results)
        assert len(records) == 1
        record = records[0]
        assert record.source == "teacher-panel"
        assert record.actor.startswith("panel:")
        assert not record.actor.startswith("human:")
        assert record.choice == "Ghomra" and record.accepted
        assert record.extras["opinions"]      # full provenance kept

    def test_queue_rows_carry_every_opinion(self, memory):
        results, _ = self._results(memory)
        rows = queue_from_panel(results)
        assert len(rows) == 1
        assert rows[0]["subject"] == "gonf"
        assert {o["teacher"] for o in rows[0]["opinions"]} == \
            {"engine", "gpt", "claude"}

    def test_summarize_counts(self, memory):
        results, teachers = self._results(memory)
        report = summarize(results, teachers)
        assert (report.total, report.bank_accept, report.queued) == (2, 1, 1)

    def test_prompt_shows_names_evidence_and_rules(self, memory):
        item = make_item()
        prompt = build_panel_prompt(item, ["Ghomra"])
        assert "Ghomra" in prompt and item.evidence[0] in prompt
        assert "KNOWN NAMES" in prompt


class TestTeacherAnswerer:
    def test_scripted_answer_flows_through(self, memory):
        teachers, _ = discover_teachers({"TEACHERS": "fake:answer:Ghomra"})
        answer = teacher_answerer(memory, CAMPAIGN, teachers[0])(
            Question(question_id="q", campaign_id=CAMPAIGN, qtype="alias",
                     question='Which known name does "Gomra" refer to?',
                     expected=["Ghomra"]))
        assert answer == "Ghomra"

    def test_broken_teacher_refuses_instead_of_guessing(self, memory):
        teachers, _ = discover_teachers({"TEACHERS": "fake:garbage"})
        answer = teacher_answerer(memory, CAMPAIGN, teachers[0])(
            Question(question_id="q", campaign_id=CAMPAIGN, qtype="fact",
                     question="Who guards the Silver Spire?", expected=["Silas"]))
        assert answer == NOT_IN_RECORD


class TestPanelCli:
    def _seed_db(self, path):
        memory = CampaignMemory(path)
        ghomra = memory.upsert_entity(
            EntityRecord(name="Ghomra", kind=EntityKind.NPC, campaign_id=CAMPAIGN),
            actor="human:harry",
        )
        memory.add_alias(
            AliasRecord(campaign_id=CAMPAIGN, observed="Gomra", canonical="Ghomra",
                        entity_id=ghomra.entity_id, approved_by="human:harry"),
            actor="human:harry",
        )
        memory.enqueue_review(make_item("Gomra"), actor="engine")
        memory.enqueue_review(make_item("gonf", item_type="entity"), actor="engine")
        memory.close()

    def test_teachers_command_lists_ready_and_skipped(self, monkeypatch, capsys):
        monkeypatch.setenv("TEACHERS", "fake:accept:Ghomra,openai:gpt-5-mini")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert lab_main(["teachers"]) == 0
        out = capsys.readouterr().out
        assert "ready    fake-accept" in out and "skipped  openai:gpt-5-mini" in out

    def test_teachers_command_none_configured(self, monkeypatch, capsys):
        monkeypatch.delenv("TEACHERS", raising=False)
        assert lab_main(["teachers"]) == 1
        assert "no teachers configured" in capsys.readouterr().out

    def test_panel_end_to_end_with_fakes(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "memory.sqlite"
        self._seed_db(db)
        monkeypatch.setenv("TEACHERS", "fake:accept:Ghomra,fake:accept:Ghomra")
        queue = tmp_path / "queue.jsonl"
        banked = tmp_path / "banked.jsonl"
        assert lab_main(["panel", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out-queue", str(queue),
                         "--out-records", str(banked)]) == 0
        out = capsys.readouterr().out
        assert "1 banked accept" in out and "1 queued" in out
        records = read_records(banked)
        assert len(records) == 1 and records[0].actor.startswith("panel:")
        rows = [json.loads(l) for l in queue.read_text().splitlines()]
        assert rows and rows[0]["subject"] == "gonf"
        # panel never resolves anything: both items still await the human
        memory = CampaignMemory(db)
        assert len(memory.pending_reviews(CAMPAIGN)) == 2
        memory.close()

    def test_panel_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "memory.sqlite"
        self._seed_db(db)
        monkeypatch.setenv("TEACHERS", "fake:accept:Ghomra")
        queue, banked = tmp_path / "q.jsonl", tmp_path / "b.jsonl"
        assert lab_main(["panel", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out-queue", str(queue), "--out-records", str(banked),
                         "--dry-run"]) == 0
        assert "DRY RUN" in capsys.readouterr().out
        assert not queue.exists() and not banked.exists()

    def test_panel_without_teachers_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "memory.sqlite"
        self._seed_db(db)
        monkeypatch.delenv("TEACHERS", raising=False)
        assert lab_main(["panel", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out-queue", str(tmp_path / "q.jsonl"),
                         "--out-records", str(tmp_path / "b.jsonl")]) == 1

    def test_score_with_teacher_answerer(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "memory.sqlite"
        self._seed_db(db)
        monkeypatch.setenv("TEACHERS", "fake:answer:Ghomra")
        bank = tmp_path / "bank.jsonl"
        write_bank(bank, [Question(
            question_id="q1", campaign_id=CAMPAIGN, qtype="alias",
            question='Which known name does "Gomra" refer to?',
            expected=["Ghomra"])])
        assert lab_main(["score", "--db", str(db), "--campaign", CAMPAIGN,
                         "--bank", str(bank),
                         "--answerer", "teacher:fake-answer"]) == 0
        out = capsys.readouterr().out
        assert "teacher:fake-answer" in out and "1/1" in out

    def test_score_unknown_teacher_fails_loudly(self, tmp_path, monkeypatch):
        db = tmp_path / "memory.sqlite"
        self._seed_db(db)
        monkeypatch.setenv("TEACHERS", "fake:answer:Ghomra")
        bank = tmp_path / "bank.jsonl"
        write_bank(bank, [Question(
            question_id="q1", campaign_id=CAMPAIGN, qtype="trick",
            question="When did Boblin die?", expected=[])])
        with pytest.raises(SystemExit, match="no teacher named"):
            lab_main(["score", "--db", str(db), "--campaign", CAMPAIGN,
                      "--bank", str(bank), "--answerer", "teacher:nope"])
