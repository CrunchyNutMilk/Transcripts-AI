"""Layer 3: compiling training records into SFT + DPO fine-tuning files."""
import json

import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.compile import (
    LEAVE_UNCHANGED,
    NOT_AN_ENTITY,
    REFUSAL,
    compile_dataset,
    dpo_pairs,
    sft_example,
)
from transcripts_ai.lab.records import TrainingRecord, write_records

CAMPAIGN = "heckuva"
SESSIONS = [f"s{i}" for i in range(1, 11)]


def rec(**overrides):
    base = dict(
        campaign_id=CAMPAIGN, session_id="s1", item_type="entity",
        subject="Gomra", proposal="Gomra->Ghomra", decision="alias",
        choice="Ghomra", accepted=True, actor="human:harry",
        evidence_quote="Gomra swings the axe.",
    )
    base.update(overrides)
    return TrainingRecord(**base)


class TestSftExamples:
    def test_alias_resolution(self):
        example = sft_example(rec())
        assert example.task == "resolve"
        assert example.assistant == "Ghomra"
        assert '"Gomra"' in example.user
        assert "Gomra swings the axe." in example.user

    def test_rejection_teaches_refusal_vocabulary(self):
        example = sft_example(rec(decision="reject", accepted=False,
                                  subject="jink", choice="link"))
        assert example.task == "reject-link"
        assert LEAVE_UNCHANGED in example.assistant
        assert "link" in example.assistant

    def test_not_entity(self):
        example = sft_example(rec(decision="not_entity", subject="gonf",
                                  choice=None, accepted=True))
        assert example.assistant == NOT_AN_ENTITY

    def test_transcription_fix(self):
        example = sft_example(rec(decision="substitute", subject="gomra",
                                  choice="ghomra",
                                  evidence_quote="then gomra spoke"))
        assert example.task == "transcription-fix"
        assert example.assistant == "ghomra"
        assert "then gomra spoke" in example.user

    def test_fact_false_positive_teaches_refusal(self):
        example = sft_example(rec(decision="false_positive",
                                  subject="The mayor is a dragon",
                                  choice=None, accepted=False))
        assert REFUSAL in example.assistant

    def test_unmappable_returns_none(self):
        assert sft_example(rec(decision="defer", choice=None)) is None

    def test_chat_json_shape(self):
        payload = json.loads(sft_example(rec()).to_json())
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert REFUSAL in payload["messages"][0]["content"]


class TestDpoPairs:
    def test_accept_reject_same_subject_pairs(self):
        records = [
            rec(),                                            # Gomra -> Ghomra
            rec(decision="reject", accepted=False, choice="Gondra"),
        ]
        pairs = dpo_pairs(records)
        assert len(pairs) == 1
        assert (pairs[0].chosen, pairs[0].rejected) == ("Ghomra", "Gondra")

    def test_panel_records_never_form_preferences(self):
        records = [
            rec(actor="panel:gpt+claude"),
            rec(decision="reject", accepted=False, choice="Gondra",
                actor="panel:gpt+claude"),
        ]
        assert dpo_pairs(records) == []

    def test_rejection_without_choice_uses_refusal(self):
        records = [rec(), rec(decision="reject", accepted=False, choice=None)]
        assert dpo_pairs(records)[0].rejected == REFUSAL

    def test_same_answer_never_pairs_with_itself(self):
        records = [rec(), rec(decision="reject", accepted=False, choice="ghomra")]
        assert dpo_pairs(records) == []


class TestCompile:
    def _records(self):
        rows = []
        for index, session in enumerate(SESSIONS):
            rows.append(rec(session_id=session, subject=f"Name{index}"))
            rows.append(rec(session_id=session, subject=f"Name{index}",
                            decision="reject", accepted=False,
                            choice=f"Wrong{index}"))
        return rows

    def test_split_is_leak_free(self, tmp_path):
        report = compile_dataset(self._records(), out_dir=tmp_path)
        train_sessions = {json.loads(l)["session_id"]
                          for l in (tmp_path / "sft_train.jsonl").read_text().splitlines()}
        eval_sessions = {json.loads(l)["session_id"]
                         for l in (tmp_path / "sft_eval.jsonl").read_text().splitlines()}
        assert train_sessions and eval_sessions
        assert not train_sessions & eval_sessions
        dpo_train = {json.loads(l)["session_id"]
                     for l in (tmp_path / "dpo_train.jsonl").read_text().splitlines()}
        assert not dpo_train & eval_sessions
        assert report.sft_train + report.sft_eval == len(self._records())

    def test_frozen_sessions_always_eval(self, tmp_path):
        compile_dataset(self._records(), out_dir=tmp_path, frozen=["s3"])
        train = (tmp_path / "sft_train.jsonl").read_text()
        assert '"session_id": "s3"' not in train

    def test_human_only_drops_panel_records(self, tmp_path):
        rows = self._records() + [rec(session_id="s1", subject="PanelThing",
                                      actor="panel:gpt")]
        report = compile_dataset(rows, out_dir=tmp_path, human_only=True)
        assert report.total_records == len(rows) - 1

    def test_unmapped_decisions_are_reported_not_dropped(self, tmp_path):
        rows = self._records() + [rec(session_id="s1", decision="defer",
                                      choice=None)]
        report = compile_dataset(rows, out_dir=tmp_path)
        assert report.unmapped["defer"] == 1
        assert "defer" in (tmp_path / "compile_report.md").read_text()

    def test_cli_end_to_end(self, tmp_path, capsys):
        records_path = tmp_path / "records.jsonl"
        write_records(records_path, self._records())
        out = tmp_path / "dataset"
        assert lab_main(["compile", "--records", str(records_path),
                         "--out-dir", str(out), "--frozen", "s5"]) == 0
        printed = capsys.readouterr().out
        assert "SFT" in printed and "DPO" in printed
        assert (out / "sft_train.jsonl").exists()
        assert (out / "dpo_eval.jsonl").exists()
        assert "leak-checked" in (out / "compile_report.md").read_text()

    def test_cli_empty_records_fails(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert lab_main(["compile", "--records", str(empty),
                         "--out-dir", str(tmp_path / "d")]) == 1
