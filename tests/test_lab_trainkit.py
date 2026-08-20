"""Layer 4: the train kit and the promotion gate."""
import json

import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.trainkit import (
    MIN_EXAMPLES_TO_TRAIN,
    build_plan,
    promotion_verdict,
    render_modelfile,
    render_runbook,
    render_unsloth_script,
    write_train_kit,
)


def make_dataset(tmp_path, *, sft_train=120, sft_eval=30, dpo_train=50,
                 dpo_eval=10):
    for name, count in (("sft_train", sft_train), ("sft_eval", sft_eval),
                        ("dpo_train", dpo_train), ("dpo_eval", dpo_eval)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join('{"x": 1}\n' for _ in range(count)), encoding="utf-8")
    return tmp_path


class TestPlan:
    def test_ready_dataset(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path))
        assert plan.ready and plan.run_dpo
        assert plan.epochs == 4          # small archive: more repetition
        assert not plan.warnings

    def test_tiny_dataset_warns_and_is_not_ready(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path, sft_train=20, dpo_train=5))
        assert not plan.ready
        assert not plan.run_dpo
        assert any(str(MIN_EXAMPLES_TO_TRAIN) in w for w in plan.warnings)
        assert any("DPO" in w for w in plan.warnings)

    def test_no_eval_split_warns(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path, sft_eval=0))
        assert any("eval" in w for w in plan.warnings)

    def test_epochs_scale_down_with_data(self, tmp_path):
        assert build_plan(make_dataset(tmp_path, sft_train=1500)).epochs == 2


class TestRenderedKit:
    def test_script_carries_plan_facts(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path))
        script = render_unsloth_script(plan, model_name="heckuva-engine")
        assert plan.base_model in script
        assert "num_train_epochs=4" in script
        assert "DPOTrainer" in script            # pairs above the floor
        assert "save_pretrained_gguf" in script
        compile(script, "train.py", "exec")      # must be valid Python

    def test_dpo_stage_omitted_when_pairs_scarce(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path, dpo_train=3))
        script = render_unsloth_script(plan, model_name="m")
        assert "DPOTrainer" not in script
        assert "disabled" in script

    def test_modelfile_serves_refusal_system_prompt(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path))
        modelfile = render_modelfile(plan, model_name="heckuva-engine")
        assert "not in the record" in modelfile
        assert "temperature 0" in modelfile

    def test_runbook_names_the_gate(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path))
        runbook = render_runbook(plan, model_name="heckuva-engine")
        assert "promote" in runbook and "memory-baseline" in runbook

    def test_write_kit_files(self, tmp_path):
        plan = build_plan(make_dataset(tmp_path))
        files = write_train_kit(plan, tmp_path / "kit", model_name="m")
        assert files == ["Modelfile", "TRAIN_RUNBOOK.md", "train_m.py"]
        for name in files:
            assert (tmp_path / "kit" / name).stat().st_size > 0


def run(answerer, correct, total):
    return {"answerer": answerer, "correct": correct, "total": total,
            "campaign_id": "c", "run_at": 0.0, "by_type": {}, "failures": []}


class TestPromotionGate:
    def test_beating_baseline_promotes(self):
        verdict = promotion_verdict(
            [run("memory-baseline", 60, 100), run("llama", 70, 100)],
            candidate="llama", baseline="memory-baseline")
        assert verdict.promote
        assert "+10.0%" in verdict.reasons[0]

    def test_under_the_bar_holds(self):
        verdict = promotion_verdict(
            [run("memory-baseline", 70, 100), run("llama", 71, 100)],
            candidate="llama", baseline="memory-baseline")
        assert not verdict.promote

    def test_bank_size_mismatch_refuses_to_guess(self):
        verdict = promotion_verdict(
            [run("memory-baseline", 60, 80), run("llama", 70, 100)],
            candidate="llama", baseline="memory-baseline")
        assert not verdict.promote
        assert "same bank size" in verdict.reasons[0]

    def test_regression_vs_own_past_holds(self):
        verdict = promotion_verdict(
            [run("memory-baseline", 60, 100),
             run("llama", 80, 100), run("llama", 70, 100)],
            candidate="llama", baseline="memory-baseline")
        assert not verdict.promote
        assert any("regressed" in r for r in verdict.reasons)

    def test_missing_candidate_run(self):
        verdict = promotion_verdict([run("memory-baseline", 60, 100)],
                                    candidate="llama",
                                    baseline="memory-baseline")
        assert not verdict.promote


class TestCli:
    def test_train_kit_end_to_end(self, tmp_path, capsys):
        make_dataset(tmp_path)
        out = tmp_path / "kit"
        assert lab_main(["train-kit", "--dataset", str(tmp_path),
                         "--out-dir", str(out)]) == 0
        assert "READY" in capsys.readouterr().out
        assert (out / "train_heckuva-engine.py").exists()

    def test_train_kit_not_ready_exits_nonzero(self, tmp_path, capsys):
        make_dataset(tmp_path, sft_train=5)
        assert lab_main(["train-kit", "--dataset", str(tmp_path),
                         "--out-dir", str(tmp_path / "kit")]) == 1
        assert "NOT READY" in capsys.readouterr().out

    def test_promote_cli(self, tmp_path, capsys):
        history = tmp_path / "history.jsonl"
        history.write_text(
            json.dumps(run("memory-baseline", 60, 100)) + "\n"
            + json.dumps(run("llama", 75, 100)) + "\n", encoding="utf-8")
        assert lab_main(["promote", "--history", str(history),
                         "--candidate", "llama"]) == 0
        assert "PROMOTE" in capsys.readouterr().out
        history.write_text(
            json.dumps(run("memory-baseline", 80, 100)) + "\n"
            + json.dumps(run("llama", 75, 100)) + "\n", encoding="utf-8")
        assert lab_main(["promote", "--history", str(history),
                         "--candidate", "llama"]) == 1
