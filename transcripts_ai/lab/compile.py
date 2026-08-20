"""Layer 3: compile training records into fine-tuning files for the 3080.

Everything upstream produces ``TrainingRecord`` JSONL — the human review
ledger (``export-records``), the teacher panel's banked verdicts, and
gold-transcript corrections (``gold-score --out-records``). This layer
turns those archives into the two files a QLoRA run actually eats:

- **SFT** (``sft_train.jsonl`` / ``sft_eval.jsonl``): chat-format
  ``{"messages": [...]}`` examples in the axolotl/unsloth dialect. Each
  record becomes a task the local model must learn: resolve a name,
  reject a bad link, refuse an invented entity, fix a transcription.
  Rejections are training data too — the assistant's answer for them is
  the same refusal vocabulary the scorecard accepts, so training and
  measurement pull in the same direction.
- **DPO** (``dpo_train.jsonl`` / ``dpo_eval.jsonl``): preference pairs
  ``{"prompt", "chosen", "rejected"}`` built ONLY from real human
  accept/reject decisions about the same subject — the most
  sample-efficient use of a small archive.

Two rules are load-bearing:

- **Splits are by session and leak-checked.** The same eval_fraction,
  seed and frozen registry as the scorecard (``records.split_sessions``),
  and compilation refuses to write anything if a session ends up on both
  sides.
- **Nothing is dropped silently.** Records that don't map to a task are
  counted and reported, never vanished.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .records import RecordError, SessionSplit, TrainingRecord, split_sessions

REFUSAL = "not in the record"
NOT_AN_ENTITY = "not an entity — ordinary language"
LEAVE_UNCHANGED = "leave it unchanged"


def system_prompt(campaign_id: str) -> str:
    return (
        f"You are the transcript engine for the D&D campaign "
        f"\"{campaign_id}\". Answer from the campaign record only. "
        f"If the record does not contain the answer, reply exactly: {REFUSAL}"
    )


# ---------------------------------------------------------------------------
# Record -> task
# ---------------------------------------------------------------------------

@dataclass
class SftExample:
    campaign_id: str
    session_id: str
    task: str
    user: str
    assistant: str
    source: str

    def to_json(self) -> str:
        return json.dumps({
            "messages": [
                {"role": "system", "content": system_prompt(self.campaign_id)},
                {"role": "user", "content": self.user},
                {"role": "assistant", "content": self.assistant},
            ],
            "task": self.task,
            "session_id": self.session_id,
            "source": self.source,
        }, sort_keys=True, ensure_ascii=False)


def _resolve_question(record: TrainingRecord) -> str:
    evidence = f"\nEvidence: \"{record.evidence_quote}\"" if record.evidence_quote else ""
    return (f'In this campaign, the transcript contains the name '
            f'"{record.subject}".{evidence}\n'
            f"Which known campaign name does it refer to? If none, say so.")


def sft_example(record: TrainingRecord) -> SftExample | None:
    """One record -> one SFT example, or None when no task fits."""
    kwargs = dict(campaign_id=record.campaign_id, session_id=record.session_id,
                  source=record.source)
    decision = record.decision
    if decision in ("correct", "alias", "vault_link", "accept") and record.choice:
        return SftExample(task="resolve", user=_resolve_question(record),
                          assistant=record.choice, **kwargs)
    if decision in ("reject",) or (decision == "accept" and not record.accepted):
        target = f' It is not "{record.choice}".' if record.choice else ""
        return SftExample(task="reject-link", user=_resolve_question(record),
                          assistant=f"{LEAVE_UNCHANGED}.{target}".strip(),
                          **kwargs)
    if decision == "not_entity":
        return SftExample(
            task="not-entity",
            user=(f'Is "{record.subject}" a character, place or thing in '
                  f"this campaign?"),
            assistant=NOT_AN_ENTITY, **kwargs)
    if decision == "new":
        return SftExample(
            task="new-entity",
            user=(f'Is "{record.subject}" a character, place or thing in '
                  f"this campaign?"),
            assistant=f'Yes — "{record.subject}" is a campaign entity.',
            **kwargs)
    if decision == "substitute" and record.choice:
        quote = record.evidence_quote or record.subject
        return SftExample(
            task="transcription-fix",
            user=(f'A transcription contains the word "{record.subject}" in:\n'
                  f'"{quote}"\nWhat word was actually said?'),
            assistant=record.choice, **kwargs)
    if decision == "false_positive":
        return SftExample(
            task="fact-verify",
            user=(f"Proposed fact: {record.subject}\n"
                  f"Is this supported by the campaign record?"),
            assistant=f"Unsupported — {REFUSAL}.", **kwargs)
    return None       # counted by the caller, never silently dropped


# ---------------------------------------------------------------------------
# DPO pairs
# ---------------------------------------------------------------------------

@dataclass
class DpoPair:
    campaign_id: str
    session_id: str
    prompt: str
    chosen: str
    rejected: str

    def to_json(self) -> str:
        return json.dumps({
            "system": system_prompt(self.campaign_id),
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "session_id": self.session_id,
        }, sort_keys=True, ensure_ascii=False)


def dpo_pairs(records: list[TrainingRecord]) -> list[DpoPair]:
    """Real preference pairs: same subject, one human accept, one reject.

    Only human decisions qualify — teacher-panel verdicts must not teach
    preferences (teachers are wrong in correlated ways).
    """
    by_subject: dict[tuple[str, str], list[TrainingRecord]] = {}
    for record in records:
        if not record.actor.startswith("human:"):
            continue
        key = (record.campaign_id, record.subject.casefold())
        by_subject.setdefault(key, []).append(record)

    pairs: list[DpoPair] = []
    for group in by_subject.values():
        accepted = [r for r in group if r.accepted and r.choice]
        rejected = [r for r in group if not r.accepted]
        for good in accepted:
            for bad in rejected:
                bad_answer = bad.choice or REFUSAL
                if bad_answer.casefold() == good.choice.casefold():
                    continue
                pairs.append(DpoPair(
                    campaign_id=good.campaign_id,
                    session_id=good.session_id,
                    prompt=_resolve_question(good),
                    chosen=good.choice,
                    rejected=bad_answer,
                ))
    return pairs


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

@dataclass
class CompileReport:
    total_records: int
    sft_train: int
    sft_eval: int
    dpo_train: int
    dpo_eval: int
    unmapped: Counter = field(default_factory=Counter)
    by_task: Counter = field(default_factory=Counter)
    split: SessionSplit | None = None


def compile_dataset(
    records: list[TrainingRecord],
    *,
    out_dir: str | Path,
    frozen: list[str] | None = None,
    eval_fraction: float = 0.2,
    human_only: bool = False,
) -> CompileReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if human_only:
        records = [r for r in records if r.actor.startswith("human:")]

    split = split_sessions((r.session_id for r in records),
                           frozen=frozen or (), eval_fraction=eval_fraction)
    overlap = set(split.train_sessions) & set(split.eval_sessions)
    if overlap:      # split_sessions guarantees this; refuse loudly anyway
        raise RecordError(f"session leak between train and eval: {sorted(overlap)}")

    report = CompileReport(total_records=len(records), sft_train=0,
                           sft_eval=0, dpo_train=0, dpo_eval=0, split=split)

    sft: dict[str, list[SftExample]] = {"train": [], "eval": []}
    for record in records:
        example = sft_example(record)
        if example is None:
            report.unmapped[record.decision] += 1
            continue
        side = split.side_of(record.session_id)
        if side == "unassigned":
            report.unmapped["unassigned-session"] += 1
            continue
        report.by_task[example.task] += 1
        sft[side].append(example)

    train_sessions = set(split.train_sessions)
    eval_sessions = set(split.eval_sessions)
    pairs = dpo_pairs(records)
    dpo: dict[str, list[DpoPair]] = {"train": [], "eval": []}
    for pair in pairs:
        if pair.session_id in train_sessions:
            dpo["train"].append(pair)
        elif pair.session_id in eval_sessions:
            dpo["eval"].append(pair)

    for side in ("train", "eval"):
        with open(out / f"sft_{side}.jsonl", "w", encoding="utf-8") as f:
            for example in sft[side]:
                f.write(example.to_json() + "\n")
        with open(out / f"dpo_{side}.jsonl", "w", encoding="utf-8") as f:
            for pair in dpo[side]:
                f.write(pair.to_json() + "\n")

    report.sft_train, report.sft_eval = len(sft["train"]), len(sft["eval"])
    report.dpo_train, report.dpo_eval = len(dpo["train"]), len(dpo["eval"])
    (out / "compile_report.md").write_text(render_report(report),
                                           encoding="utf-8")
    return report


def render_report(report: CompileReport) -> str:
    lines = [
        "# Training-data compile report",
        "",
        f"- Input records: **{report.total_records}**",
        f"- SFT examples: **{report.sft_train} train / {report.sft_eval} eval**",
        f"- DPO pairs: **{report.dpo_train} train / {report.dpo_eval} eval** "
        "(human decisions only)",
        "",
        "## By task",
        "",
    ]
    for task, count in report.by_task.most_common():
        lines.append(f"- {task}: {count}")
    if report.unmapped:
        lines += ["", "## Not compiled (kept out, never silently dropped)", ""]
        for decision, count in report.unmapped.most_common():
            lines.append(f"- decision {decision!r}: {count}")
    if report.split:
        lines += [
            "", "## Session split (leak-checked)", "",
            f"- train ({len(report.split.train_sessions)}): "
            + ", ".join(report.split.train_sessions),
            f"- eval ({len(report.split.eval_sessions)}): "
            + ", ".join(report.split.eval_sessions)
            + (f"   [frozen: {', '.join(report.split.frozen_sessions)}]"
               if report.split.frozen_sessions else ""),
        ]
    lines += [
        "", "## Feed it to the 3080", "",
        "The SFT files are chat-format JSONL (`messages`), the DPO files are "
        "prompt/chosen/rejected JSONL — both load directly in axolotl or "
        "unsloth. Train on `*_train.jsonl` ONLY; `*_eval.jsonl` is the exam.",
    ]
    return "\n".join(lines) + "\n"
