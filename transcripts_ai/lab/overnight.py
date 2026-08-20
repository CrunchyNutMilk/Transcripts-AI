"""The overnight loop: leave it running, wake up to a morning report.

Layer 2b of docs/TRAINING.md. One command that works through the pending
review queue with the teacher panel, banks what the rules allow, queues
what needs a human, scores the nightly scorecard, and writes a morning
report — checkpointing after EVERY item so a 3 a.m. crash (or Ctrl-C)
loses nothing: re-running with the same --out-dir resumes exactly where
it stopped and never re-pays for an item already asked.

Files in the out-dir:

- ``state.json``            — processed item ids (the resume checkpoint)
- ``results.jsonl``         — one line per panel verdict, appended live
- ``banked_records.jsonl``  — training records from unanimous verdicts
- ``morning_queue.jsonl``   — items for the human, pre-answered
- ``scorecard_history.jsonl`` (with --bank) — nightly accuracy, appended
- ``morning_report.md``     — what happened, what to do next

Invariants inherited from the panel layer, unchanged here: campaign
memory is never written; teachers never bank alone (the engine's vote is
not quorum); every banked record carries ``actor="panel:…"``. The one
memory-writing step the plan mentions — retraining the learner — happens
AFTER the human clears the morning queue, via ``python -m transcripts_ai
train``; the report says so rather than doing it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..memory import CampaignMemory
from ..schemas import ReviewItem
from .panel import (
    MIN_VOTES_TO_BANK,
    Opinion,
    PanelResult,
    panel_item,
    queue_from_panel,
    records_from_panel,
)
from .records import write_records
from .scorecard import append_history, read_bank, read_history, run_scorecard
from .teachers import Teacher

DEFAULT_MAX_ITEMS = 100        # per night; the hard spend cap on panel items


# ---------------------------------------------------------------------------
# Crash-safe persistence
# ---------------------------------------------------------------------------

@dataclass
class OvernightState:
    campaign_id: str
    processed: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, out_dir: Path, campaign_id: str) -> "OvernightState":
        path = out_dir / "state.json"
        if not path.exists():
            return cls(campaign_id=campaign_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("campaign_id") != campaign_id:
            raise ValueError(
                f"{path} belongs to campaign {data.get('campaign_id')!r}, "
                f"not {campaign_id!r}; use a separate --out-dir per campaign"
            )
        return cls(campaign_id=campaign_id,
                   processed=list(data.get("processed", [])))

    def save(self, out_dir: Path) -> None:
        path = out_dir / "state.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"campaign_id": self.campaign_id, "processed": self.processed},
            indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(path)      # atomic: a crash never truncates the state


def result_to_json(result: PanelResult) -> str:
    item = result.item
    return json.dumps({
        "item": {
            "campaign_id": item.campaign_id,
            "session_id": item.session_id,
            "item_type": item.item_type,
            "subject": item.subject,
            "reason": item.reason,
            "evidence": item.evidence,
            "suggestions": item.suggestions,
            "confidence": item.confidence,
            "item_id": item.item_id,
        },
        "opinions": [asdict(o) for o in result.opinions],
        "decision": result.decision,
        "agreed_choice": result.agreed_choice,
        "gate_note": result.gate_note,
    }, sort_keys=True, ensure_ascii=False)


def result_from_json(raw: str) -> PanelResult:
    data = json.loads(raw)
    return PanelResult(
        item=ReviewItem(**data["item"]),
        opinions=[Opinion(**o) for o in data["opinions"]],
        decision=data["decision"],
        agreed_choice=data.get("agreed_choice", ""),
        gate_note=data.get("gate_note", ""),
    )


def read_results(path: Path) -> list[PanelResult]:
    if not path.exists():
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(result_from_json(line))
    return results


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class OvernightReport:
    campaign_id: str
    processed_tonight: int
    skipped_resumed: int
    remaining_pending: int
    bank_accept: int
    bank_reject: int
    queued: int
    abstains_by_teacher: dict[str, int]
    usage_by_teacher: dict[str, dict[str, int]]
    scorecard_line: str = ""
    out_dir: str = ""


def run_overnight(
    memory: CampaignMemory,
    campaign_id: str,
    teachers: list[Teacher],
    *,
    out_dir: str | Path,
    session_id: str | None = None,
    max_items: int = DEFAULT_MAX_ITEMS,
    min_votes: int = MIN_VOTES_TO_BANK,
    bank_path: str | None = None,
    answerer=None,
    answerer_name: str = "memory-baseline",
    on_progress=print,
) -> OvernightReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = OvernightState.load(out, campaign_id)
    results_path = out / "results.jsonl"

    pending = memory.pending_reviews(campaign_id, session_id=session_id)
    done = set(state.processed)
    todo = [i for i in pending if i.item_id not in done]
    skipped = len(pending) - len(todo)
    if skipped:
        on_progress(f"resume: {skipped} item(s) already asked, skipping")
    tonight = todo[:max_items]
    if len(todo) > len(tonight):
        on_progress(f"spend cap: {len(tonight)} of {len(todo)} pending items "
                    "tonight (raise --max-items to do more)")

    for index, item in enumerate(tonight, start=1):
        result = panel_item(memory, campaign_id, item, teachers,
                            min_votes=min_votes)
        # Append the verdict, THEN mark processed — a crash between the two
        # re-asks one item; the reverse order would silently drop one.
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(result_to_json(result) + "\n")
        state.processed.append(item.item_id)
        state.save(out)
        on_progress(f"[{index}/{len(tonight)}] {result.decision:12s} "
                    f"{item.subject[:48]!r}")

    # Outputs regenerate from the FULL results file, so a resumed night's
    # banked/queue files always cover every item asked, not just tonight's.
    results = read_results(results_path)
    seen_ids: set[str] = set()
    unique_results = []
    for result in results:
        if result.item.item_id not in seen_ids:
            seen_ids.add(result.item.item_id)
            unique_results.append(result)
    banked = records_from_panel(unique_results)
    queued = queue_from_panel(unique_results)
    write_records(out / "banked_records.jsonl", banked)
    with open(out / "morning_queue.jsonl", "w", encoding="utf-8") as f:
        for row in queued:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    abstains: dict[str, int] = {}
    for result in unique_results:
        for opinion in result.opinions:
            if opinion.verdict == "abstain" and opinion.teacher != "engine":
                abstains[opinion.teacher] = abstains.get(opinion.teacher, 0) + 1

    scorecard_line = ""
    if bank_path and answerer is not None:
        questions = read_bank(bank_path)
        entry = run_scorecard(questions, answerer, answerer_name=answerer_name)
        history_path = out / "scorecard_history.jsonl"
        previous = [h for h in read_history(history_path)
                    if h.get("answerer") == answerer_name]
        append_history(history_path, entry)
        scorecard_line = (f"{entry.answerer}: {entry.correct}/{entry.total} "
                          f"({entry.accuracy:.0%})")
        if previous:
            last = previous[-1]
            last_accuracy = last["correct"] / last["total"] if last["total"] else 0.0
            scorecard_line += f" — previous run {last_accuracy:.0%}"

    report = OvernightReport(
        campaign_id=campaign_id,
        processed_tonight=len(tonight),
        skipped_resumed=skipped,
        remaining_pending=len(todo) - len(tonight),
        bank_accept=sum(r.decision == "bank_accept" for r in unique_results),
        bank_reject=sum(r.decision == "bank_reject" for r in unique_results),
        queued=len(queued),
        abstains_by_teacher=abstains,
        usage_by_teacher={t.name: dict(t.usage_totals)
                          for t in teachers if t.usage_totals},
        scorecard_line=scorecard_line,
        out_dir=str(out),
    )
    (out / "morning_report.md").write_text(
        render_morning_report(report, unique_results, teachers),
        encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# The morning report
# ---------------------------------------------------------------------------

def render_morning_report(report: OvernightReport,
                          results: list[PanelResult],
                          teachers: list[Teacher],
                          *, max_queue_preview: int = 15) -> str:
    stamp = time.strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Morning report — {report.campaign_id}",
        "",
        f"_{stamp} · teachers: "
        + (", ".join(f"{t.name} ({t.model})" for t in teachers) or "none")
        + "_",
        "",
        f"- Items asked tonight: **{report.processed_tonight}**"
        + (f" (+{report.skipped_resumed} already done, resumed)"
           if report.skipped_resumed else ""),
        f"- Banked automatically: **{report.bank_accept} accepted**, "
        f"**{report.bank_reject} rejected** (hard negatives)",
        f"- Waiting for you: **{report.queued}** (pre-answered below)",
    ]
    if report.remaining_pending:
        lines.append(f"- Still pending beyond tonight's cap: "
                     f"{report.remaining_pending}")
    if report.scorecard_line:
        lines += ["", f"**Nightly scorecard:** {report.scorecard_line}"]
    if report.abstains_by_teacher:
        lines += ["", "## Teacher health", ""]
        for name, count in sorted(report.abstains_by_teacher.items()):
            lines.append(f"- {name}: **{count} abstention(s)** — check its "
                         "key/server if this is most of the night")
    if report.usage_by_teacher:
        lines += ["", "## Spend", ""]
        for name, usage in sorted(report.usage_by_teacher.items()):
            spent = ", ".join(f"{k}={v}" for k, v in sorted(usage.items()))
            lines.append(f"- {name}: {spent}")

    queued = [r for r in results if not r.banked]
    if queued:
        lines += ["", f"## Your morning questions ({len(queued)})", ""]
        for result in queued[:max_queue_preview]:
            item = result.item
            lines.append(f"### {item.subject!r} — {item.item_type}, "
                         f"session {item.session_id}")
            lines.append(f"_{result.gate_note}_")
            for opinion in result.opinions:
                verdict = opinion.verdict.upper()
                choice = f" → {opinion.choice}" if opinion.choice else ""
                lines.append(f"- **{opinion.teacher}**: {verdict}{choice}"
                             + (f" — {opinion.reason}" if opinion.reason else ""))
            lines.append("")
        if len(queued) > max_queue_preview:
            lines.append(f"…and {len(queued) - max_queue_preview} more in "
                         "morning_queue.jsonl")

    lines += [
        "", "## Next steps", "",
        "1. Answer the queue: `python -m transcripts_ai reviews --campaign "
        f'"{report.campaign_id}"` (the six actions apply).',
        "2. Retrain the learner on your new answers: "
        "`python -m transcripts_ai train`.",
        "3. Re-run the loop tomorrow night — banked records accumulate in "
        f"`{report.out_dir}`.",
    ]
    return "\n".join(lines) + "\n"
