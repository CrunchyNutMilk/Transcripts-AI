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
    usage: dict[str, dict[str, int]] = field(default_factory=dict)

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
                   processed=list(data.get("processed", [])),
                   usage={k: dict(v) for k, v in data.get("usage", {}).items()})

    def save(self, out_dir: Path) -> None:
        path = out_dir / "state.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(
            {"campaign_id": self.campaign_id, "processed": self.processed,
             "usage": self.usage},
            indent=0, sort_keys=True), encoding="utf-8")
        tmp.replace(path)      # atomic: a crash never truncates the state

    def merge_usage(self, teachers: list[Teacher]) -> None:
        """Fold this run's spend into the whole-night totals."""
        for teacher in teachers:
            if not teacher.usage_totals:
                continue
            bucket = self.usage.setdefault(teacher.name, {})
            for key, value in teacher.usage_totals.items():
                bucket[key] = bucket.get(key, 0) + value


class OutDirLock:
    """One overnight run per out-dir. A wedged-looking run that gets
    'resumed' in a second terminal would double-pay every item and
    interleave torn lines; O_EXCL makes that a clear error instead."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / "overnight.lock"
        self._fd: int | None = None

    def __enter__(self) -> "OutDirLock":
        import os
        try:
            self._fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(self._fd, str(os.getpid()).encode())
        except FileExistsError:
            owner = ""
            try:
                owner = self.path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
            raise RuntimeError(
                f"another overnight run owns this out-dir (lock "
                f"{self.path}, pid {owner or '?'}); if that run is dead, "
                f"delete the lock file and retry") from None
        return self

    def __exit__(self, *exc) -> None:
        import os
        if self._fd is not None:
            os.close(self._fd)
        self.path.unlink(missing_ok=True)


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


def read_results(path: Path) -> tuple[list[PanelResult], int]:
    """(results, corrupt line count). Torn lines — a crash or power loss
    mid-append — are skipped, never fatal: the torn item was never marked
    processed, so a resume re-asks it. That is the promised at-most-one
    re-pay, not a bricked out-dir."""
    if not path.exists():
        return [], 0
    results: list[PanelResult] = []
    corrupt = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                results.append(result_from_json(line))
            except (json.JSONDecodeError, KeyError, TypeError):
                corrupt += 1
    return results, corrupt


def _append_result(path: Path, result: PanelResult) -> None:
    """Append one verdict, healing a torn tail first: if the file does not
    end with a newline (crash mid-append), write one so the new verdict
    can never merge into the torn fragment."""
    needs_newline = False
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            needs_newline = f.read(1) != b"\n"
    with open(path, "a", encoding="utf-8") as f:
        if needs_newline:
            f.write("\n")
        f.write(result_to_json(result) + "\n")


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

@dataclass
class OvernightReport:
    campaign_id: str
    processed_tonight: int
    skipped_resumed: int
    remaining_pending: int
    bank_accept: int              # whole night, live pending items only
    bank_reject: int
    queued: int                   # still awaiting the human RIGHT NOW
    abstains_by_teacher: dict[str, int]
    usage_by_teacher: dict[str, dict[str, int]]   # whole night, all runs
    scorecard_line: str = ""
    out_dir: str = ""
    corrupt_lines: int = 0


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
    if len(teachers) < min_votes:
        raise ValueError(
            f"{len(teachers)} teacher(s) configured but min_votes={min_votes}:"
            " nothing could ever bank — add teachers or lower --min-votes"
            " before spending a night")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with OutDirLock(out):
        return _run_overnight_locked(
            memory, campaign_id, teachers, out,
            session_id=session_id, max_items=max_items, min_votes=min_votes,
            bank_path=bank_path, answerer=answerer,
            answerer_name=answerer_name, on_progress=on_progress)


def _run_overnight_locked(
    memory: CampaignMemory,
    campaign_id: str,
    teachers: list[Teacher],
    out: Path,
    *,
    session_id: str | None,
    max_items: int,
    min_votes: int,
    bank_path: str | None,
    answerer,
    answerer_name: str,
    on_progress,
) -> OvernightReport:
    state = OvernightState.load(out, campaign_id)
    results_path = out / "results.jsonl"

    pending = memory.pending_reviews(campaign_id, session_id=session_id)
    pending_ids = {i.item_id for i in pending}
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
        _append_result(results_path, result)
        state.processed.append(item.item_id)
        state.merge_usage(teachers)
        for teacher in teachers:
            teacher.usage_totals = {}      # merged; don't double-count
        state.save(out)
        on_progress(f"[{index}/{len(tonight)}] {result.decision:12s} "
                    f"{item.subject[:48]!r}")

    # Outputs regenerate from the FULL results file, so a resumed night's
    # banked/queue files cover every item asked — but queue rows for items
    # a human has since resolved are retired, not re-served every morning.
    results, corrupt = read_results(results_path)
    if corrupt:
        on_progress(f"WARNING: skipped {corrupt} torn line(s) in "
                    f"{results_path.name} (crash mid-write); the affected "
                    "item(s) will be re-asked")
    seen_ids: set[str] = set()
    unique_results = []
    for result in results:
        if result.item.item_id not in seen_ids:
            seen_ids.add(result.item.item_id)
            unique_results.append(result)
    live_results = [r for r in unique_results
                    if r.banked or r.item.item_id in pending_ids]
    banked = records_from_panel(unique_results)
    queued = queue_from_panel(live_results)
    write_records(out / "banked_records.jsonl", banked)
    with open(out / "morning_queue.jsonl", "w", encoding="utf-8") as f:
        for row in queued:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    state.merge_usage(teachers)     # any spend since the last per-item save
    for teacher in teachers:
        teacher.usage_totals = {}
    state.save(out)

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
        usage_by_teacher={k: dict(v) for k, v in state.usage.items()},
        scorecard_line=scorecard_line,
        out_dir=str(out),
        corrupt_lines=corrupt,
    )
    (out / "morning_report.md").write_text(
        render_morning_report(report, live_results, teachers),
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
        f"- Items asked this run: **{report.processed_tonight}**"
        + (f" (+{report.skipped_resumed} done in earlier runs of this "
           "out-dir)" if report.skipped_resumed else ""),
        f"- Banked across the whole out-dir: **{report.bank_accept} "
        f"accepted**, **{report.bank_reject} rejected** (hard negatives)",
        f"- Waiting for you right now: **{report.queued}** (pre-answered "
        "below; items you already resolved are retired)",
    ]
    if report.corrupt_lines:
        lines.append(f"- ⚠ {report.corrupt_lines} torn line(s) in "
                     "results.jsonl were skipped (crash mid-write); the "
                     "affected items will be re-asked next run")
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
                # Provider-error abstain reasons can carry newlines and
                # JSON bodies; one flattened, bounded line keeps the
                # report readable.
                reason = " ".join(opinion.reason.split())[:160]
                lines.append(f"- **{opinion.teacher}**: {verdict}{choice}"
                             + (f" — {reason}" if reason else ""))
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
