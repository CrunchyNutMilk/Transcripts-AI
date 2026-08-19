"""Versioned training records and session-level dataset splits.

The flywheel's data contract (docs/TRAINING.md → "The data flywheel"):
every human decision becomes one immutable JSONL record carrying the
system's proposal, the human's verdict, and provenance — rejections
included, because hard negatives are training data too.

Two rules are enforced in code, not by convention:

- **Versioned schema.** Records carry ``record_version``; readers fail
  closed on versions they do not understand instead of silently
  misparsing months of archive.
- **Splits are by session, never by item.** Facts repeat across a
  campaign; item-level splits leak between train and eval and flatter
  every score computed afterwards. Frozen sessions always land in eval.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..memory import CampaignMemory

RECORD_SCHEMA_VERSION = 1

# feedback-ledger kind -> (item_type, decision, accepted-means)
_KIND_MAP = {
    "correction_accepted": ("spelling", "correct"),
    "correction_rejected": ("spelling", "reject"),
    "alias_added": ("entity", "alias"),
    "entity_confirmed": ("entity", "new"),
    "entity_misclassified": ("entity", "not_entity"),
    "fact_missed": ("fact", "missed"),
    "fact_false_positive": ("fact", "false_positive"),
    "summary_corrected": ("summary", "corrected"),
    "vault_link_chosen": ("entity", "vault_link"),
    "deferred": ("entity", "defer"),
}


class RecordError(ValueError):
    """Raised on malformed or unsupported training records."""


@dataclass
class TrainingRecord:
    campaign_id: str
    session_id: str
    item_type: str            # spelling | entity | fact | summary
    subject: str              # the text the decision was about
    proposal: str             # what the system proposed ("" if nothing)
    decision: str             # correct | reject | alias | new | not_entity | ...
    choice: str | None        # target name for correct/alias, else None
    accepted: bool            # did the human accept the system's proposal?
    evidence_quote: str = ""
    source: str = "feedback-ledger"
    checker_version: str = ""
    actor: str = ""
    created_at: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)
    record_version: int = RECORD_SCHEMA_VERSION

    def validate(self) -> None:
        if self.record_version != RECORD_SCHEMA_VERSION:
            raise RecordError(
                f"record_version {self.record_version} unsupported "
                f"(this reader understands {RECORD_SCHEMA_VERSION})"
            )
        if not self.campaign_id or not self.session_id:
            raise RecordError("record requires campaign_id and session_id")
        if not self.subject:
            raise RecordError("record requires a subject")
        if not self.decision:
            raise RecordError("record requires a decision")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "TrainingRecord":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RecordError(f"not valid JSON: {exc}") from exc
        version = data.get("record_version")
        if version != RECORD_SCHEMA_VERSION:
            raise RecordError(
                f"record_version {version!r} unsupported "
                f"(this reader understands {RECORD_SCHEMA_VERSION})"
            )
        record = cls(**data)
        record.validate()
        return record


def write_records(path: str | Path, records: Iterable[TrainingRecord]) -> int:
    """Write records as JSONL (overwrites). Returns the count written."""
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            record.validate()
            f.write(record.to_json() + "\n")
            count += 1
    return count


def read_records(path: str | Path) -> list[TrainingRecord]:
    """Read a JSONL archive; fails closed on any malformed line."""
    records: list[TrainingRecord] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(TrainingRecord.from_json(line))
            except RecordError as exc:
                raise RecordError(f"{path} line {number}: {exc}") from exc
    return records


def _parse_arrow_subject(subject: str) -> tuple[str, str | None]:
    """'Gomra->Ghomra' -> ('Gomra', 'Ghomra'); plain subjects pass through."""
    if "->" in subject:
        observed, _, target = subject.partition("->")
        return observed.strip(), target.strip() or None
    return subject.strip(), None


def export_from_feedback(memory: CampaignMemory, campaign_id: str) -> list[TrainingRecord]:
    """Convert the campaign's feedback ledger into training records.

    Every row is exported — accepted and rejected alike. Rows whose kind is
    unknown are kept with item_type 'other' rather than dropped, so nothing
    in the archive silently vanishes.
    """
    records: list[TrainingRecord] = []
    for row in memory.all_feedback(campaign_id):
        kind = row["kind"]
        item_type, decision = _KIND_MAP.get(kind, ("other", kind))
        subject, choice = _parse_arrow_subject(row["subject"])
        try:
            extras = json.loads(row.get("detail_json") or "{}")
        except json.JSONDecodeError:
            extras = {"detail_raw": row.get("detail_json")}
        records.append(
            TrainingRecord(
                campaign_id=campaign_id,
                session_id=row["session_id"],
                item_type=item_type,
                subject=subject,
                proposal=row["subject"],
                decision=decision,
                choice=choice,
                accepted=bool(row["accepted"]),
                source="feedback-ledger",
                actor=row.get("actor") or "",
                created_at=float(row.get("created_at") or 0.0) or time.time(),
                extras=extras,
            )
        )
    return records


def tally(records: list[TrainingRecord]) -> dict[str, Counter]:
    """Distribution audit: counts by item_type, decision, and session."""
    return {
        "by_item_type": Counter(r.item_type for r in records),
        "by_decision": Counter(r.decision for r in records),
        "by_session": Counter(r.session_id for r in records),
        "by_accepted": Counter("accepted" if r.accepted else "rejected" for r in records),
    }


# ---------------------------------------------------------------------------
# Session-level splitting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSplit:
    train_sessions: tuple[str, ...]
    eval_sessions: tuple[str, ...]
    frozen_sessions: tuple[str, ...]   # subset of eval, never trained on, ever

    def side_of(self, session_id: str) -> str:
        if session_id in self.eval_sessions:
            return "eval"
        if session_id in self.train_sessions:
            return "train"
        return "unassigned"


def split_sessions(
    session_ids: Iterable[str],
    *,
    frozen: Iterable[str] = (),
    eval_fraction: float = 0.2,
    seed: str = "transcripts-ai-v1",
) -> SessionSplit:
    """Deterministic session-level split.

    Frozen sessions always land in eval. Remaining sessions are ordered by a
    seeded hash (stable across runs and machines — no RNG state to lose) and
    the top ``eval_fraction`` join eval. The same inputs always produce the
    same split, so a scorecard from last month is comparable to today's.
    """
    if not 0.0 <= eval_fraction < 1.0:
        raise RecordError("eval_fraction must be in [0, 1)")
    frozen_set = {s for s in frozen}
    known = set(session_ids)
    unknown_frozen = frozen_set - known
    if unknown_frozen:
        raise RecordError(f"frozen sessions not in data: {sorted(unknown_frozen)}")
    everything = sorted(known)

    def rank(session_id: str) -> str:
        return hashlib.sha256(f"{seed}:{session_id}".encode()).hexdigest()

    movable = [s for s in everything if s not in frozen_set]
    movable.sort(key=rank)
    extra_eval_count = max(0, round(eval_fraction * len(everything)) - len(frozen_set))
    extra_eval = set(movable[:extra_eval_count])
    eval_sessions = tuple(sorted(frozen_set | extra_eval))
    train_sessions = tuple(s for s in everything if s not in eval_sessions)
    return SessionSplit(
        train_sessions=train_sessions,
        eval_sessions=eval_sessions,
        frozen_sessions=tuple(sorted(frozen_set)),
    )


def split_records(
    records: list[TrainingRecord], split: SessionSplit
) -> tuple[list[TrainingRecord], list[TrainingRecord]]:
    """Apply a session split to records: (train, eval). No item leaks."""
    train = [r for r in records if split.side_of(r.session_id) == "train"]
    evaluation = [r for r in records if split.side_of(r.session_id) == "eval"]
    return train, evaluation
