"""The time-travel test: is campaign memory actually compounding?

docs/TRAINING.md's most direct measurement. Build memory from sessions
1..N, then look at session N+1 COLD — before ingesting it — and ask: of
the new names this session mentions, how many does accumulated memory
already recognise or plausibly suggest? Repeat across the whole campaign
and the recognition rate over time IS the "memory compounds" claim, as a
curve instead of a feeling.

The human's role is simulated by an explicit oracle: after measuring a
session, its entity candidates are auto-approved into memory the way the
real reviewer would approve them (``--oracle`` in the CLI, on by
default and stamped into every result). Without an approval step memory
would never grow and the curve would measure nothing; with it, the curve
measures exactly what the resolver + phonetics + learned aliases can do
with an attentive reviewer behind them.

Everything runs in a throwaway working DB named by the caller — never a
real campaign memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..memory import CampaignMemory
from ..pipeline import SessionPipeline
from ..resolver import BandAction, NameResolver
from ..schemas import EntityKind, EntityRecord
from ..detector import detect_names
from ..transcript import parse_transcript

_DATE_IN_NAME = re.compile(r"(\d{8})")


def _valid_dates_in(name: str) -> list[str]:
    """8-digit runs that are plausible YYYYMMDD dates — epoch timestamps,
    uuid fragments and counters ('1699843200', '12345678') never qualify."""
    dates = []
    for raw in _DATE_IN_NAME.findall(name):
        year, month, day = int(raw[:4]), int(raw[4:6]), int(raw[6:])
        if 1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
            dates.append(raw)
    return dates


def session_date_of(path: str | Path) -> str | None:
    """The session's YYYYMMDD from its filename, or None. When several
    valid dates appear, the LAST wins (prefixes come first in exports)."""
    dates = _valid_dates_in(Path(path).name)
    return dates[-1] if dates else None


def session_order_key(path: str | Path) -> tuple[str, str]:
    """Order transcripts chronologically by the validated date in the
    filename; dateless files sort last, by name."""
    name = Path(path).name
    return (session_date_of(path) or "99999999", name)


@dataclass
class StepMetrics:
    session_id: str
    candidates: int          # new-name candidates detected in the cold look
    auto_linked: int         # resolver already certain (AUTO_LINK band)
    suggested: int           # plausible match offered (SUGGEST band)
    unknown: int             # memory had nothing
    known_mentions: int      # mentions of already-known entities in the text

    @property
    def recognised_share(self) -> float | None:
        """None when nothing was measured — a session with zero new-name
        candidates must never render as a flattering 100%."""
        total = self.candidates
        return (self.auto_linked + self.suggested) / total if total else None


def _session_id_for(path: Path) -> str:
    raw = session_date_of(path)
    if raw:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return path.stem


def _register_speakers_as_pcs(memory: CampaignMemory, campaign_id: str,
                              speakers: list[str]) -> None:
    for speaker in speakers:
        name = speaker.strip()
        if not name or name.casefold() in {"dm", "gm", "dungeon master",
                                           "unknown"}:
            continue
        memory.upsert_entity(
            EntityRecord(name=name, kind=EntityKind.PC,
                         campaign_id=campaign_id),
            actor="engine:time-travel-cast",
        )


def _measure_cold(memory: CampaignMemory, campaign_id: str,
                  text: str) -> StepMetrics:
    parsed = parse_transcript(text, source_path="<cold>")
    known = frozenset(e.name.casefold()
                      for e in memory.entities(campaign_id))
    resolver = NameResolver(memory)
    metrics = StepMetrics(session_id="", candidates=0, auto_linked=0,
                          suggested=0, unknown=0, known_mentions=0)
    for candidate in detect_names(parsed.entries, known_names=known):
        if set(candidate.reasons) == {"known_entity_mention"}:
            metrics.known_mentions += candidate.mentions
            continue
        metrics.candidates += 1
        best = resolver.resolve(campaign_id, candidate.text).best
        if best is None:
            metrics.unknown += 1
        elif best.band is BandAction.AUTO_LINK:
            metrics.auto_linked += 1
        elif best.band is BandAction.SUGGEST:
            metrics.suggested += 1
        else:
            metrics.unknown += 1
    return metrics


def _oracle_approve(memory: CampaignMemory, campaign_id: str,
                    session_id: str) -> int:
    """Approve this session's entity candidates the way the reviewer would.

    Candidates whose best suggestion is a plausible existing entity become
    aliases of it; the rest become new NPC entities. Explicitly a
    simulation — real campaigns keep the human in this seat.
    """
    from ..schemas import AliasRecord, ReviewAction

    resolver = NameResolver(memory)
    approved = 0
    for item in memory.pending_reviews(campaign_id, session_id=session_id):
        if item.item_type != "entity":
            continue
        best = resolver.resolve(campaign_id, item.subject).best
        if best is not None and best.band in (BandAction.AUTO_LINK,
                                              BandAction.SUGGEST):
            memory.add_alias(
                AliasRecord(campaign_id=campaign_id, observed=item.subject,
                            canonical=best.canonical,
                            entity_id=best.entity_id,
                            approved_by="oracle:time-travel"),
                actor="oracle:time-travel")
        else:
            memory.upsert_entity(
                EntityRecord(name=item.subject, kind=EntityKind.NPC,
                             campaign_id=campaign_id),
                actor="oracle:time-travel")
        memory.resolve_review(campaign_id, item.item_id,
                              action=ReviewAction.CORRECT,
                              actor="oracle:time-travel")
        approved += 1
    return approved


def run_time_travel(
    transcript_paths: list[str | Path],
    *,
    campaign_id: str,
    db_path: str | Path,
    game_name: str = "",
    on_progress=print,
) -> list[StepMetrics]:
    db = Path(db_path)
    if db.exists() and db.stat().st_size:
        # Stale memory fakes the curve; a real campaign DB must never be
        # grown by an eval. The eval owns its DB from birth, or not at all.
        raise ValueError(
            f"{db} already exists — the time-travel eval needs a FRESH "
            "throwaway database (pick a new --db path)")
    paths = sorted((Path(p) for p in transcript_paths), key=session_order_key)
    memory = CampaignMemory(db_path)
    steps: list[StepMetrics] = []
    try:
        pipeline = SessionPipeline(memory)
        for path in paths:
            session_id = _session_id_for(path)
            text = path.read_text(encoding="utf-8-sig")
            metrics = _measure_cold(memory, campaign_id, text)
            metrics.session_id = session_id
            steps.append(metrics)
            on_progress(
                f"{session_id}: {metrics.candidates} new-name candidate(s) — "
                f"{metrics.auto_linked} certain / {metrics.suggested} suggested"
                f" / {metrics.unknown} unknown; "
                f"{metrics.known_mentions} known-name mention(s)")
            # Grow memory: cast, facts, then oracle-approved candidates.
            parsed = parse_transcript(text, source_path=str(path))
            _register_speakers_as_pcs(memory, campaign_id, parsed.speakers)
            pipeline.process_session_native(
                campaign_id=campaign_id, session_id=session_id,
                transcript_path=path,
                game_name=game_name or campaign_id, session_date=session_id)
            _oracle_approve(memory, campaign_id, session_id)
    finally:
        memory.close()
    return steps


def render_report(steps: list[StepMetrics], *, campaign_id: str) -> str:
    lines = [
        f"# Time-travel test — {campaign_id}",
        "",
        "Each row looks at that session COLD, knowing only what memory",
        "accumulated from the sessions above it (reviewer simulated by the",
        "oracle described in `transcripts_ai/lab/timetravel.py`).",
        "",
        "| Session | New names | Recognised | Suggested | Unknown | Recognition | Known mentions |",
        "|---|---|---|---|---|---|---|",
    ]
    for step in steps:
        share_cell = (f"{step.recognised_share:.0%}"
                      if step.recognised_share is not None else "n/a")
        lines.append(
            f"| {step.session_id} | {step.candidates} | {step.auto_linked} "
            f"| {step.suggested} | {step.unknown} "
            f"| {share_cell} | {step.known_mentions} |")
    if len(steps) >= 4:
        early = steps[1:max(2, len(steps) // 3)]
        late = steps[-max(2, len(steps) // 3):]

        def share(chunk):
            """None when the chunk measured nothing — an empty denominator
            must never manufacture a 100% conclusion."""
            recognised = sum(s.auto_linked + s.suggested for s in chunk)
            total = sum(s.candidates for s in chunk)
            return recognised / total if total else None

        early_share, late_share = share(early), share(late)
        early_txt = f"{early_share:.0%}" if early_share is not None else "n/a"
        late_txt = f"{late_share:.0%}" if late_share is not None else "n/a"
        lines += [
            "",
            f"**Early sessions recognition:** {early_txt}   "
            f"**Late sessions recognition:** {late_txt}",
            "",
            "If the late number is meaningfully higher, memory is",
            "compounding: the campaign's returning names resolve on sight",
            "and only genuinely new ones need the reviewer.",
        ]
    return "\n".join(lines) + "\n"
