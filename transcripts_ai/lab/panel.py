"""The teacher panel: many opinions, one deterministic gate, zero authority.

Layer 2a of docs/TRAINING.md. The panel takes the engine's pending
review items — the questions the human reviewer would otherwise answer by hand — and
asks every configured teacher plus the engine's own resolver for a
verdict. What happens next is decided by rules, not by any model:

- **Unanimous accept** on the same choice, *and* the choice passes the
  evidence gate (it must already be a known name in campaign memory) →
  banked as a training record.
- **Unanimous reject** → banked as a hard negative.
- **Anything else** — a split, an uncertain vote, too few voices, or a
  gate failure — goes to the morning queue with every opinion attached,
  so the human reviews a pre-answered question instead of a blank one.

Safety properties, enforced here and tested:

- The panel NEVER writes to campaign memory. Banked records and queue
  entries are JSONL files; review items stay unresolved until a human
  resolves them.
- Banked records carry ``actor="panel:..."`` and ``source="teacher-panel"``
  — they can never be mistaken for human decisions, so the scorecard's
  question bank (human rows only) stays uncontaminated.
- The gate means teachers cannot launder an invented name into the
  archive even by agreeing on it.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..memory import CampaignMemory
from ..providers import ProviderError, ValidationFailed, call_role
from ..resolver import BandAction, NameResolver
from ..schemas import ReviewItem
from .records import TrainingRecord
from .teachers import Teacher

ENGINE_NAME = "engine"
NAME_ITEM_TYPES = ("spelling", "entity")   # where the resolver's vote means something
MIN_VOTES_TO_BANK = 2          # TEACHER votes; the engine never counts toward quorum
DEFAULT_ITEM_LIMIT = 25        # cost guard: explicit --limit to raise

VERDICTS = ("accept", "reject", "uncertain", "abstain")

# Provider error text can echo request credentials (e.g. a 401 body quoting
# the offending key). Anything persisted to queue/banked files goes through
# this first.
_SECRET_RE = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|AIza[0-9A-Za-z_\-]{10,}|Bearer\s+[A-Za-z0-9._\-]{8,}"
    r"|x-api-key[\"':\s=]+[A-Za-z0-9._\-]{8,})"
)


def redact_secrets(text: str) -> str:
    return _SECRET_RE.sub("[redacted]", text)

PANEL_SYSTEM = """\
You are one voice on a review panel for a Dungeons & Dragons transcript system.
The panel checks proposals made by an automatic engine against evidence quotes
from the session transcript. Rules you must follow:
- The evidence quotes are your ONLY source of truth about this campaign.
- Never invent a name. If you vote accept, your choice MUST be copied exactly
  from the KNOWN NAMES list.
- If the evidence is not enough to decide, vote uncertain. That is a good
  answer, not a failure.
- Reply with ONLY a JSON object: {"verdict": "accept" | "reject" | "uncertain",
  "choice": "<name from KNOWN NAMES, or empty>", "reason": "<one short sentence>"}
"""

_QUESTION_BY_TYPE = {
    "spelling": ('Should "{subject}" be corrected to one of the KNOWN NAMES, '
                 "or is it fine as ordinary text? Accept = correct it to your "
                 "choice. Reject = leave it alone."),
    "entity": ('Is "{subject}" a real character, place or thing in this '
               "campaign? Accept = yes, and your choice is its proper name "
               "from KNOWN NAMES. Reject = it is an ordinary word or a "
               "transcription error, not an entity."),
    "fact": ("Do the evidence quotes support this statement: {subject!r}? "
             "Accept = clearly supported. Reject = contradicted or absent."),
    "summary_claim": ("Do the evidence quotes support this summary claim: "
                      "{subject!r}? Accept = supported. Reject = not supported."),
}


def _verdict_validator(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["reply must be a single JSON object"]
    problems = []
    if payload.get("verdict") not in ("accept", "reject", "uncertain"):
        problems.append('verdict must be "accept", "reject" or "uncertain"')
    if not isinstance(payload.get("choice", ""), str):
        problems.append("choice must be a string")
    if not isinstance(payload.get("reason", ""), str):
        problems.append("reason must be a string")
    return problems


@dataclass
class Opinion:
    teacher: str                   # "engine" or a teacher name
    verdict: str                   # accept | reject | uncertain | abstain
    choice: str = ""
    reason: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {self.verdict!r}")


# ---------------------------------------------------------------------------
# Prompt building (deterministic; also what --dry-run shows)
# ---------------------------------------------------------------------------

def known_names_for(memory: CampaignMemory, campaign_id: str, item: ReviewItem,
                    *, limit: int = 6) -> list[str]:
    """Candidate canonical names the teachers may choose from."""
    resolver = NameResolver(memory)
    names: list[str] = []
    seen: set[str] = set()
    suggested = [s.get("canonical", "") for s in item.suggestions]
    resolved = [s.canonical for s in
                resolver.resolve(campaign_id, item.subject, limit=limit).suggestions]
    for name in suggested + resolved:
        folded = name.casefold().strip()
        if name and folded not in seen:
            seen.add(folded)
            names.append(name)
    return names[:limit]


def build_panel_prompt(item: ReviewItem, known_names: list[str]) -> str:
    from ..dnd5e import nearest_official

    template = _QUESTION_BY_TYPE.get(item.item_type, _QUESTION_BY_TYPE["entity"])
    lines = [
        f"ITEM TYPE: {item.item_type}",
        f"QUESTION: {template.format(subject=item.subject)}",
        f"ENGINE'S CONCERN: {item.reason}",
    ]
    official = nearest_official(item.subject)
    if official is not None:
        exact = official.score >= 0.999
        lines.append(
            f"OFFICIAL 5e REFERENCE: the subject "
            + ("exactly matches" if exact else
               f"resembles ({official.score:.0%})")
            + f' the official {official.category} "{official.official}".')
    lines += [
        "",
        "KNOWN NAMES (the only allowed values for choice):",
    ]
    lines += [f"- {name}" for name in known_names] or ["- (none found)"]
    lines += ["", "EVIDENCE QUOTES:"]
    lines += [f"{i}. {quote}" for i, quote in enumerate(item.evidence, start=1)] \
        or ["(no quotes attached)"]
    if item.suggestions:
        lines += ["", "ENGINE SUGGESTIONS (for context, not binding):"]
        lines += [
            f"- {s.get('canonical', '?')} (score {s.get('score', 0):.2f})"
            for s in item.suggestions[:5]
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------

def engine_opinion(memory: CampaignMemory, campaign_id: str, item: ReviewItem) -> Opinion:
    """The engine's own vote, from the resolver's confidence bands.

    The resolver judges NAMES, so it only votes on spelling/entity items:
    AUTO_LINK-grade evidence → accept; SUGGEST-grade → uncertain (that is
    exactly what the band means); anything weaker → reject. On fact and
    summary items its opinion would be a content-free "that sentence is
    not a name" — so it abstains and leaves those to the teachers.
    """
    if item.item_type not in NAME_ITEM_TYPES:
        return Opinion(teacher=ENGINE_NAME, verdict="abstain", model="native",
                       reason="engine's resolver only judges name items")
    resolution = NameResolver(memory).resolve(campaign_id, item.subject)
    best = resolution.best
    if best is None:
        return Opinion(teacher=ENGINE_NAME, verdict="reject", model="native",
                       reason="no match in campaign memory")
    if best.band is BandAction.AUTO_LINK:
        return Opinion(teacher=ENGINE_NAME, verdict="accept", choice=best.canonical,
                       model="native", reason=best.explanation or "strong match")
    if best.band is BandAction.SUGGEST:
        return Opinion(teacher=ENGINE_NAME, verdict="uncertain", choice=best.canonical,
                       model="native", reason=best.explanation or "plausible match")
    return Opinion(teacher=ENGINE_NAME, verdict="reject", model="native",
                   reason=f"best match {best.canonical!r} below suggestion band")


def teacher_opinion(teacher: Teacher, prompt: str) -> Opinion:
    """One teacher's vote. Errors and unparseable replies become abstentions
    — a flaky teacher can cost the panel a voice, never the run. The broad
    except is that guarantee: whatever a provider throws, the run survives,
    and persisted reasons are redacted so error bodies cannot leak keys."""
    try:
        payload, _ = call_role(
            teacher.provider,
            system=PANEL_SYSTEM,
            user=prompt,
            validator=_verdict_validator,
            max_tokens=teacher.max_tokens,
            usage_sink=teacher.record_usage,
        )
    except (ProviderError, ValidationFailed, Exception) as exc:  # noqa: B014
        return Opinion(teacher=teacher.name, verdict="abstain",
                       model=teacher.model,
                       reason=redact_secrets(f"{type(exc).__name__}: {exc}")[:200])
    return Opinion(
        teacher=teacher.name,
        verdict=payload["verdict"],
        choice=(payload.get("choice") or "").strip(),
        reason=(payload.get("reason") or "").strip()[:300],
        model=teacher.model,
    )


# ---------------------------------------------------------------------------
# The deterministic gate and the decision
# ---------------------------------------------------------------------------

def evidence_gate(memory: CampaignMemory, campaign_id: str, item: ReviewItem,
                  choice: str) -> tuple[bool, str]:
    """Rules the panel cannot vote its way around.

    Every banked accept needs evidence quotes. For NAME items the agreed
    choice must additionally exist in campaign memory already (as an
    entity or an approved alias) — teachers agreeing on an invented name
    is still an invented name. Fact/summary items carry statements, not
    names, so no choice is demanded of them.
    """
    if not item.evidence:
        return False, "item has no evidence quotes"
    if item.item_type not in NAME_ITEM_TYPES:
        return True, "statement item with evidence quotes"
    if not choice:
        return False, "accept votes carried no choice"
    if memory.find_entity(campaign_id, choice) is not None:
        return True, "choice is a known entity"
    alias = memory.resolve_alias(campaign_id, choice)
    if alias is not None:
        return True, f"choice is an approved alias of {alias.canonical}"
    return False, f"{choice!r} is not a known entity or alias in this campaign"


@dataclass
class PanelResult:
    item: ReviewItem
    opinions: list[Opinion]
    decision: str                  # bank_accept | bank_reject | queue
    agreed_choice: str = ""
    gate_note: str = ""

    @property
    def banked(self) -> bool:
        return self.decision.startswith("bank_")


def decide(memory: CampaignMemory, campaign_id: str, item: ReviewItem,
           opinions: list[Opinion], *, min_votes: int = MIN_VOTES_TO_BANK) -> PanelResult:
    """Banking rules. The engine's vote can BLOCK a bank (it is a real
    disagreement signal) but never counts toward the quorum — its reject on
    an unknown name is near-automatic, and quorum met by it would let a
    single real teacher bank records alone."""
    votes = [o for o in opinions if o.verdict in ("accept", "reject")]
    teacher_votes = [o for o in votes if o.teacher != ENGINE_NAME]
    # Only TEACHER uncertainty blocks: the engine's uncertain (its SUGGEST
    # band) is the very reason the item reached the panel — the teachers
    # exist to resolve it, not to be vetoed by it.
    uncertain_teachers = [o for o in opinions
                          if o.verdict == "uncertain" and o.teacher != ENGINE_NAME]
    if len(teacher_votes) < min_votes or uncertain_teachers:
        return PanelResult(item=item, opinions=opinions, decision="queue",
                           gate_note="not enough agreement to bank")
    if all(o.verdict == "accept" for o in votes):
        if item.item_type in NAME_ITEM_TYPES:
            choices = {o.choice.casefold().strip() for o in votes}
            if len(choices) != 1:
                return PanelResult(item=item, opinions=opinions,
                                   decision="queue",
                                   gate_note="accepts disagree on the choice")
            agreed = votes[0].choice.strip()
        else:
            # fact/summary accepts agree by verdict alone — any 'choice'
            # text a teacher volunteered is commentary, not a name.
            agreed = ""
        engine_uncertain = next(
            (o for o in opinions
             if o.teacher == ENGINE_NAME and o.verdict == "uncertain"), None)
        if (engine_uncertain and engine_uncertain.choice
                and engine_uncertain.choice.casefold().strip()
                != agreed.casefold()):
            return PanelResult(item=item, opinions=opinions, decision="queue",
                               agreed_choice=agreed,
                               gate_note="engine suggests a different name")
        ok, note = evidence_gate(memory, campaign_id, item, agreed)
        if not ok:
            return PanelResult(item=item, opinions=opinions, decision="queue",
                               agreed_choice=agreed, gate_note=f"gate: {note}")
        return PanelResult(item=item, opinions=opinions, decision="bank_accept",
                           agreed_choice=agreed, gate_note=note)
    if all(o.verdict == "reject" for o in votes):
        if not item.evidence:
            # A hard negative without evidence is untraceable junk (e.g. the
            # pipeline's own failure items) — a human should look instead.
            return PanelResult(item=item, opinions=opinions, decision="queue",
                               gate_note="gate: hard negative needs evidence quotes")
        return PanelResult(item=item, opinions=opinions, decision="bank_reject",
                           gate_note="unanimous reject")
    return PanelResult(item=item, opinions=opinions, decision="queue",
                       gate_note="split panel")


# ---------------------------------------------------------------------------
# Running the panel
# ---------------------------------------------------------------------------

def panel_item(
    memory: CampaignMemory,
    campaign_id: str,
    item: ReviewItem,
    teachers: list[Teacher],
    *,
    min_votes: int = MIN_VOTES_TO_BANK,
) -> PanelResult:
    """One item through the whole panel. Read-only on memory."""
    known = known_names_for(memory, campaign_id, item)
    prompt = build_panel_prompt(item, known)
    opinions = [engine_opinion(memory, campaign_id, item)]
    opinions += [teacher_opinion(teacher, prompt) for teacher in teachers]
    return decide(memory, campaign_id, item, opinions, min_votes=min_votes)


def run_panel(
    memory: CampaignMemory,
    campaign_id: str,
    items: Iterable[ReviewItem],
    teachers: list[Teacher],
    *,
    min_votes: int = MIN_VOTES_TO_BANK,
) -> list[PanelResult]:
    """Ask the panel about each item. Read-only on memory, by construction:
    everything this touches is a query; results live in the returned list."""
    return [panel_item(memory, campaign_id, item, teachers, min_votes=min_votes)
            for item in items]


# ---------------------------------------------------------------------------
# Outputs: banked training records + the morning queue
# ---------------------------------------------------------------------------

def records_from_panel(results: Iterable[PanelResult]) -> list[TrainingRecord]:
    """Banked panel decisions as training records — never as human ones."""
    records: list[TrainingRecord] = []
    for result in results:
        if not result.banked:
            continue
        item = result.item
        voters = "+".join(o.teacher for o in result.opinions
                          if o.verdict in ("accept", "reject"))
        accepted = result.decision == "bank_accept"
        record = TrainingRecord(
            campaign_id=item.campaign_id,
            session_id=item.session_id,
            item_type=item.item_type,
            subject=item.subject,
            proposal=(item.suggestions[0].get("canonical", "")
                      if item.suggestions else ""),
            decision="accept" if accepted else "reject",
            choice=result.agreed_choice or None,
            accepted=accepted,
            evidence_quote=item.evidence[0] if item.evidence else "",
            source="teacher-panel",
            actor=f"panel:{voters}",
            extras={
                "review_item_id": item.item_id,
                "gate": result.gate_note,
                "opinions": [asdict(o) for o in result.opinions],
            },
        )
        record.validate()
        records.append(record)
    return records


def queue_from_panel(results: Iterable[PanelResult]) -> list[dict]:
    """Morning-queue rows: the un-banked items, pre-answered by the panel."""
    rows = []
    for result in results:
        if result.banked:
            continue
        item = result.item
        rows.append({
            "item_id": item.item_id,
            "campaign_id": item.campaign_id,
            "session_id": item.session_id,
            "item_type": item.item_type,
            "subject": item.subject,
            "reason": item.reason,
            "evidence": item.evidence,
            "why_queued": result.gate_note,
            "opinions": [asdict(o) for o in result.opinions],
        })
    return rows


def write_queue(path: str | Path, rows: list[dict]) -> int:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    return len(rows)


@dataclass
class PanelReport:
    total: int = 0
    bank_accept: int = 0
    bank_reject: int = 0
    queued: int = 0
    usage_by_teacher: dict[str, dict[str, int]] = field(default_factory=dict)


def summarize(results: list[PanelResult], teachers: list[Teacher]) -> PanelReport:
    report = PanelReport(total=len(results))
    for result in results:
        if result.decision == "bank_accept":
            report.bank_accept += 1
        elif result.decision == "bank_reject":
            report.bank_reject += 1
        else:
            report.queued += 1
    report.usage_by_teacher = {
        t.name: dict(t.usage_totals) for t in teachers if t.usage_totals
    }
    return report
