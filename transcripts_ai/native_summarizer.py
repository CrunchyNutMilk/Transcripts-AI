"""Native session summariser — extractive, no external AI, no generation.

Builds the SessionSummary entirely from evidence already in hand: native
facts, detected game events, scene structure and campaign memory. Because
every line of the summary is copied or templated from a sourced fact, the
summary cannot hallucinate — its weakness is prose elegance, not accuracy.
Confirmed vs uncertain separation falls directly out of each fact's
epistemic status and time status.
"""
from __future__ import annotations

from collections import Counter

from .dnd_patterns import DetectedEvent, GameEvent, detect_events
from .memory import CampaignMemory
from .scenes import Scene, SceneType
from .schemas import (
    ChangeType,
    EntityKind,
    EpistemicStatus,
    Fact,
    FactCategory,
    SessionSummary,
    SummarySection,
    TimeStatus,
)
from .summarizer import SECTION_TITLES
from .transcript import ParsedTranscript

NATIVE_SUMMARIZER_VERSION = "native-summarizer-v1"

_CONFIRMED_RANK = EpistemicStatus.DM_HINT.rank  # this rank or stronger => confirmed


def _line(fact: Fact) -> str:
    ref = f" [line {fact.provenance.line_start}]"
    return fact.statement + ref


def build_native_summary(
    memory: CampaignMemory,
    *,
    campaign_id: str,
    session_id: str,
    game_name: str,
    session_date: str,
    parsed: ParsedTranscript,
    facts: list[Fact],
    scenes: list[Scene],
    manifest_hash: str,
) -> SessionSummary:
    events = detect_events(parsed.entries)
    sections = {title: SummarySection(title=title) for title in SECTION_TITLES}

    def put(title: str, fact: Fact) -> None:
        section = sections[title]
        happened = fact.time_status in (TimeStatus.HAPPENED, TimeStatus.CURRENT)
        confirmed = fact.status.rank <= _CONFIRMED_RANK and happened
        (section.confirmed if confirmed else section.uncertain).append(_line(fact))

    ordered = sorted(facts, key=lambda f: f.provenance.line_start)
    for fact in ordered:
        if fact.time_status is TimeStatus.PLANNED:
            put("Plans", fact)
        elif fact.category is FactCategory.QUEST:
            put("Quests", fact)
        elif fact.category is FactCategory.LOOT:
            put("Loot", fact)
        elif fact.category in (FactCategory.COMBAT, FactCategory.CONDITION):
            put("Combat", fact)
        elif fact.category in (FactCategory.NPC, FactCategory.ALIAS):
            put("Important Dialogue & Revealed Information", fact)
        elif fact.relationship == "long_rest":
            pass  # rendered once, under Decisions Made below
        else:
            put("Key Events", fact)

    # Unresolved questions: open contradictions + disputed facts.
    risks = sections["Risks & Unresolved Questions"]
    for record in memory.open_contradictions(campaign_id):
        risks.uncertain.append(
            f"Unresolved contradiction between facts {record.fact_id} and "
            f"{record.conflicting_fact_id}: {record.note}"
        )
    for fact in ordered:
        if fact.needs_review:
            risks.uncertain.append(f"Needs review: {_line(fact)}")

    # Decisions: explicit party choices are hard to pattern-match reliably;
    # only long-rest / travel commitments land here, the rest stays for the
    # reviewed summary. Never invent content for empty sections.
    decisions = sections["Decisions Made"]
    for fact in ordered:
        if fact.relationship in ("travelled_to", "long_rest") and \
                fact.time_status is TimeStatus.HAPPENED:
            decisions.confirmed.append(_line(fact))

    # Vault update suggestions: entities with facts but no vault page yet.
    vault = sections["Vault Update Suggestions"]
    known = {e.name.casefold(): e for e in memory.entities(campaign_id)}
    suggested: set[str] = set()
    for fact in ordered:
        for entity in fact.entities:
            folded = entity.casefold()
            if folded in suggested:
                continue
            record = known.get(folded)
            if record is None:
                vault.uncertain.append(
                    f"Possible new entity page: {entity} (from fact {fact.fact_id})"
                )
                suggested.add(folded)
            elif record.vault_path is None:
                vault.uncertain.append(
                    f"{entity} has campaign memory but no linked vault page"
                )
                suggested.add(folded)

    # Combat extras from raw events (initiative fact already included).
    combat = sections["Combat"]
    combat_counts = Counter(e.event for e in events)
    if combat_counts.get(GameEvent.COMBAT_START):
        combat.confirmed.append(
            f"Combat broke out {combat_counts[GameEvent.COMBAT_START]} time(s)"
        )
    if not any(e.event == GameEvent.INITIATIVE for e in events):
        combat.uncertain.append("Initiative order: not stated in transcript")

    # Party & NPCs from memory + this session's speakers.
    speakers = {s.casefold() for s in parsed.speakers}
    party = sorted(
        e.name for e in memory.entities(campaign_id)
        if e.kind is EntityKind.PC and e.name.casefold() in speakers
    )

    scene_mix = Counter(s.scene_type for s in scenes)
    key = sections["Key Events"]
    if not key.confirmed and not key.uncertain:
        key.uncertain.append(
            "No pattern-extractable key events; session was "
            + ", ".join(f"{t.value} ({c} scenes)" for t, c in scene_mix.most_common(3))
        )

    return SessionSummary(
        campaign_id=campaign_id,
        session_id=session_id,
        game_name=game_name,
        session_date=session_date,
        party=party,
        sections=list(sections.values()),
        manifest_hash=manifest_hash,
        generator=NATIVE_SUMMARIZER_VERSION,
    )
