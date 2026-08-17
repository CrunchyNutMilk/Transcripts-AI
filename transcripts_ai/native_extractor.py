"""Native fact extractor — no external AI, no LLM.

Pattern-driven extraction of structured facts from Mapped transcript entries.
Every fact is anchored to the exact line it came from, so evidence is correct
*by construction* (the quote IS the line) — there is nothing to hallucinate.

Coverage is deliberately event-shaped: acquisitions, transfers, deaths,
revivals, travel/arrival, introductions, alias reveals, spell casts, quest
transitions, rests, conditions and damage. Free-form narrative nuance that
patterns cannot reach is left to the review queue rather than guessed at —
that trade (recall for precision) is the point of an evidence-first engine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .detector import COMMON_WORDS, GAME_MECHANIC_WORDS
from .dnd_patterns import (
    WORLD_FACT_MODES,
    assess_entry,
    assess_speaker_mode,
    assess_time_status,
    initiative_order,
)
from .schemas import (
    ChangeType,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    SpeakerMode,
    TimeStatus,
)
from .transcript import TranscriptEntry

NATIVE_EXTRACTOR_VERSION = "native-extractor-v1"

_NAME = r"(?P<name>[A-Z][\w'\-]+(?:\s+(?:of|the|[A-Z][\w'\-]+)){0,3})"
# Items must be either a capitalised name sequence (known entities get recased
# before matching) or an explicit currency amount — free-form lowercase speech
# is far too noisy to treat as an item name.
_ITEM = r"(?P<item>(?:an?|the|some)\s+)?(?P<item_name>[A-Z][\w'\-]+(?:\s+(?:of|the|[A-Z][\w'\-]+)){0,4}|\d+\s+(?:gold|silver|copper|platinum|gp|sp|cp|pp)(?:\s+pieces?)?)"


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    category: FactCategory
    change_type: ChangeType
    relationship: str
    template: str          # .format(speaker=, name=, item=)
    base_confidence: float
    dm_only: bool = False  # only trust when the DM/world says it


_RULES: tuple[_Rule, ...] = (
    # --- introductions & aliases -----------------------------------------
    _Rule(re.compile(r"\b(?i:my name is|i am called|they call me)\s+" + _NAME),
          FactCategory.NPC, ChangeType.INTRODUCED, "introduced_as",
          '{name} introduced themselves by name', 0.85),
    _Rule(re.compile(r"\b(?i:this is|you meet|allow me to introduce)\s+" + _NAME),
          FactCategory.NPC, ChangeType.INTRODUCED, "introduced",
          '{name} was introduced', 0.75, dm_only=True),
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:is also known as|goes by|used to be called|is revealed to be)\s+(?P<alias>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,3})"),
          FactCategory.ALIAS, ChangeType.DISCOVERED, "alias_of",
          '{name} is also known as {alias}', 0.85),
    # --- travel / locations ----------------------------------------------
    _Rule(re.compile(r"\b(?i:you (?:arrive|arrived) (?:at|in)|welcome to)\s+(?:(?i:the\s+\w+\s+of)\s+)?" + _NAME),
          FactCategory.LOCATION, ChangeType.DISCOVERED, "arrived_at",
          'The party arrived at {name}', 0.85, dm_only=True),
    _Rule(re.compile(r"\b(?i:the (?:city|town|village|keep|castle|tavern|inn|temple|forest|mountain|mine|mines|island) of)\s+" + _NAME),
          FactCategory.LOCATION, ChangeType.MENTIONED, "location_named",
          'A location called {name} was named', 0.7),
    _Rule(re.compile(r"\b(?i:we (?:should |will |'ll |gonna |going to |might |could )?(?:travel|traveled|travelled|head|headed|journey) (?:to|toward|towards))\s+" + _NAME),
          FactCategory.LOCATION, ChangeType.UPDATED, "travelled_to",
          'The party travelled to {name}', 0.7),
    # --- loot ---------------------------------------------------------------
    _Rule(re.compile(r"\b(?i:you (?:find|found|receive|received|are given|take|took|loot|pick up))\s+" + _ITEM),
          FactCategory.LOOT, ChangeType.DISCOVERED, "acquired",
          'The party acquired {item}', 0.8, dm_only=True),
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:hands? (?:you|over|him|her|them)|gives? (?:you|him|her|them))\s+" + _ITEM),
          FactCategory.LOOT, ChangeType.TRANSFERRED, "gave",
          '{name} gave the party {item}', 0.8, dm_only=True),
    _Rule(re.compile(r"\b(?i:the (?:chest|crate|box|body|room) contains?)\s+" + _ITEM),
          FactCategory.LOOT, ChangeType.DISCOVERED, "contains",
          'Found {item}', 0.75, dm_only=True),
    _Rule(re.compile(r"\b(?i:your reward is|the treasure includes)\s+" + _ITEM),
          FactCategory.LOOT, ChangeType.AWARDED, "rewarded",
          'The party was rewarded with {item}', 0.8, dm_only=True),
    # --- deaths / status -----------------------------------------------------
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:dies|is dead|drops? dead|is slain|breathes? (?:his|her|their) last)\b"),
          FactCategory.DEATH_OR_STATUS, ChangeType.UPDATED, "died",
          '{name} died', 0.8, dm_only=True),
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:is (?:revived|resurrected|brought back to life))\b"),
          FactCategory.DEATH_OR_STATUS, ChangeType.UPDATED, "revived",
          '{name} was revived', 0.8, dm_only=True),
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:is (?:knocked )?unconscious|falls? unconscious|is knocked out)\b"),
          FactCategory.CONDITION, ChangeType.UPDATED, "unconscious",
          '{name} fell unconscious', 0.75, dm_only=True),
    # --- combat --------------------------------------------------------------
    _Rule(re.compile(r"\b(?i:i cast|casts?)\s+(?P<item_name>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,3})"),
          FactCategory.COMBAT, ChangeType.CONFIRMED, "cast_spell",
          '{speaker} cast {item}', 0.7),
    _Rule(re.compile(r"\b" + _NAME + r"\s+(?i:takes?|took)\s+(?P<amount>\d+)\s+(?P<dtype>\w+\s+)?damage\b"),
          FactCategory.COMBAT, ChangeType.UPDATED, "took_damage",
          '{name} took {amount} damage', 0.75, dm_only=True),
    # --- quests --------------------------------------------------------------
    _Rule(re.compile(r"\b(?i:will you|we need you to|i need you to)\s+(?P<objective>[a-z][^.?!]{10,80})\?"),
          FactCategory.QUEST, ChangeType.PROPOSED, "quest_offered",
          'A task was proposed: {objective}', 0.6, dm_only=True),
    _Rule(re.compile(r"\b(?i:quest|task|job|contract|mission|bounty)\b.{0,40}\b(?i:complete|completed|done|finished)\b"),
          FactCategory.QUEST, ChangeType.COMPLETED, "quest_completed",
          'A quest was described as completed', 0.6),
    # --- rest ---------------------------------------------------------------
    _Rule(re.compile(r"\b(?i:long rest)\b"),
          FactCategory.STORY_EVENT, ChangeType.CONFIRMED, "long_rest",
          'The party took (or planned) a long rest', 0.6),
)


def _clean_name(raw: str | None) -> str:
    if not raw:
        return ""
    words = raw.strip().strip(".,;:!?\"'").split()
    while words and words[0].casefold() in COMMON_WORDS:
        words.pop(0)
    while words and words[-1].casefold() in COMMON_WORDS | {"of", "the"}:
        words.pop()
    name = " ".join(words)
    folded = name.casefold()
    if not name or folded in COMMON_WORDS or folded in GAME_MECHANIC_WORDS:
        return ""
    return name


def _known_name_normaliser(known_entities: frozenset[str]):
    """Build a function that restores canonical casing of known entity names.

    Whisper-style transcripts often lowercase proper names ("welcome to
    silverspire"); patterns anchored on capitals would miss them. Known
    campaign names are safe to re-case because a human already confirmed
    them — this is exactly how memory makes extraction better every session.
    """
    if not known_entities:
        return lambda text: text
    replacements = sorted(known_entities, key=len, reverse=True)
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(name) for name in replacements) + r")\b",
        re.IGNORECASE,
    )
    canonical = {name.casefold(): name.title() for name in replacements}

    def normalise(text: str) -> str:
        return pattern.sub(lambda m: canonical[m.group(1).casefold()], text)

    return normalise


def extract_native_facts(
    entries: list[TranscriptEntry],
    *,
    campaign_id: str,
    session_id: str,
    source_path: str,
    source_hash: str,
    known_entities: frozenset[str] = frozenset(),
) -> list[Fact]:
    """Extract structured facts from entries using deterministic patterns."""
    facts: list[Fact] = []
    seen: set[tuple[str, int]] = set()
    normalise = _known_name_normaliser(known_entities)

    for entry in entries:
        assessment = assess_entry(entry)
        if assessment.status is EpistemicStatus.TABLE_TALK:
            continue  # jokes / OOC / rules talk never produce facts
        speaker_mode = assess_speaker_mode(entry)
        time_status = assess_time_status(entry.text)
        # Patterns run on known-name-recased text; evidence quotes stay
        # verbatim from the original line.
        search_text = normalise(entry.text)

        for rule in _RULES:
            match = rule.pattern.search(search_text)
            if match is None:
                continue
            if rule.dm_only and speaker_mode not in WORLD_FACT_MODES:
                continue
            groups = match.groupdict()
            name = _clean_name(groups.get("name"))
            raw_item = (groups.get("item_name") or "").strip()
            item = raw_item if raw_item and raw_item[0].isdigit() else _clean_name(raw_item)
            alias = _clean_name(groups.get("alias"))
            if "name" in groups and not name:
                continue
            if "item_name" in groups and not item:
                continue

            statement = rule.template.format(
                speaker=entry.speaker, name=name or "", item=item or "",
                alias=alias or "", amount=groups.get("amount") or "",
                objective=(groups.get("objective") or "").strip(),
            ).strip()
            key = (rule.relationship + ":" + statement.casefold(), entry.line_number)
            if key in seen:
                continue
            seen.add(key)

            entities = [e for e in (name, item, alias) if e]
            confidence = rule.base_confidence
            if speaker_mode in WORLD_FACT_MODES:
                confidence = min(confidence + 0.1, 0.95)
            for entity in entities:
                if entity.casefold() in known_entities:
                    confidence = min(confidence + 0.05, 0.97)
                    break

            status = assessment.status
            effective_time = time_status
            if effective_time in (TimeStatus.PLANNED, TimeStatus.HYPOTHETICAL,
                                  TimeStatus.NEGATED):
                if status.rank < EpistemicStatus.UNCONFIRMED_THEORY.rank:
                    status = EpistemicStatus.UNCONFIRMED_THEORY
            elif effective_time is TimeStatus.UNKNOWN:
                effective_time = TimeStatus.HAPPENED if rule.change_type is not ChangeType.PROPOSED else TimeStatus.PLANNED
            if speaker_mode is SpeakerMode.NPC_DIALOGUE and status.rank < EpistemicStatus.CHARACTER_BELIEF.rank:
                status = EpistemicStatus.CHARACTER_BELIEF

            facts.append(
                Fact(
                    statement=statement,
                    category=rule.category,
                    change_type=rule.change_type,
                    entities=entities,
                    status=status,
                    confidence=round(confidence, 3),
                    subject=name or entry.speaker,
                    relationship=rule.relationship,
                    object_=item or alias or "",
                    time_status=effective_time,
                    speaker_mode=speaker_mode,
                    provenance=Provenance(
                        campaign_id=campaign_id,
                        session_id=session_id,
                        source_path=source_path,
                        source_hash=source_hash,
                        line_start=entry.line_number,
                        line_end=entry.line_number,
                        speaker=entry.speaker,
                        quote=entry.text[:300],
                        extractor=NATIVE_EXTRACTOR_VERSION,
                        extractor_version="1",
                    ),
                )
            )

    # Initiative order is a session-level combat fact when actually spoken.
    order = initiative_order(entries)
    if order:
        facts.append(
            Fact(
                statement="Initiative order (as spoken): "
                + ", ".join(f"{n} ({v})" for n, v in order),
                category=FactCategory.COMBAT,
                change_type=ChangeType.CONFIRMED,
                entities=[n for n, _ in order],
                status=EpistemicStatus.STRONGLY_SUPPORTED,
                confidence=0.8,
                subject="party",
                relationship="initiative_order",
                object_="",
                time_status=TimeStatus.HAPPENED,
                speaker_mode=SpeakerMode.MECHANICAL_RESULT,
                provenance=Provenance(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    source_path=source_path,
                    source_hash=source_hash,
                    line_start=entries[0].line_number,
                    line_end=entries[-1].line_number,
                    quote="initiative values spoken during the session",
                    extractor=NATIVE_EXTRACTOR_VERSION,
                    extractor_version="1",
                ),
            )
        )
    return facts
