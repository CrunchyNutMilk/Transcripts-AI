"""D&D gameplay pattern recognition and epistemic cue classification.

Deterministic detectors for the moments that structure a session (initiative,
combat, rests, deaths, loot, quests, travel) and for *how much to trust* a
line (DM statement vs player guess vs joke vs out-of-character chatter).

These signals feed the extractor's context and the confidence scorer; none of
them writes anything on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .schemas import EpistemicStatus, SpeakerMode, TimeStatus
from .transcript import TranscriptEntry


class GameEvent(str, Enum):
    INITIATIVE = "initiative"
    COMBAT_START = "combat_start"
    COMBAT_END = "combat_end"
    ATTACK_ROLL = "attack_roll"
    SAVING_THROW = "saving_throw"
    ABILITY_CHECK = "ability_check"
    DAMAGE = "damage"
    HEALING = "healing"
    CONDITION = "condition"
    SPELL_CAST = "spell_cast"
    MULTIATTACK = "multiattack"
    LEGENDARY_ACTION = "legendary_action"
    LAIR_ACTION = "lair_action"
    SHORT_REST = "short_rest"
    LONG_REST = "long_rest"
    LOOT = "loot"
    QUEST_UPDATE = "quest_update"
    TRAVEL = "travel"
    NPC_INTRODUCTION = "npc_introduction"
    ALIAS_REVEAL = "alias_reveal"
    DEATH = "death"
    REVIVAL = "revival"
    DEATH_SAVE = "death_save"


_EVENT_PATTERNS: tuple[tuple[GameEvent, re.Pattern[str]], ...] = (
    (GameEvent.INITIATIVE, re.compile(r"\broll(?:ing)? (?:for )?initiative\b|\binitiative order\b", re.I)),
    (GameEvent.COMBAT_START, re.compile(r"\bcombat (?:begins|starts)\b|\broll initiative\b|\bsurprise round\b", re.I)),
    (GameEvent.COMBAT_END, re.compile(r"\bcombat (?:ends|is over)\b|\bout of (?:combat|initiative)\b|\blast enemy (?:falls|dies|drops)\b", re.I)),
    (GameEvent.ATTACK_ROLL, re.compile(r"\b(?:attack roll|to hit|does an? \d+ hit|rolled? an? (?:nat(?:ural)? )?\d+ to hit)\b", re.I)),
    (GameEvent.SAVING_THROW, re.compile(r"\b(?:make|roll(?:s|ed)?) an? \w+ sav(?:e|ing throw)\b|\bsaving throw\b", re.I)),
    (GameEvent.ABILITY_CHECK, re.compile(r"\b(?:make|roll(?:s|ed)?) an? \w+ check\b|\b(?:perception|investigation|stealth|insight|arcana|athletics|acrobatics|persuasion|deception|intimidation|survival|history|nature|religion|medicine) check\b", re.I)),
    (GameEvent.DAMAGE, re.compile(r"\btakes? \d+ (?:points? of )?\w* ?damage\b|\bdeal(?:s|t)? \d+\b", re.I)),
    (GameEvent.HEALING, re.compile(r"\b(?:heal(?:s|ed)? (?:for )?\d+|regains? \d+ hit points?)\b", re.I)),
    (GameEvent.CONDITION, re.compile(r"\b(?:is|are|becomes?) (?:now )?(?:poisoned|stunned|paralyzed|paralysed|frightened|charmed|blinded|deafened|restrained|grappled|prone|unconscious|petrified|exhausted|invisible)\b", re.I)),
    (GameEvent.SPELL_CAST, re.compile(r"\bcasts?\b|\bat (?:\d+(?:st|nd|rd|th) )?level\b.*\bspell\b|\bupcast\b|\bconcentration\b", re.I)),
    (GameEvent.MULTIATTACK, re.compile(r"\bmulti-?attack\b", re.I)),
    (GameEvent.LEGENDARY_ACTION, re.compile(r"\blegendary (?:action|resistance)\b", re.I)),
    (GameEvent.LAIR_ACTION, re.compile(r"\blair action\b", re.I)),
    (GameEvent.SHORT_REST, re.compile(r"\bshort rest\b", re.I)),
    (GameEvent.LONG_REST, re.compile(r"\blong rest\b", re.I)),
    (GameEvent.LOOT, re.compile(r"\b(?:loot|treasure|you find|gold pieces?|\d+ ?gp|\bsilver pieces?\b|\d+ ?sp|picks? up|hands? (?:you|over)|receives?)\b", re.I)),
    (GameEvent.QUEST_UPDATE, re.compile(r"\bquest\b|\b(?:task|job|contract|mission) (?:is )?(?:complete|done|finished|failed|abandoned)\b|\bnew (?:task|job|mission)\b", re.I)),
    (GameEvent.TRAVEL, re.compile(r"\b(?:travel|journey|arrive|depart|set(?:s|ting)? (?:off|out)|head(?:s|ing)? (?:to|toward|north|south|east|west))\b", re.I)),
    (GameEvent.NPC_INTRODUCTION, re.compile(r"\b(?:my name is|introduces? (?:himself|herself|themselves)|this is|you meet)\b", re.I)),
    (GameEvent.ALIAS_REVEAL, re.compile(r"\b(?:also known as|real name|true name|goes by|used to be called|revealed? to be)\b", re.I)),
    (GameEvent.DEATH, re.compile(r"\b(?:dies|is dead|drops? dead|killed|slain|breathes? (?:his|her|their) last)\b", re.I)),
    (GameEvent.REVIVAL, re.compile(r"\b(?:revivify|revived?|resurrect(?:s|ed|ion)?|raise dead|back to life|brought back)\b", re.I)),
    (GameEvent.DEATH_SAVE, re.compile(r"\bdeath sav(?:e|ing throw)s?\b", re.I)),
)

INITIATIVE_COUNT = re.compile(r"\b([A-Z][\w'\-]*|[a-z][\w'\-]*)\s+(?:got|has|is at|with|rolled)\s+(?:an?\s+)?(\d{1,2})\b")


@dataclass
class DetectedEvent:
    event: GameEvent
    line_number: int
    speaker: str
    excerpt: str


def detect_events(entries: list[TranscriptEntry]) -> list[DetectedEvent]:
    events: list[DetectedEvent] = []
    for entry in entries:
        for event, pattern in _EVENT_PATTERNS:
            if pattern.search(entry.text):
                events.append(
                    DetectedEvent(
                        event=event,
                        line_number=entry.line_number,
                        speaker=entry.speaker,
                        excerpt=entry.text[:160],
                    )
                )
    return events


def initiative_evidence(
    entries: list[TranscriptEntry],
) -> list[tuple[str, int, TranscriptEntry]]:
    """Initiative values actually spoken, each with its source entry.

    Only returns names spoken with a number; empty when the table never
    states the order out loud (the summary must then say "not stated").
    """
    found: dict[str, tuple[int, TranscriptEntry]] = {}
    window_active = 0
    for entry in entries:
        if _EVENT_PATTERNS[0][1].search(entry.text):
            window_active = 12  # look at the next dozen entries
            continue
        if window_active <= 0:
            continue
        window_active -= 1
        for name, value in INITIATIVE_COUNT.findall(entry.text):
            folded = name.strip().casefold()
            if folded in {"i", "he", "she", "it", "that", "who"}:
                continue
            found.setdefault(name.strip(), (int(value), entry))
    return sorted(
        ((name, value, entry) for name, (value, entry) in found.items()),
        key=lambda item: -item[1],
    )


def initiative_order(entries: list[TranscriptEntry]) -> list[tuple[str, int]]:
    return [(name, value) for name, value, _entry in initiative_evidence(entries)]


# ---------------------------------------------------------------------------
# Epistemic cues — how much should a statement on this line be trusted?
# ---------------------------------------------------------------------------

_JOKE_CUES = re.compile(
    r"\b(?:lol|lmao|haha+|hehe+|jk|just kidding|kidding|joking|imagine if|"
    r"what if we just|that would be funny|meme)\b", re.I,
)
_OOC_CUES = re.compile(
    r"\b(?:out of character|ooc|as a player|irl|in real life|pizza|snack|"
    r"bathroom|toilet|next week|schedule|can'?t make it|my (?:cat|dog|kid)|"
    r"volume|mic|discord|roll20|stream|mute)\b", re.I,
)
_SPECULATION_CUES = re.compile(
    r"\b(?:i (?:think|bet|guess|reckon|assume|wonder)|maybe|probably|"
    r"what if|could be|might be|my theory|i suspect|presumably|"
    r"if i had to guess)\b", re.I,
)
_PLANNING_CUES = re.compile(
    r"\b(?:we should|let'?s|we could|the plan is|planning to|we need to|"
    r"tomorrow we|next session|before we go|our next move)\b", re.I,
)
_BELIEF_CUES = re.compile(
    r"\b(?:my character (?:thinks|believes)|in character|"
    r"\w+ (?:believes|is convinced|assumes))\b", re.I,
)
_DM_HINT_CUES = re.compile(
    r"\b(?:you (?:notice|sense|feel|get the feeling)|something (?:seems|feels) "
    r"(?:off|wrong|strange)|for (?:some|reasons) (?:reason|unknown)|"
    r"you can'?t quite|oddly|strangely|suspiciously)\b", re.I,
)
_RULES_TALK_CUES = re.compile(
    r"\b(?:rules?[- ]?(?:wise|lawyer)|raw\b|rules as written|errata|sage advice|"
    r"page \d+|phb|dmg|monster manual|how does .{0,40}work|does that stack)\b", re.I,
)


@dataclass
class EpistemicAssessment:
    status: EpistemicStatus
    cues: list[str]
    in_character_likely: bool


def assess_entry(entry: TranscriptEntry) -> EpistemicAssessment:
    """Classify one transcript entry's epistemic weight.

    Priority: OOC/joke marks table talk regardless of speaker; otherwise the
    DM's plain statements are strong world facts, DM hedges are hints, player
    speculation/planning is downgraded accordingly.
    """
    text = entry.text
    cues: list[str] = []
    if _JOKE_CUES.search(text):
        cues.append("joke")
    if _OOC_CUES.search(text):
        cues.append("out_of_character")
    if _RULES_TALK_CUES.search(text):
        cues.append("rules_discussion")
    if cues:
        return EpistemicAssessment(EpistemicStatus.TABLE_TALK, cues, False)

    if _PLANNING_CUES.search(text):
        cues.append("planning")
    if _SPECULATION_CUES.search(text):
        cues.append("speculation")
    if _BELIEF_CUES.search(text):
        cues.append("character_belief")

    if entry.is_dm:
        if _DM_HINT_CUES.search(text):
            cues.append("dm_hedge")
            return EpistemicAssessment(EpistemicStatus.DM_HINT, cues, True)
        if "speculation" in cues:
            return EpistemicAssessment(EpistemicStatus.DM_HINT, cues, True)
        return EpistemicAssessment(EpistemicStatus.STRONGLY_SUPPORTED, cues or ["dm_statement"], True)

    if "character_belief" in cues:
        return EpistemicAssessment(EpistemicStatus.CHARACTER_BELIEF, cues, True)
    if "speculation" in cues:
        return EpistemicAssessment(EpistemicStatus.PLAYER_ASSUMPTION, cues, True)
    if "planning" in cues:
        return EpistemicAssessment(EpistemicStatus.UNCONFIRMED_THEORY, cues, True)
    # A player's plain declarative line: usually describing their own action.
    return EpistemicAssessment(EpistemicStatus.STRONGLY_SUPPORTED, cues or ["player_statement"], True)


# ---------------------------------------------------------------------------
# Time status — did it happen, or was it only planned / negated / hypothetical?
# ---------------------------------------------------------------------------

_NEGATION_CUES = re.compile(
    r"\b(?:didn'?t|did not|don'?t|never|was going to|were going to|almost|"
    r"changed (?:my|his|her|their) mind|decided? not to|instead of|"
    r"would have|could have(?! to)|wouldn'?t have|nevermind|never mind)\b", re.I,
)
_PLAN_CUES = re.compile(
    r"\b(?:we should|let'?s|we could|we will|we'?ll|going to|gonna|plan(?:ning)? to|"
    r"tomorrow|next time|later we|before we go|intend to|i might|maybe we)\b", re.I,
)
_HYPOTHETICAL_CUES = re.compile(
    r"\b(?:what if|imagine|hypothetically|suppose|in theory|if we had|"
    r"would it work if|can (?:i|we|it)\b.*\?)", re.I,
)
_PAST_PRESENT_ACTION = re.compile(
    r"\b(?:i (?:cast|attack|move|open|take|grab|drink|use|swing|shoot|stab)|"
    r"(?:you|we) (?:enter|arrive|travel|find|receive|take|attack|see|hear|reach)|"
    r"(?:hands?|gives?|gave|handed|took|entered|arrived|found|received|killed|"
    r"defeated|died|opened))\b", re.I,
)


def assess_time_status(text: str) -> TimeStatus:
    """Deterministic time-status floor for one statement/evidence line.

    Negation beats planning beats hypothetical beats plain action; a line
    with none of those cues is UNKNOWN and the extractor's judgement stands.
    """
    if _NEGATION_CUES.search(text):
        return TimeStatus.NEGATED
    if _HYPOTHETICAL_CUES.search(text):
        return TimeStatus.HYPOTHETICAL
    if _PLAN_CUES.search(text):
        return TimeStatus.PLANNED
    if _PAST_PRESENT_ACTION.search(text):
        return TimeStatus.HAPPENED
    return TimeStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Speaker mode — the authority hierarchy for who is really talking
# ---------------------------------------------------------------------------

_QUOTED_SPEECH = re.compile(r"[\"“].+?[\"”]|\bsays?,|\bsaid,|\breplies?,|\basks?,")
_MECHANICAL = re.compile(
    r"\b(?:rolled? (?:a |an )?\d+|nat(?:ural)? (?:1|20)|takes? \d+ .*damage|"
    r"that'?s a (?:hit|miss)|save (?:succeeds|fails)|dc \d+|\d+ to hit)\b", re.I,
)


def assess_speaker_mode(entry: TranscriptEntry, *, is_dm: bool | None = None) -> SpeakerMode:
    """Classify who is actually speaking on this line.

    The critical distinction: the DM narrating the world (can confirm facts)
    versus the DM voicing an NPC (confirms only that the NPC *claims* it).
    Quoted speech or dialogue verbs inside a DM line mark NPC dialogue.
    """
    dm = entry.is_dm if is_dm is None else is_dm
    text = entry.text
    if _JOKE_CUES.search(text) or _OOC_CUES.search(text):
        return SpeakerMode.TABLE_TALK
    if _MECHANICAL.search(text):
        return SpeakerMode.MECHANICAL_RESULT
    if dm:
        if _QUOTED_SPEECH.search(text):
            return SpeakerMode.NPC_DIALOGUE
        return SpeakerMode.DM_NARRATION
    if _QUOTED_SPEECH.search(text):
        return SpeakerMode.PC_DIALOGUE
    return SpeakerMode.PLAYER_STATEMENT


# Speaker modes that may confirm objective world facts on their own.
WORLD_FACT_MODES = frozenset({SpeakerMode.DM_NARRATION, SpeakerMode.MECHANICAL_RESULT})
