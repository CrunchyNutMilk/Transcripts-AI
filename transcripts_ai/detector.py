"""Rule-based entity detection in transcript entries.

Finds candidate names *before* any AI involvement, with strong suppression so
ordinary words, filler speech and game mechanics are never proposed as
entities. AI later classifies/verifies candidates; this layer decides what is
worth looking at, and why — every candidate carries its trigger reasons.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .transcript import TranscriptEntry

# Words that are capitalised for grammatical reasons or too generic to be names.
COMMON_WORDS = frozenset(
    """
    the a an i you he she it we they this that these those there here
    ah oh um uh hmm yeah yes no okay ok right well so anyway like just
    what who where when why how which whose and but or nor for yet because
    cause not now all thank thanks sorry get got does did do can could
    will would should shall may might must unless really very alright
    also then than too maybe please gonna wanna sure fine cool nice good
    great god jesus christ damn hell fuck fucking shit crap wow whoa hey
    if is are was were been being have has had let go come came went
    look looks looking wait stop
    i'm i've i'll i'd you're you've we're we've they're don't can't won't
    it's that's there's let's he's she's what's who's didn't doesn't isn't
    monday tuesday wednesday thursday friday saturday sunday
    january february march april may june july august september october
    november december
    north south east west
    dm gm dungeon master game
    """.split()
)

GAME_MECHANIC_WORDS = frozenset(
    """
    initiative attack damage roll rolls rolled check save saves saving
    advantage disadvantage crit critical nat natural d4 d6 d8 d10 d12 d20 d100
    hp hit points armor armour class ac dc str dex con int wis cha
    turn round action bonus reaction movement spell slot slots level
    perception investigation stealth athletics acrobatics arcana history
    insight intimidation medicine nature performance persuasion religion
    sleight survival deception animal handling
    """.split()
)

_NAME = r"(?P<name>[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,3})"
INTRODUCTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?i:my name is|i am called|they call me|call me)\s+" + _NAME), "self_introduction"),
    (re.compile(r"\b(?i:this is|meet|named|called)\s+" + _NAME), "introduction"),
    (re.compile(r"\b(?i:the (?:city|town|village|keep|castle|tavern|inn|temple|forest|mountain|river|island) of)\s+" + _NAME), "location_of_phrase"),
    (re.compile(r"\b(?i:welcome to|arrive(?:s|d)? (?:at|in)|travel(?:s|led|ing)? to|head(?:s|ed|ing)? to|enter(?:s|ed)?)\s+" + _NAME), "travel_target"),
)

CAPITALISED_SEQUENCE = re.compile(r"\b[A-Z][\w'\-]+(?:\s+(?:of|the|[A-Z][\w'\-]+)){0,3}\b")
SENTENCE_START = re.compile(r"(?:^|[.!?]\s+|[\"']\s*)$")


@dataclass
class DetectedName:
    text: str
    entry_line: int
    speaker: str
    reasons: list[str] = field(default_factory=list)
    mentions: int = 1
    context: str = ""

    @property
    def folded(self) -> str:
        return " ".join(self.text.casefold().split())


def _is_suppressed(token_text: str) -> bool:
    folded = token_text.casefold().strip("'\"")
    if not folded:
        return True
    words = folded.split()
    if all(w in COMMON_WORDS for w in words):
        return True
    if all(w in GAME_MECHANIC_WORDS or w in COMMON_WORDS for w in words):
        return True
    if len(folded) <= 2:
        return True
    if any(ch.isdigit() for ch in folded):
        return True
    return False


def detect_names(
    entries: list[TranscriptEntry], *, known_names: frozenset[str] = frozenset()
) -> list[DetectedName]:
    """Find candidate entity names with reasons; heavily suppression-biased.

    known_names (case-folded) are always surfaced when mentioned, so existing
    campaign entities are tracked even in lowercase or mid-sentence positions.
    """
    found: dict[str, DetectedName] = {}

    def add(text: str, entry: TranscriptEntry, reason: str) -> None:
        text = text.strip().strip(".,;:!?\"'")
        if text.endswith(("'s", "’s")):
            text = text[:-2].rstrip()
        if not text:
            return
        key = " ".join(text.casefold().split())
        if key in found:
            found[key].mentions += 1
            if reason not in found[key].reasons:
                found[key].reasons.append(reason)
            return
        found[key] = DetectedName(
            text=text,
            entry_line=entry.line_number,
            speaker=entry.speaker,
            reasons=[reason],
            context=entry.text[:240],
        )

    for entry in entries:
        text = entry.text

        # Known campaign names, any casing, any position.
        lowered = f" {text.casefold()} "
        for known in known_names:
            if f" {known} " in lowered or lowered.strip().startswith(known):
                add(known, entry, "known_entity_mention")

        # Introduction/travel phrasing — the strongest new-name signals.
        for pattern, reason in INTRODUCTION_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group("name")
                if not _is_suppressed(candidate):
                    add(candidate, entry, reason)

        # Capitalised sequences not at sentence start.
        for match in CAPITALISED_SEQUENCE.finditer(text):
            candidate = match.group(0).strip()
            prefix = text[: match.start()]
            at_sentence_start = bool(SENTENCE_START.search(prefix)) or match.start() == 0
            if _is_suppressed(candidate):
                continue
            trimmed = _trim_stopword_edges(candidate)
            if not trimmed or _is_suppressed(trimmed):
                continue
            if at_sentence_start and " " not in trimmed:
                # A lone capitalised word opening a sentence is usually just
                # grammar; only multiword sequences count from that position.
                continue
            add(trimmed, entry, "capitalised_mid_sentence" if not at_sentence_start
                else "capitalised_multiword")

    results = list(found.values())
    # Single weak-signal, single-mention candidates are dropped: repetition or
    # a strong pattern is required, exactly like the bot's suppression policy.
    strong = {"self_introduction", "introduction", "location_of_phrase",
              "travel_target", "known_entity_mention"}
    return [
        d
        for d in results
        if d.mentions > 1 or strong.intersection(d.reasons)
        or "capitalised_mid_sentence" in d.reasons and len(d.text.split()) > 1
    ]


def _trim_stopword_edges(text: str) -> str:
    words = text.split()
    while words and words[0].casefold() in COMMON_WORDS:
        words.pop(0)
    while words and words[-1].casefold() in COMMON_WORDS | {"of", "the"}:
        words.pop()
    return " ".join(words)
