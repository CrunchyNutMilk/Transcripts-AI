"""Official D&D 5e names: recognise them, and catch Whisper mangling them.

``data/dnd5e_names.txt`` carries ~1,100 official names (SRD spells and
monsters, standard magic items, deities). Two uses:

- ``scan_text`` — sweep a transcript for official-name mentions (exact,
  any casing) and for *near-miss* windows that are probably a mangled
  official name ("steph of swarming insects", "corn of blasting"). The
  gold-transcript workflow uses this to fix rule terms with confidence.
- ``official_names`` — the raw name→category map, for prompts and review
  hints ("Insect Plague" → probably a spell, not an NPC).

Matching is anchored: a window is only considered when at least one of
its tokens is a *rare* token of some official name (rare = not common
English), so "of the" never triggers anything, while any window
containing "swarming" is checked against every name that contains it.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from importlib import resources

from .phonetics import phonetic_key
from .schemas import EntityKind
from .transcript import TranscriptEntry, parse_transcript
from .wordlist import zipf

_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")
_COMMON_ANCHOR_ZIPF = 4.3       # tokens at/above this are too common to anchor
_WINDOW_RATIO = 0.84            # whole-window similarity to call it a near-miss
_TOKEN_RATIO = 0.72             # per-token floor so one token can't carry it


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


@lru_cache(maxsize=1)
def official_names() -> dict[str, str]:
    """Official name -> category ('spell' | 'monster' | 'item' | 'deity').

    Defensive on load: placeholder rows ("Unknown …", "Generic …") are
    ignored, and a name listed under two categories keeps its FIRST one
    (the data file lists spells before deities, so "Bane" is the spell).
    """
    names: dict[str, str] = {}
    raw = (resources.files("transcripts_ai.data") / "dnd5e_names.txt").read_text(
        encoding="utf-8")
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        category, _, name = line.partition("\t")
        name = name.strip()
        if (not name or name.casefold().startswith(("unknown", "generic"))
                or name in names):
            continue
        names[name] = category.strip()
    return names


@lru_cache(maxsize=1)
def _index() -> tuple[dict[tuple[str, ...], str], dict[str, list[str]], int]:
    """(exact n-gram -> name, anchor token -> names, longest name tokens)."""
    exact: dict[tuple[str, ...], str] = {}
    anchors: dict[str, list[str]] = {}
    longest = 1
    for name in official_names():
        toks = tuple(_tokens(name))
        if not toks:
            continue
        exact[toks] = name
        longest = max(longest, len(toks))
        for token in set(toks):
            if len(token) >= 4 and zipf(token) < _COMMON_ANCHOR_ZIPF:
                anchors.setdefault(token, []).append(name)
    return exact, anchors, longest


_WEAPON_WORDS = frozenset(
    "sword dagger mace axe bow blade javelin trident hammer flail spear "
    "scimitar oathbow defender arrow".split())
_ARMOUR_WORDS = frozenset("armor armour mail plate shield chain".split())
_RARE_SINGLE_ZIPF = 3.3   # single-token names rarer than this are registrable


def kind_for(name: str) -> EntityKind:
    """The engine EntityKind an official name maps to."""
    category = official_names().get(name, "")
    if category == "spell":
        return EntityKind.SPELL
    if category == "monster":
        return EntityKind.CREATURE
    if category == "deity":
        return EntityKind.DEITY
    toks = set(_tokens(name))
    if "potion" in toks or "oil" in toks or "philter" in toks:
        return EntityKind.POTION
    if toks & _ARMOUR_WORDS:
        return EntityKind.ARMOUR
    if toks & _WEAPON_WORDS:
        return EntityKind.WEAPON
    return EntityKind.ITEM


# The SRD's generic NPC statblocks: everyday class/role words that a table
# says constantly about PEOPLE ("our druid", "the archmage nods"). Never
# auto-registered, no matter how rare general English says they are.
_SRD_NPC_STATBLOCKS = frozenset({
    "acolyte", "archmage", "assassin", "bandit", "bandit captain",
    "berserker", "commoner", "cult fanatic", "cultist", "druid",
    "gladiator", "guard", "knight", "mage", "noble", "priest", "scout",
    "spy", "thug", "tribal warrior", "veteran",
})


def _fantasy_token(token: str) -> bool:
    """Genuinely fantasy vocabulary: rare AND (very rare or not English)."""
    from .wordlist import is_dictionary_word

    if zipf(token) >= _RARE_SINGLE_ZIPF:
        return False
    return zipf(token) < 2.6 or not is_dictionary_word(token)


def is_lootable(name: str) -> bool:
    """Eligible for the official-item LOOT pass. Looser than registration:
    the acquisition cue next to the exact name IS the evidence, so any
    multi-word item qualifies ("Necklace of Prayer Beads"); single-token
    items still need fantasy vocabulary ("Oathbow" yes, "Defender" no)."""
    toks = _tokens(name)
    if not toks:
        return False
    return len(toks) >= 2 or _fantasy_token(toks[0])


def is_registrable(name: str) -> bool:
    """Safe to auto-register as an entity from a BARE mention?

    The adversarial review proved everyday phrases fire constantly ("a
    gust of wind snuffs the torches" is not the spell; "the traveler
    pushes open the door" is not the deity; "our druid" is a person), so
    the gate demands vocabulary that cannot occur in normal speech:

    - SRD NPC-statblock words never qualify, however rare ("Archmage").
    - every OTHER name — single or multi-word — needs at least one
      genuinely fantasy token: "Aboleth", "Tiamat", "Vorpal Sword",
      "Abi-Dalzims Horrid Wilting" pass; "Gust Of Wind", "The Traveler",
      "Horn of Blasting", "Black Bear" do not.

    Mundanely-worded ITEMS still become entities — through loot evidence
    (an actual award in the transcript), not through bare mention; see
    the pipeline's registration pass.
    """
    folded = " ".join(_tokens(name))
    if not folded or folded in _SRD_NPC_STATBLOCKS:
        return False
    return any(_fantasy_token(t) for t in _tokens(name))


@lru_cache(maxsize=1)
def protected_tokens() -> set[str]:
    """Rare tokens of official names — the spellchecker must never
    'correct' thunderous, aboleth, or tiamat into everyday words."""
    protected: set[str] = set()
    for name in official_names():
        for token in _tokens(name):
            if len(token) >= 4 and zipf(token) < 3.9:
                protected.add(token)
    return protected


@dataclass
class NearMiss:
    heard: str            # the transcript window, as written
    official: str         # the official name it resembles
    category: str
    line: int
    score: float


@dataclass
class OfficialMention:
    name: str             # official casing
    category: str
    line: int
    quote: str


def _window_score(window: tuple[str, ...], target: tuple[str, ...]) -> float:
    if len(window) != len(target):
        return 0.0
    total = 0.0
    for heard, truth in zip(window, target):
        ratio = SequenceMatcher(a=heard, b=truth).ratio()
        if heard != truth:
            # Phonetically identical tokens ("steph"/"staff",
            # "teamat"/"tiamat") are exactly the mangle Whisper makes;
            # sound lifts them above the window bar so even a
            # single-token name is reachable on phonetics alone.
            if phonetic_key(heard) == phonetic_key(truth):
                ratio = max(ratio, 0.86)
            elif ratio < _TOKEN_RATIO:
                return 0.0
        total += ratio
    return total / len(target)


def _exact_in_tokens(toks: list[str]) -> tuple[list[str], set[int]]:
    """Official names exactly present in a token list, longest-match-first.
    Returns (names in order found, token positions they occupy)."""
    exact, _, longest = _index()
    found: list[str] = []
    occupied: set[int] = set()
    for size in range(min(longest, len(toks)), 0, -1):
        for start in range(len(toks) - size + 1):
            if any(p in occupied for p in range(start, start + size)):
                continue
            name = exact.get(tuple(toks[start:start + size]))
            if name:
                found.append(name)
                occupied.update(range(start, start + size))
    return found, occupied


def mentions_in_entries(entries: list[TranscriptEntry]) -> list[OfficialMention]:
    """Exact official-name mentions with line provenance."""
    mentions = []
    for entry in entries:
        for name in _exact_in_tokens(_tokens(entry.text))[0]:
            mentions.append(OfficialMention(
                name=name, category=official_names()[name],
                line=entry.line_number, quote=entry.text[:240]))
    return mentions


def nearest_official(text: str) -> NearMiss | None:
    """Best official near-match for a short candidate string, if any."""
    toks = tuple(_tokens(text))
    if not toks:
        return None
    exact, anchors, _ = _index()
    if toks in exact:
        name = exact[toks]
        return NearMiss(heard=text, official=name,
                        category=official_names()[name], line=0, score=1.0)
    best: NearMiss | None = None
    candidates = {name for token in toks for name in anchors.get(token, [])}
    if len(toks) == 1:
        key = phonetic_key(toks[0])
        candidates |= {name for name in official_names()
                       if len(_tokens(name)) == 1
                       and phonetic_key(_tokens(name)[0]) == key}
    for name in candidates:
        score = _window_score(toks, tuple(_tokens(name)))
        if score >= _WINDOW_RATIO and (best is None or score > best.score):
            best = NearMiss(heard=text, official=name,
                            category=official_names()[name],
                            line=0, score=round(score, 3))
    return best


def scan_text(text: str) -> tuple[Counter, list[NearMiss]]:
    """(exact mention counts by official name, near-miss suspects).

    Works on a raw transcript: only speaker-line text is scanned, so a
    speaker named "Wolf" never counts as a monster mention.
    """
    exact, anchors, _ = _index()
    parsed = parse_transcript(text, source_path="<names-check>")
    mentions: Counter = Counter()
    # Best official name per (line, window), by score — the first name in
    # an anchor's list must never shadow a better-scoring one.
    best: dict[tuple[int, tuple[str, ...]], NearMiss] = {}

    for entry in parsed.entries:
        toks = _tokens(entry.text)
        found, occupied = _exact_in_tokens(toks)
        for name in found:
            mentions[name] += 1
        # Near-miss pass, anchored on rare tokens outside exact matches.
        for position, token in enumerate(toks):
            if position in occupied or token not in anchors:
                continue
            for name in anchors[token]:
                target = tuple(_tokens(name))
                size = len(target)
                for start in range(max(0, position - size + 1), position + 1):
                    window = tuple(toks[start:start + size])
                    if len(window) < size or window == target:
                        continue
                    score = _window_score(window, target)
                    if score < _WINDOW_RATIO:
                        continue
                    key = (entry.line_number, window)
                    current = best.get(key)
                    if current is None or score > current.score:
                        best[key] = NearMiss(
                            heard=" ".join(window), official=name,
                            category=official_names()[name],
                            line=entry.line_number, score=round(score, 3),
                        )
    suspects = sorted(best.values(), key=lambda s: (-s.score, s.line))
    return mentions, suspects
