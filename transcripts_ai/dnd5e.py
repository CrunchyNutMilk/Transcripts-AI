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
    """Official name -> category ('spell' | 'monster' | 'item' | 'deity')."""
    names: dict[str, str] = {}
    raw = (resources.files("transcripts_ai.data") / "dnd5e_names.txt").read_text(
        encoding="utf-8")
    for line in raw.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        category, _, name = line.partition("\t")
        if name.strip():
            names[name.strip()] = category.strip()
    return names


@lru_cache(maxsize=1)
def _index() -> tuple[dict[tuple[str, ...], str], dict[str, list[str]]]:
    """(exact n-gram -> official name, anchor token -> official names)."""
    exact: dict[tuple[str, ...], str] = {}
    anchors: dict[str, list[str]] = {}
    for name in official_names():
        toks = tuple(_tokens(name))
        if not toks:
            continue
        exact[toks] = name
        for token in set(toks):
            if len(token) >= 4 and zipf(token) < _COMMON_ANCHOR_ZIPF:
                anchors.setdefault(token, []).append(name)
    return exact, anchors


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


def is_registrable(name: str) -> bool:
    """Safe to auto-register as an entity from a bare mention?

    Multi-word official names ("Horn of Blasting") cannot be accidental.
    A single-token name only qualifies when the word is rare English
    ("Aboleth" yes; the spells "Command", "Light", "Fly" would register
    on everyday speech and are left to evidence-based extraction).
    """
    toks = _tokens(name)
    if len(toks) >= 2:
        return True
    return bool(toks) and zipf(toks[0]) < _RARE_SINGLE_ZIPF


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
        if heard != truth and ratio < _TOKEN_RATIO:
            # Character-distant but phonetically identical ("steph"/"staff")
            # is exactly the mangle Whisper makes; sound rescues it.
            if phonetic_key(heard) != phonetic_key(truth):
                return 0.0
            ratio = max(ratio, _TOKEN_RATIO)
        total += ratio
    return total / len(target)


def _exact_in_tokens(toks: list[str]) -> tuple[list[str], set[int]]:
    """Official names exactly present in a token list, longest-match-first.
    Returns (names in order found, token positions they occupy)."""
    exact, _ = _index()
    found: list[str] = []
    occupied: set[int] = set()
    for size in range(min(6, len(toks)), 0, -1):
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
    _, anchors = _index()
    exact, _ = _index()
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
    exact, anchors = _index()
    parsed = parse_transcript(text, source_path="<names-check>")
    mentions: Counter = Counter()
    suspects: list[NearMiss] = []
    seen_windows: set[tuple[int, tuple[str, ...]]] = set()

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
                    if len(window) < size or (entry.line_number, window) in seen_windows:
                        continue
                    if window == target:
                        continue        # exact already counted (or subset)
                    score = _window_score(window, target)
                    if score >= _WINDOW_RATIO:
                        seen_windows.add((entry.line_number, window))
                        suspects.append(NearMiss(
                            heard=" ".join(window), official=name,
                            category=official_names()[name],
                            line=entry.line_number, score=round(score, 3),
                        ))
    suspects.sort(key=lambda s: (-s.score, s.line))
    return mentions, suspects
