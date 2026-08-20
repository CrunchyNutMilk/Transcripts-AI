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
from .transcript import parse_transcript
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


@dataclass
class NearMiss:
    heard: str            # the transcript window, as written
    official: str         # the official name it resembles
    category: str
    line: int
    score: float


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
        occupied: set[int] = set()     # token positions inside exact matches
        # Exact pass, longest names first so subsets don't double-count.
        for size in range(min(6, len(toks)), 0, -1):
            for start in range(len(toks) - size + 1):
                if any(p in occupied for p in range(start, start + size)):
                    continue
                window = tuple(toks[start:start + size])
                name = exact.get(window)
                if name:
                    mentions[name] += 1
                    occupied.update(range(start, start + size))
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
