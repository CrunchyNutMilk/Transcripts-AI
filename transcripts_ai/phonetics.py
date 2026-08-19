"""Phonetic and edit-distance primitives for name resolution.

Two deterministic signals:

- ``phonetic_key`` — a Metaphone-style key tuned for speech-to-text errors in
  fantasy names (Whisper confuses voiced/unvoiced pairs, drops leading
  consonants, and mangles vowels). It is deliberately a *simplified* metaphone:
  the goal is grouping plausible mis-hearings, not linguistic completeness,
  and it is only ever used as a suggestion signal — never to auto-apply.
- ``damerau_levenshtein`` — true edit distance with transpositions.
"""
from __future__ import annotations

import re
import unicodedata

_VOWELS = set("aeiouy")

# Sounds Whisper regularly confuses are collapsed to one symbol.
_EQUIV = {
    "b": "p", "p": "p",
    "d": "t", "t": "t",
    "g": "k", "k": "k", "q": "k", "c": "k",
    "v": "f", "f": "f",
    "z": "s", "s": "s", "x": "s",
    "j": "j",
    "m": "m", "n": "n",
    "l": "l", "r": "r",
    "w": "w", "h": "",
}


def _strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def phonetic_key(name: str) -> str:
    """Speech-tolerant phonetic key. Same key => plausibly the same spoken name."""
    text = _strip_accents(name).casefold()
    text = re.sub(r"[^a-z]", "", text)
    if not text:
        return ""
    # Common digraph normalisations before per-letter mapping.
    for src, dst in (
        ("ph", "f"), ("gh", "k"), ("ck", "k"), ("sch", "sk"), ("sh", "j"),
        ("ch", "j"), ("th", "t"), ("wr", "r"), ("kn", "n"), ("gn", "n"),
    ):
        text = text.replace(src, dst)
    out: list[str] = []
    for i, ch in enumerate(text):
        if ch in _VOWELS:
            # Keep only a leading vowel: interior vowels are what STT mangles most.
            if i == 0:
                out.append("a")
            continue
        mapped = _EQUIV.get(ch, ch)
        if mapped and (not out or out[-1] != mapped):
            out.append(mapped)
    return "".join(out)


def damerau_levenshtein(a: str, b: str, *, cap: int | None = None) -> int:
    """Optimal-string-alignment distance (adjacent transpositions count 1)."""
    a, b = a.casefold(), b.casefold()
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    previous2: list[int] = []
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current[j] = min(
                previous[j] + 1,        # deletion
                current[j - 1] + 1,     # insertion
                previous[j - 1] + cost, # substitution
            )
            if i > 1 and j > 1 and ca == b[j - 2] and a[i - 2] == cb:
                current[j] = min(current[j], previous2[j - 2] + cost)
        previous2, previous = previous, current
        if cap is not None and min(previous) > cap:
            return cap + 1
    return previous[len(b)]


def similarity_ratio(a: str, b: str) -> float:
    """1.0 identical → 0.0 unrelated, from Damerau-Levenshtein."""
    if not a and not b:
        return 1.0
    distance = damerau_levenshtein(a, b)
    longest = max(len(a), len(b))
    return 1.0 - distance / longest if longest else 1.0
