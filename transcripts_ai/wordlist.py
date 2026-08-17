"""Bundled English word-frequency dictionary (offline, zero-dependency).

Loaded lazily from ``data/word_frequencies.txt`` (top ~25k English words with
Zipf frequencies, derived from the wordfreq project — data CC-BY-SA 4.0).
Zipf scale: ~7 = extremely common ("the"), ~3 = ordinary vocabulary,
below ~2.5 = rare. Used for the ordinary-word lane of spell-checking and for
keeping ordinary words out of the entity review queue.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).parent / "data" / "word_frequencies.txt"
_DND_PATH = Path(__file__).parent / "data" / "dnd_words.txt"
_MISSPELLINGS_PATH = Path(__file__).parent / "data" / "common_misspellings.txt"

# A word at or above this Zipf is "ordinary vocabulary" — never an entity
# candidate on its own, and an eligible target for auto-correction.
COMMON_ZIPF = 3.0


@lru_cache(maxsize=1)
def frequencies() -> dict[str, float]:
    table: dict[str, float] = {}
    with open(_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            word, _, zipf = line.rstrip("\n").partition("\t")
            table[word] = float(zipf)
    # Supplemental D&D / table-talk vocabulary: membership only (zipf 2.0),
    # so these are recognised as real words but stay below the target floor —
    # nothing ever gets "corrected" TO or FROM fantasy jargon automatically.
    with open(_DND_PATH, encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if word and not word.startswith("#"):
                table.setdefault(word, 2.0)
    # Web-frequency corpora contain famous misspellings ("becuase"); those
    # must not count as valid words.
    for wrong in known_misspellings():
        table.pop(wrong, None)
    return table


@lru_cache(maxsize=1)
def known_misspellings() -> dict[str, str]:
    """Curated original -> correction map for famous misspellings."""
    table: dict[str, str] = {}
    with open(_MISSPELLINGS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            wrong, _, right = line.rstrip("\n").partition("\t")
            if wrong and right and wrong != right:
                table[wrong] = right
    return table


def zipf(word: str) -> float:
    """Zipf frequency of a word; 0.0 if not in the bundled list."""
    return frequencies().get(word.casefold(), 0.0)


def is_dictionary_word(word: str) -> bool:
    return word.casefold() in frequencies()


def is_common_word(word: str, *, min_zipf: float = COMMON_ZIPF) -> bool:
    return zipf(word) >= min_zipf


_ALPHABET = "abcdefghijklmnopqrstuvwxyz'"


def edits1(word: str):
    """All strings at Damerau edit distance 1 (classic Norvig construction)."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = (a + b[1:] for a, b in splits if b)
    transposes = (a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1)
    replaces = (a + c + b[1:] for a, b in splits if b for c in _ALPHABET)
    inserts = (a + c + b for a, b in splits for c in _ALPHABET)
    return set(deletes) | set(transposes) | set(replaces) | set(inserts)


def best_correction(word: str, *, min_zipf: float = COMMON_ZIPF) -> tuple[str, float, float] | None:
    """Most likely standard word one edit away.

    Returns (correction, its_zipf, margin_over_runner_up) or None. The margin
    lets callers demand an unambiguous winner before auto-correcting.
    """
    table = frequencies()
    candidates = [
        (table[c], c) for c in edits1(word.casefold())
        if table.get(c, 0.0) >= min_zipf
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    best_zipf, best_word = candidates[0]
    margin = best_zipf - candidates[1][0] if len(candidates) > 1 else best_zipf
    return best_word, best_zipf, margin
