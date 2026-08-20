"""Two-lane spelling correction.

Lane 1 — ordinary words (this module): misspelled standard vocabulary
("becuase", "thier", "definately") is corrected automatically in the Mapped
transcript using the bundled dictionary, sentence-position rules and a
confidence score. Each applied correction is written to the audit log under
its own action (``auto_spell_correction``) — separate from entity decisions.
Ordinary words never become entities, never enter campaign memory, and never
appear in the human review queue.

Lane 2 — possible named entities (PCs, NPCs, locations) stays with the
resolver + review workflow: campaign search, phonetic/spelling/context
comparison, suggestions, and the six human actions. Anything that might be a
name is explicitly OUT of scope here:

- capitalised tokens are never touched (they are entity-lane material);
- tokens matching a campaign entity or alias (exactly or phonetically) are
  never touched, so "gomra" is left for the entity workflow even though it
  is not a dictionary word;
- short tokens (< 4 letters) are never touched — too risky.

The Unmapped transcript is never modified: `apply_corrections` refuses any
path that does not look like a Mapped transcript.
"""
from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from collections import Counter

from .memory import CampaignMemory
from .phonetics import damerau_levenshtein, phonetic_key
from .transcript import ParsedTranscript
from .wordlist import (
    best_correction,
    is_dictionary_word,
    known_misspellings,
    zipf,
)

SPELLCHECK_VERSION = "ordinary-word-spellcheck-v2"

MIN_TOKEN_LENGTH = 4
MIN_TARGET_ZIPF = 3.0        # corrections may only land on ordinary words
MIN_MARGIN = 0.30            # winner must beat the runner-up clearly
DEFAULT_MIN_CONFIDENCE = 0.85
# An unknown token repeated this often is a term or a name, not a typo —
# typos rarely repeat identically; names and jargon do.
REPEAT_AS_TERM = 3

_TOKEN = re.compile(r"\b[a-z][a-z']{1,29}\b")


def _is_derivative_of_dictionary_word(token: str) -> bool:
    """Playful coinages built on real words are not misspellings.

    "smacky", "branchy", "owly", "gloving", "booped" — table talk derives
    words constantly; stripping common suffixes (with e-restoration) against
    the dictionary catches these so they are left exactly as spoken.
    """
    if token.endswith("'s") and is_dictionary_word(token[:-2]):
        return True
    for suffix in ("ies", "ing", "ed", "er", "est", "y", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            stem = token[: -len(suffix)]
            if is_dictionary_word(stem) or is_dictionary_word(stem + "e"):
                return True
    return False


@dataclass(frozen=True)
class SpellCorrection:
    line_number: int
    original: str
    corrected: str
    confidence: float
    context: str


def _confidence(target_zipf: float, margin: float, length: int) -> float:
    """Confidence that a one-edit fix to a common word is right.

    Grounded in: how common the target is (rarer targets are riskier), how
    clearly it beats alternative candidates, and token length (longer
    misspellings are less likely to collide with other words).
    """
    score = 0.55
    score += min((target_zipf - MIN_TARGET_ZIPF) * 0.08, 0.24)
    score += min(margin * 0.25, 0.15)
    if length >= 6:
        score += 0.05
    return round(min(score, 0.99), 3)


class SpellChecker:
    def __init__(self, memory: CampaignMemory):
        self.memory = memory

    def _protected_tokens(self, campaign_id: str) -> tuple[set[str], set[str], set[str]]:
        """(exact-protected, fuzzy-protected, phonetic keys).

        Campaign entity/alias tokens get the full treatment: exact,
        distance-1 AND phonetic protection ("jink" next to the PC "Jinx"
        is name territory). Official 5e tokens (thunderous, aboleth,
        tiamat) are exact-only: ~650 rulebook words in the distance-1
        loop would both crush throughput and shadow curated fixes, and
        their phonetic keys are coarse enough to shield everyday
        misspellings ("becuase" shares a key with "pegasus"). A curated
        Tier-A misspelling is never protected by the rulebook list —
        "wich" must stay fixable even if some official name contains it.
        """
        from .dnd5e import protected_tokens as official_5e_tokens

        fuzzy: set[str] = set()
        keys: set[str] = set()
        for entity in self.memory.entities(campaign_id):
            for token in entity.name.casefold().split():
                fuzzy.add(token)
                keys.add(phonetic_key(token))
        for alias in self.memory.aliases(campaign_id):
            for token in (alias.observed + " " + alias.canonical).casefold().split():
                fuzzy.add(token)
                keys.add(phonetic_key(token))
        keys.discard("")
        exact = fuzzy | (official_5e_tokens() - set(known_misspellings()))
        return exact, fuzzy, keys

    def find_corrections(
        self,
        parsed: ParsedTranscript,
        campaign_id: str,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    ) -> list[SpellCorrection]:
        protected_words, fuzzy_protected, protected_keys = \
            self._protected_tokens(campaign_id)
        misspellings = known_misspellings()

        # Repetition census: an unknown token used again and again is a term
        # or a name (Sanar, Gonf), never a typo to "fix".
        occurrence: Counter[str] = Counter()
        for entry in parsed.entries:
            occurrence.update(_TOKEN.findall(entry.text))

        corrections: list[SpellCorrection] = []
        for entry in parsed.entries:
            for match in _TOKEN.finditer(entry.text):
                token = match.group(0)
                if len(token) < MIN_TOKEN_LENGTH:
                    continue

                # Entity lane: known names/aliases — exact, phonetic, or one
                # edit away ("jink" next to the PC "Jinx" is name territory,
                # never dictionary territory).
                if token in protected_words or phonetic_key(token) in protected_keys:
                    continue
                if any(
                    damerau_levenshtein(token, protected, cap=1) <= 1
                    for protected in fuzzy_protected
                ):
                    continue

                # Tier A: curated famous misspellings — high confidence.
                if token in misspellings:
                    corrections.append(
                        SpellCorrection(
                            line_number=entry.line_number,
                            original=token,
                            corrected=misspellings[token],
                            confidence=0.95,
                            context=entry.text[:160],
                        )
                    )
                    continue

                # Tier B: true non-words only, with structural guards.
                if is_dictionary_word(token):
                    continue
                if _is_derivative_of_dictionary_word(token):
                    continue
                if occurrence[token] >= REPEAT_AS_TERM:
                    continue  # repeated unknown = term/name, entity lane's call
                found = best_correction(token, min_zipf=MIN_TARGET_ZIPF)
                if found is None:
                    continue
                corrected, target_zipf, margin = found
                if margin < MIN_MARGIN:
                    continue
                confidence = _confidence(target_zipf, margin, len(token))
                if confidence < min_confidence:
                    continue
                corrections.append(
                    SpellCorrection(
                        line_number=entry.line_number,
                        original=token,
                        corrected=corrected,
                        confidence=confidence,
                        context=entry.text[:160],
                    )
                )
        return corrections

    def apply_corrections(
        self,
        mapped_path: str | Path,
        corrections: list[SpellCorrection],
        *,
        campaign_id: str,
        session_id: str,
        actor: str = "engine-spellcheck",
    ) -> int:
        """Apply lane-1 corrections to the Mapped transcript, atomically.

        Refuses anything that is not clearly a Mapped transcript. Each applied
        correction is audited under ``auto_spell_correction`` with original,
        corrected, confidence and line — nothing is written to entities,
        aliases, facts or the review queue.
        """
        path = Path(mapped_path)
        name = path.name.casefold()
        if "unmapped" in name or "mapped" not in name:
            raise ValueError(
                f"refusing to modify {path.name!r}: automatic spelling "
                "corrections may only be applied to a Mapped transcript"
            )
        if not corrections:
            return 0
        lines = path.read_text(encoding="utf-8-sig").splitlines(keepends=True)
        applied = 0
        for correction in corrections:
            index = correction.line_number - 1
            if index >= len(lines):
                continue
            pattern = re.compile(rf"\b{re.escape(correction.original)}\b")
            new_line, hits = pattern.subn(correction.corrected, lines[index])
            if hits:
                lines[index] = new_line
                applied += 1
                self.memory.log_auto_correction(
                    campaign_id,
                    session_id,
                    original=correction.original,
                    corrected=correction.corrected,
                    confidence=correction.confidence,
                    line_number=correction.line_number,
                    source_path=str(path),
                    actor=actor,
                )
        if applied:
            # Same-directory temp + atomic replace, preserving the original on
            # any failure. The Unmapped transcript is never involved.
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=path.name, suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                    f.write("".join(lines))
                os.replace(tmp_name, path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        return applied


def is_ordinary_word_candidate(text: str, reasons: set[str] | frozenset[str]) -> bool:
    """True when a detected candidate is ordinary vocabulary, not a name.

    Used to keep the review queue entity-only: a single-token candidate whose
    word is common English is dropped unless a strong naming signal saw it
    (introduction, travel target, known-entity mention, ...). Multi-word
    candidates ("Silver Spire") are never filtered here.
    """
    strong = {"self_introduction", "introduction", "location_of_phrase",
              "travel_target", "known_entity_mention"}
    if strong & set(reasons):
        return False
    tokens = text.split()
    if len(tokens) != 1:
        return False
    return zipf(tokens[0]) >= 3.3
