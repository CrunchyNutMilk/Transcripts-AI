"""Name, alias and spelling resolution against one campaign's memory.

Given an observed name from a transcript, produce ranked, *explained*
suggestions from the campaign's known entities and approved aliases. The
resolver only suggests; applying a correction is the review layer's job, and
low-confidence matches are explicitly marked as needing a human.

Matching signals, combined per candidate:
- exact (case/space-insensitive) name or approved-alias match
- token containment ("Cloud" ⊂ "Black Storm Cloud")
- Damerau-Levenshtein ratio with length-tiered thresholds
  (short names must match exactly; longer names tolerate more)
- phonetic key equality (speech-to-text confusions, e.g. Aragon/Daragon
  differ by an initial consonant Whisper often drops)
- prior human feedback: a previously rejected correction is never re-proposed

Every suggestion carries machine reasons + a plain-English explanation.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from .memory import CampaignMemory
from .phonetics import damerau_levenshtein, phonetic_key, similarity_ratio
from .schemas import EntityRecord

# Length-tiered acceptance thresholds (mirrors the bot's proven values).
def fuzzy_threshold(length: int) -> float:
    if length <= 3:
        return 1.01  # effectively exact-only
    if length <= 5:
        return 0.90
    if length <= 8:
        return 0.80
    return 0.76


AUTO_SUGGEST_FLOOR = 0.55   # below this we don't even suggest
CONFIDENT_FLOOR = 0.85      # at/above this the UI may preselect (never auto-apply)


# -- confidence bands (configurable via env) ---------------------------------


class BandAction(str, Enum):
    AUTO_LINK = "auto_link"          # 0.95+ AND strong evidence: link to existing
    SUGGEST = "suggest"              # 0.80-0.94: propose, request confirmation
    SAVE_FOR_REVIEW = "save_for_review"  # 0.60-0.79
    DROP = "drop"                    # below 0.60: do not create or alter anything


def _band(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def decide_band_action(confidence: float, *, strong_evidence: bool = False) -> BandAction:
    """Map a confidence score to the configured action band.

    AUTO_LINK additionally requires ``strong_evidence`` (an exact match or a
    human-approved alias) — a confident-sounding score alone never auto-links,
    and auto-link only ever attaches to an EXISTING entity; creation always
    goes through review.
    """
    auto = _band("ENGINE_BAND_AUTO_LINK", 0.95)
    suggest = _band("ENGINE_BAND_SUGGEST", 0.80)
    review = _band("ENGINE_BAND_REVIEW", 0.60)
    if confidence >= auto and strong_evidence:
        return BandAction.AUTO_LINK
    if confidence >= suggest:
        return BandAction.SUGGEST
    if confidence >= review:
        return BandAction.SAVE_FOR_REVIEW
    return BandAction.DROP


@dataclass
class Suggestion:
    canonical: str
    entity_id: str | None
    score: float                    # 0..1
    reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    via_alias: str | None = None
    band: BandAction = BandAction.SAVE_FOR_REVIEW

    @property
    def confident(self) -> bool:
        return self.score >= CONFIDENT_FLOOR


@dataclass
class Resolution:
    observed: str
    suggestions: list[Suggestion]
    needs_human: bool
    note: str = ""

    @property
    def best(self) -> Suggestion | None:
        return self.suggestions[0] if self.suggestions else None

    @property
    def band_action(self) -> BandAction:
        return self.best.band if self.best else BandAction.DROP


def _fold(name: str) -> str:
    return " ".join(name.casefold().split())


def _token_containment(short: str, long: str) -> bool:
    short_tokens = {t for t in _fold(short).split() if len(t) > 3}
    long_tokens = set(_fold(long).split())
    return bool(short_tokens) and short_tokens.issubset(long_tokens)


class NameResolver:
    def __init__(self, memory: CampaignMemory):
        self.memory = memory

    def resolve(
        self, campaign_id: str, observed: str, *, context_hint: str = "", limit: int = 3
    ) -> Resolution:
        observed_clean = observed.strip()
        folded = _fold(observed_clean)
        if not folded:
            return Resolution(observed=observed, suggestions=[], needs_human=False,
                              note="empty name")

        # 1. Approved alias — the strongest signal we have.
        alias = self.memory.resolve_alias(campaign_id, observed_clean)
        if alias is not None:
            score = 1.0 if alias.human_approved else 0.9
            suggestion = Suggestion(
                canonical=alias.canonical,
                entity_id=alias.entity_id,
                score=score,
                reasons=["approved_alias" if alias.human_approved else "engine_alias"],
                explanation=(
                    f'"{observed_clean}" is a previously approved alias of '
                    f'"{alias.canonical}"' + (f" ({alias.reason})" if alias.reason else "")
                ),
                via_alias=alias.alias_id,
                band=decide_band_action(score, strong_evidence=alias.human_approved),
            )
            return Resolution(observed=observed_clean, suggestions=[suggestion],
                              needs_human=suggestion.band is not BandAction.AUTO_LINK)

        entities = self.memory.entities(campaign_id)

        # 2. Exact canonical name.
        for entity in entities:
            if _fold(entity.name) == folded:
                suggestion = Suggestion(
                    canonical=entity.name,
                    entity_id=entity.entity_id,
                    score=1.0,
                    reasons=["exact_match"],
                    explanation=f'"{observed_clean}" exactly matches the known '
                                f"{entity.kind.value} \"{entity.name}\"",
                    band=decide_band_action(1.0, strong_evidence=True),
                )
                return Resolution(
                    observed=observed_clean, suggestions=[suggestion],
                    needs_human=suggestion.band is not BandAction.AUTO_LINK,
                )

        # 3. Fuzzy / phonetic / containment candidates — against entity
        # names AND recorded alias spellings ("Valadar" is far from
        # "Vel Nadar" but one edit from the known alias "Val Nadar").
        threshold = fuzzy_threshold(len(folded))
        observed_key = phonetic_key(observed_clean)
        scored: list[Suggestion] = []
        for entity in entities:
            candidate = self._score_candidate(observed_clean, folded, observed_key,
                                              threshold, entity)
            if candidate is not None:
                scored.append(candidate)
        entity_kinds = {e.name.casefold(): e.kind.value for e in entities}
        for alias_row in self.memory.aliases(campaign_id):
            hit = _score_name(observed_clean, folded, observed_key, threshold,
                              alias_row.observed)
            if hit is None:
                continue
            score, reasons, parts = hit
            kind = entity_kinds.get(alias_row.canonical.casefold(), "entity")
            scored.append(Suggestion(
                canonical=alias_row.canonical,
                entity_id=alias_row.entity_id,
                score=score,
                reasons=reasons + ["via_recorded_alias"],
                explanation=(
                    f'"{observed_clean}" {"; ".join(parts)} — a recorded '
                    f'spelling of the known {kind} "{alias_row.canonical}"'
                ),
                via_alias=alias_row.alias_id,
            ))
        # One suggestion per canonical name: keep the strongest evidence.
        best_by_canonical: dict[str, Suggestion] = {}
        for suggestion in scored:
            key = _fold(suggestion.canonical)
            current = best_by_canonical.get(key)
            if current is None or suggestion.score > current.score:
                best_by_canonical[key] = suggestion
        scored = list(best_by_canonical.values())

        # 4. Feedback filter, then banding: previously rejected suggestions
        # and DROP-band scores never surface; fuzzy matches are never strong
        # evidence, so AUTO_LINK is impossible here by construction.
        kept: list[Suggestion] = []
        for suggestion in scored:
            subject = f"{observed_clean}->{suggestion.canonical}"
            if self.memory.was_rejected_before(campaign_id, "correction_rejected", subject):
                continue
            suggestion.band = decide_band_action(suggestion.score, strong_evidence=False)
            if suggestion.band is BandAction.DROP:
                continue
            kept.append(suggestion)
        kept.sort(key=lambda s: (-s.score, _fold(s.canonical)))
        kept = kept[:limit]

        if not kept:
            return Resolution(
                observed=observed_clean, suggestions=[], needs_human=True,
                note="no plausible match in this campaign; may be a new entity",
            )
        # A resolver suggestion is never auto-applied: even a confident fuzzy
        # match goes to review, preselected.
        return Resolution(observed=observed_clean, suggestions=kept, needs_human=True)

    def _score_candidate(
        self,
        observed: str,
        folded: str,
        observed_key: str,
        threshold: float,
        entity: EntityRecord,
    ) -> Suggestion | None:
        hit = _score_name(observed, folded, observed_key, threshold, entity.name)
        if hit is None:
            return None
        score, reasons, parts = hit
        explanation = (
            f'"{observed}" {"; ".join(parts)} '
            f"(known {entity.kind.value} in this campaign)"
        )
        return Suggestion(
            canonical=entity.name,
            entity_id=entity.entity_id,
            score=round(score, 4),
            reasons=reasons,
            explanation=explanation,
        )


def _score_name(
    observed: str,
    folded: str,
    observed_key: str,
    threshold: float,
    target: str,
) -> tuple[float, list[str], list[str]] | None:
    """Fuzzy evidence that ``observed`` is ``target``: (score, reasons,
    explanation parts), or None below the suggestion floor."""
    target_folded = _fold(target)
    reasons: list[str] = []
    parts: list[str] = []
    score = 0.0

    ratio = similarity_ratio(folded, target_folded)
    distance = damerau_levenshtein(folded, target_folded, cap=3)
    if ratio >= threshold:
        score = max(score, ratio)
        reasons.append(f"edit_distance_{distance}")
        parts.append(f"spelled within {distance} edit(s) of \"{target}\"")

    # Whisper loves eating the space in multi-word names: "Silverspire",
    # "Valadar" for "Val Nadar". Compare with spaces squashed too.
    squashed, target_squashed = folded.replace(" ", ""), target_folded.replace(" ", "")
    if (folded != target_folded and len(target_squashed) >= 6
            and (squashed == target_squashed
                 or damerau_levenshtein(squashed, target_squashed, cap=2) <= 1)):
        score = max(score, 0.90 if squashed == target_squashed else 0.84)
        reasons.append("spacing_variant")
        parts.append(f'matches "{target}" once spacing is ignored')

    if _token_containment(folded, target_folded) or _token_containment(
        target_folded, folded
    ):
        score = max(score, 0.92)
        reasons.append("token_containment")
        parts.append(f'shares its distinctive word(s) with "{target}"')

    if observed_key and observed_key == phonetic_key(target):
        # Phonetic equality alone gives a suggestion floor, and boosts an
        # existing fuzzy score slightly; it never reaches CONFIDENT_FLOOR
        # by itself.
        score = max(score, 0.80 if not reasons else min(score + 0.05, 0.97))
        reasons.append("phonetic_match")
        parts.append(f'sounds like "{target}" when spoken')

    if not reasons or score < AUTO_SUGGEST_FLOOR:
        return None
    return score, reasons, parts
