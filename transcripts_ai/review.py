"""Human review workflow: the six actions and the learning they produce.

The engine proposes; humans decide. Applying a decision here updates campaign
memory (aliases, entities, feedback) so the next session benefits — the
transcript edit itself stays with the bot's hash-guarded MappedTranscriptEditor.

"Let AI Pick" is advisory only: it returns a recommendation with reasons and
never applies anything by itself, mirroring the bot's display-only Sol pick.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .memory import CampaignMemory
from .providers import ChatProvider, ValidationFailed, call_role
from .schemas import (
    AliasRecord,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    ReviewAction,
    ReviewItem,
)

REVIEWER_VERSION = "engine-reviewer-v1"

_PICK_SYSTEM = """You advise on one D&D name-review decision. Choose the best action for
the observed name using ONLY the provided suggestions and evidence.
Reply with ONLY JSON: {"action": "correct|alias|new|dont_know_yet",
"choice": str|null, "confidence": float, "reason": str}
"choice" must be one of the offered canonical names for correct/alias, else null."""


@dataclass
class AIRecommendation:
    action: ReviewAction
    choice: str | None
    confidence: float
    reason: str
    applied: bool = False  # always False: display-only by contract


class ReviewCoordinator:
    def __init__(self, memory: CampaignMemory, *, reviewer: ChatProvider | None = None):
        self.memory = memory
        self.reviewer = reviewer

    # -- the six actions ----------------------------------------------------

    def correct(
        self, item: ReviewItem, *, canonical: str, actor: str,
        apply_to_all: bool = True,
    ) -> ReviewItem:
        self._require_human(actor)
        resolved = self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.CORRECT, actor=actor,
            detail={"canonical": canonical, "apply_to_all": apply_to_all},
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="correction_accepted",
            subject=f"{item.subject}->{canonical}", accepted=True, actor=actor,
            detail={"apply_to_all": apply_to_all},
        )
        # Learn the spelling for future sessions as an approved alias.
        if item.subject.strip().casefold() != canonical.strip().casefold():
            self.memory.add_alias(
                AliasRecord(
                    campaign_id=item.campaign_id,
                    observed=item.subject,
                    canonical=canonical,
                    entity_id=self._entity_id_for(item.campaign_id, canonical),
                    approved_by=actor,
                    reason="approved spelling correction",
                ),
                actor=actor,
            )
        return resolved

    def reject_suggestion(self, item: ReviewItem, *, canonical: str, actor: str) -> None:
        """The user said a proposed correction is wrong; never re-propose it."""
        self._require_human(actor)
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="correction_rejected",
            subject=f"{item.subject}->{canonical}", accepted=False, actor=actor,
        )

    def alias(self, item: ReviewItem, *, canonical: str, actor: str, reason: str = "") -> ReviewItem:
        self._require_human(actor)
        resolved = self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.ALIAS, actor=actor,
            detail={"canonical": canonical},
        )
        self.memory.add_alias(
            AliasRecord(
                campaign_id=item.campaign_id,
                observed=item.subject,
                canonical=canonical,
                entity_id=self._entity_id_for(item.campaign_id, canonical),
                approved_by=actor,
                reason=reason or "approved alias",
            ),
            actor=actor,
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="alias_added",
            subject=f"{item.subject}->{canonical}", accepted=True, actor=actor,
        )
        return resolved

    def new_entity(
        self, item: ReviewItem, *, kind: EntityKind, actor: str,
        vault_path: str | None = None, description: str = "",
    ) -> EntityRecord:
        self._require_human(actor)
        self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.NEW, actor=actor,
            detail={"kind": kind.value, "vault_path": vault_path},
        )
        entity = self.memory.upsert_entity(
            EntityRecord(
                name=item.subject,
                kind=kind,
                campaign_id=item.campaign_id,
                description=description,
                vault_path=vault_path,
                status=EpistemicStatus.CONFIRMED_CANON,
            ),
            actor=actor,
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="entity_confirmed",
            subject=item.subject, accepted=True, actor=actor,
            detail={"kind": kind.value},
        )
        return entity

    def not_entity(self, item: ReviewItem, *, actor: str) -> ReviewItem:
        """The candidate is an ordinary word/noise, not a name. Remembered so
        it is never proposed again for this campaign."""
        self._require_human(actor)
        resolved = self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.NOT_ENTITY, actor=actor,
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="entity_misclassified",
            subject=item.subject, accepted=True, actor=actor,
            detail={"decision": "not_an_entity"},
        )
        return resolved

    def save_for_review(self, item: ReviewItem, *, actor: str) -> ReviewItem:
        resolved = self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.SAVE_FOR_REVIEW,
            actor=actor,
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="deferred",
            subject=item.subject, accepted=True, actor=actor,
        )
        return resolved  # stays pending by design

    def dont_know_yet(self, item: ReviewItem, *, reason: str, actor: str) -> ReviewItem:
        self._require_human(actor)
        resolved = self.memory.resolve_review(
            item.campaign_id, item.item_id, action=ReviewAction.DONT_KNOW_YET,
            actor=actor, detail={"reason": reason},
        )
        self.memory.record_feedback(
            item.campaign_id, item.session_id, kind="deferred",
            subject=item.subject, accepted=True, actor=actor,
            detail={"reason": reason, "dont_know": True},
        )
        return resolved

    def let_ai_pick(self, item: ReviewItem) -> AIRecommendation:
        """Advisory recommendation. Never applies anything.

        With no external reviewer configured, the engine's own trained
        suggestion learner scores the offered candidates and explains its
        pick feature-by-feature — fully self-contained.
        """
        if self.reviewer is None:
            return self._native_pick(item)
        offered = [str(s.get("canonical") or s.get("choice") or "") for s in item.suggestions]
        offered = [o for o in offered if o]

        def validator(payload: Any) -> list[str]:
            problems: list[str] = []
            if not isinstance(payload, dict):
                return ["must be an object"]
            if payload.get("action") not in {"correct", "alias", "new", "dont_know_yet"}:
                problems.append("action invalid")
            choice = payload.get("choice")
            if payload.get("action") in {"correct", "alias"}:
                if choice not in offered:
                    problems.append(f"choice must be one of the offered names: {offered}")
            confidence = payload.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                problems.append("confidence must be 0..1")
            if not str(payload.get("reason", "")).strip():
                problems.append("reason required")
            return problems

        user = (
            f'Observed name: "{item.subject}"\n'
            f"Why it was flagged: {item.reason}\n"
            f"Evidence:\n" + "\n".join(f"- {e}" for e in item.evidence)
            + "\nOffered suggestions:\n"
            + ("\n".join(f"- {o}" for o in offered) or "- (none)")
        )
        try:
            payload, _ = call_role(
                self.reviewer, system=_PICK_SYSTEM, user=user, validator=validator,
                max_tokens=500,
            )
        except ValidationFailed as exc:
            return AIRecommendation(
                action=ReviewAction.DONT_KNOW_YET, choice=None, confidence=0.0,
                reason=f"AI pick failed validation: {exc}",
            )
        return AIRecommendation(
            action=ReviewAction(payload["action"]),
            choice=payload.get("choice"),
            confidence=float(payload["confidence"]),
            reason=str(payload["reason"]),
        )

    def _native_pick(self, item: ReviewItem) -> AIRecommendation:
        from .learner import SuggestionLearner

        offered = [str(s.get("canonical") or "") for s in item.suggestions]
        offered = [o for o in offered if o]
        if not offered:
            return AIRecommendation(
                action=ReviewAction.DONT_KNOW_YET, choice=None, confidence=0.0,
                reason="no candidates offered; likely a new entity or noise "
                       "— needs your judgement",
            )
        learner = SuggestionLearner(self.memory)
        scored = []
        for candidate in offered:
            probability, contributions = learner.score(
                item.campaign_id, item.subject, candidate
            )
            top = sorted(
                (f for f in contributions.items() if f[0] != "bias" and f[1] != 0),
                key=lambda kv: -abs(kv[1]),
            )[:3]
            scored.append((probability, candidate, top))
        scored.sort(key=lambda t: -t[0])
        probability, best, top = scored[0]
        reasons = ", ".join(f"{name} ({value:+.2f})" for name, value in top)
        if probability < 0.5:
            return AIRecommendation(
                action=ReviewAction.DONT_KNOW_YET, choice=None,
                confidence=round(probability, 3),
                reason=f'no offered candidate scores well (best "{best}" at '
                       f"{probability:.2f}; {reasons})",
            )
        return AIRecommendation(
            action=ReviewAction.CORRECT, choice=best,
            confidence=round(probability, 3),
            reason=f'learner favours "{best}" ({probability:.2f}): {reasons}',
        )

    # -- helpers ------------------------------------------------------------

    def _entity_id_for(self, campaign_id: str, name: str) -> str | None:
        entity = self.memory.find_entity(campaign_id, name)
        return entity.entity_id if entity else None

    @staticmethod
    def _require_human(actor: str) -> None:
        if not actor.startswith("human:"):
            raise PermissionError(
                f"review decisions require a human actor, got {actor!r}"
            )
