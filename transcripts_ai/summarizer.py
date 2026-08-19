"""Structured session summariser.

The summary is derived from the verified facts + transcript context, not from
an unconstrained "summarise this" prompt. Confirmed and uncertain material are
kept separate all the way to the rendered Markdown, and a deterministic
post-pass drops any confirmed claim that names an entity absent from the
sources (the no-hallucination gate). Missing information is rendered as
"None mentioned" — never guessed.
"""
from __future__ import annotations

import re
from typing import Any

from .dnd_patterns import initiative_order
from .memory import CampaignMemory
from .providers import ChatProvider, ValidationFailed, call_role
from .schemas import (
    ContextPackage,
    EpistemicStatus,
    Fact,
    ReviewItem,
    SessionSummary,
    SummarySection,
)
from .transcript import ParsedTranscript

SUMMARIZER_VERSION = "engine-summarizer-v1"

SECTION_TITLES = (
    "Key Events",
    "Important Dialogue & Revealed Information",
    "Decisions Made",
    "Plans",
    "Risks & Unresolved Questions",
    "Quests",
    "Loot",
    "Combat",
    "Vault Update Suggestions",
)

_SYSTEM = """You are a D&D session summariser for one campaign.
Use ONLY the provided sources; never invent names, events or numbers.
Chronological order. Separate confirmed material (backed by the verified fact
ledger or direct DM statements) from uncertain material (speculation, plans,
jokes, unverified claims). If a section has nothing, return empty lists.
Reply with ONLY JSON:
{"party": [str], "npcs_and_groups": [str], "locations": [str],
 "sections": {<title>: {"confirmed": [str], "uncertain": [str]}}}
where <title> keys are exactly: %s""" % ", ".join(f'"{t}"' for t in SECTION_TITLES)


def _validate_summary_payload(payload: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["top level must be an object"]
    for key in ("party", "npcs_and_groups", "locations"):
        if not isinstance(payload.get(key), list):
            problems.append(f"'{key}' must be a list")
    sections = payload.get("sections")
    if not isinstance(sections, dict):
        return problems + ["'sections' must be an object"]
    for title in SECTION_TITLES:
        section = sections.get(title)
        if not isinstance(section, dict):
            problems.append(f"section {title!r} missing")
            continue
        for bucket in ("confirmed", "uncertain"):
            if not isinstance(section.get(bucket), list):
                problems.append(f"section {title!r}.{bucket} must be a list")
    return problems


_WORD = re.compile(r"[A-Za-z][\w'\-]+")


class SessionSummarizer:
    def __init__(self, provider: ChatProvider, memory: CampaignMemory):
        self.provider = provider
        self.memory = memory

    def summarize(
        self,
        campaign_id: str,
        session_id: str,
        *,
        game_name: str,
        session_date: str,
        parsed: ParsedTranscript,
        verified_facts: list[Fact],
        context: ContextPackage,
    ) -> tuple[SessionSummary | None, list[ReviewItem]]:
        ledger = "\n".join(
            f"- [{f.status.value}] ({f.category.value}/{f.change_type.value}) {f.statement}"
            for f in verified_facts
        ) or "(no verified facts)"
        user = (
            "Sources:\n"
            + "\n\n".join(
                f"[{s.source_id}] ({s.kind})\n{context.sections[s.source_id]}"
                for s in context.sources
            )
            + f"\n\nVerified fact ledger:\n{ledger}\n\nProduce the session summary JSON."
        )
        try:
            payload, _ = call_role(
                self.provider,
                system=_SYSTEM,
                user=user,
                validator=_validate_summary_payload,
                max_tokens=3500,
            )
        except ValidationFailed as exc:
            item = ReviewItem(
                campaign_id=campaign_id,
                session_id=session_id,
                item_type="summary_claim",
                subject="summary generation failed validation",
                reason=str(exc),
                evidence=[],
            )
            self.memory.enqueue_review(item, actor=SUMMARIZER_VERSION)
            return None, [item]

        # No-hallucination gate: confirmed claims must only use known words —
        # names appearing in the sources, the ledger, or campaign memory.
        allowed_text = (
            context.combined_text() + "\n" + ledger + "\n"
            + " ".join(e.name for e in self.memory.entities(campaign_id))
        ).casefold()
        allowed_words = {w.casefold() for w in _WORD.findall(allowed_text)}

        review_items: list[ReviewItem] = []
        sections: list[SummarySection] = []
        for title in SECTION_TITLES:
            data = payload["sections"][title]
            confirmed: list[str] = []
            uncertain = [str(x) for x in data["uncertain"]]
            for claim in (str(x) for x in data["confirmed"]):
                unknown = [
                    w for w in _WORD.findall(claim)
                    if w[0].isupper() and w.casefold() not in allowed_words
                ]
                if unknown:
                    item = ReviewItem(
                        campaign_id=campaign_id,
                        session_id=session_id,
                        item_type="summary_claim",
                        subject=claim,
                        reason=f"confirmed claim names unknown entities: {', '.join(unknown)}",
                        evidence=[],
                    )
                    self.memory.enqueue_review(item, actor=SUMMARIZER_VERSION)
                    review_items.append(item)
                    uncertain.append(f"{claim} (unverified names: {', '.join(unknown)})")
                else:
                    confirmed.append(claim)
            sections.append(SummarySection(title=title, confirmed=confirmed,
                                           uncertain=uncertain))

        # Deterministic extras the model cannot fake:
        order = initiative_order(parsed.entries)
        combat = next(s for s in sections if s.title == "Combat")
        if order:
            combat.confirmed.append(
                "Initiative order (as spoken): "
                + ", ".join(f"{name} ({value})" for name, value in order)
            )
        else:
            combat.uncertain.append("Initiative order: not stated in transcript")

        summary = SessionSummary(
            campaign_id=campaign_id,
            session_id=session_id,
            game_name=game_name,
            session_date=session_date,
            party=[str(x) for x in payload["party"]],
            sections=sections,
            manifest_hash=context.manifest_hash,
            npcs_and_groups=[str(x) for x in payload["npcs_and_groups"]],
            locations=[str(x) for x in payload["locations"]],
            generator=SUMMARIZER_VERSION,
        )
        return summary, review_items


def render_markdown(summary: SessionSummary, *, npcs: list[str] | None = None,
                    locations: list[str] | None = None) -> str:
    lines = [
        f"# {summary.game_name} / {summary.session_date}",
        "",
        "## Names",
        "Party:",
        *([f"- {p}" for p in summary.party] or ["- None mentioned"]),
    ]
    if npcs is not None:
        lines += ["", "NPCs/Groups:"] + ([f"- {n}" for n in npcs] or ["- None mentioned"])
    if locations is not None:
        lines += ["", "## Places"] + ([f"- {l}" for l in locations] or ["- None mentioned"])
    for section in summary.sections:
        lines += ["", f"## {section.title}"]
        if section.confirmed:
            lines += ["**Confirmed:**"] + [f"- {c}" for c in section.confirmed]
        if section.uncertain:
            lines += ["**Uncertain / unconfirmed:**"] + [f"- {u}" for u in section.uncertain]
        if not section.confirmed and not section.uncertain:
            lines += ["- None mentioned"]
    lines += [
        "",
        "---",
        f"Generated by {summary.generator} v{summary.generator_version} from "
        f"context manifest `{summary.manifest_hash[:16]}`. Confirmed items are "
        "backed by the verified fact ledger; uncertain items are not canon.",
    ]
    return "\n".join(lines)
