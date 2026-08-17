"""Fact extraction and verification — the engine's Luna and Terra.

Extraction: one structured provider call per transcript chunk, constrained to
the chunk's ContextPackage. Every returned fact must cite an exact quote that
deterministically exists in the package; anything else is rejected before it
can touch memory.

Verification: a second, independent role re-reads each fact against only its
cited evidence and the package. Disagreement or uncertainty routes to the
review queue (the human is this engine's Sol until the adjudicator role is
enabled).

Epistemic ceilings from the deterministic cue classifier are applied *after*
the model's answer: a fact whose evidence line reads as a joke or table talk
can never be stored stronger than TABLE_TALK, whatever the model said.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .dnd_patterns import (
    WORLD_FACT_MODES,
    assess_entry,
    assess_speaker_mode,
    assess_time_status,
)
from .memory import CampaignMemory
from .providers import ChatProvider, ValidationFailed, call_role
from .schemas import (
    AI_ASSIGNABLE_STATUSES,
    ChangeType,
    ContextPackage,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    ReviewItem,
    SpeakerMode,
    TimeStatus,
    Verdict,
)
from .transcript import TranscriptChunk

EXTRACTOR_VERSION = "engine-extractor-v1"
VERIFIER_VERSION = "engine-verifier-v1"

_EXTRACT_SYSTEM = """You are a meticulous D&D session fact extractor for one campaign.
Rules you must follow exactly:
- Use ONLY the provided sources. Never invent or infer beyond them.
- Every fact must include an exact supporting quote copied verbatim from a source.
- Distinguish in-game events from out-of-character chatter, jokes, planning and rules talk.
- Prefer DM statements over player speculation.
- Do not treat ordinary words, filler speech, or game mechanics as entities.
- Facts are structured triples: subject / relationship / object where possible.
- time_status: happened|current|planned|negated|hypothetical|unknown — plans,
  jokes and things that explicitly did not happen are NOT events.
- speaker_mode: dm_narration|mechanical_result|npc_dialogue|pc_dialogue|player_statement|table_talk.
  An NPC speaking through the DM confirms only what the NPC CLAIMS.
Reply with ONLY JSON: {"facts": [{"statement": str, "subject": str, "relationship": str,
"object": str, "category": str, "change_type": str, "entities": [str], "status": str,
"time_status": str, "speaker_mode": str, "confidence": float, "quote": str,
"line_start": int, "line_end": int, "speaker": str,
"importance": "low|medium|high|critical"}]}"""

_VERIFY_SYSTEM = """You are an independent D&D fact verifier. For each fact, check whether
its quote genuinely supports the statement, using ONLY the provided sources.
Be skeptical: jokes, speculation and out-of-character talk do not support in-world facts.
Reply with ONLY JSON: {"verdicts": [{"fact_id": str, "verdict": "supported|contradicted|uncertain",
"confidence": float, "reason": str}]}"""

_CATEGORY_VALUES = {c.value for c in FactCategory}
_CHANGE_VALUES = {c.value for c in ChangeType}
_STATUS_VALUES = {s.value for s in AI_ASSIGNABLE_STATUSES}


@dataclass
class ExtractionResult:
    facts: list[Fact]
    rejected: list[dict[str, Any]]        # invalid items with reasons (audit)
    review_items: list[ReviewItem]
    context: ContextPackage


def _validate_extraction_payload(payload: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(payload, dict) or not isinstance(payload.get("facts"), list):
        return ["top level must be an object with a 'facts' array"]
    for index, item in enumerate(payload["facts"]):
        where = f"facts[{index}]"
        if not isinstance(item, dict):
            problems.append(f"{where} must be an object")
            continue
        if not str(item.get("statement", "")).strip():
            problems.append(f"{where}.statement missing")
        if item.get("category") not in _CATEGORY_VALUES:
            problems.append(f"{where}.category invalid")
        if item.get("change_type") not in _CHANGE_VALUES:
            problems.append(f"{where}.change_type invalid")
        if item.get("status") not in _STATUS_VALUES:
            problems.append(f"{where}.status invalid (canon is not AI-assignable)")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
            problems.append(f"{where}.confidence must be 0..1")
        if not str(item.get("quote", "")).strip():
            problems.append(f"{where}.quote missing")
        if not isinstance(item.get("entities", []), list):
            problems.append(f"{where}.entities must be a list")
        time_status = item.get("time_status")
        if time_status is not None and time_status not in {t.value for t in TimeStatus}:
            problems.append(f"{where}.time_status invalid")
        speaker_mode = item.get("speaker_mode")
        if speaker_mode is not None and speaker_mode not in {m.value for m in SpeakerMode}:
            problems.append(f"{where}.speaker_mode invalid")
    return problems


class FactExtractor:
    def __init__(self, provider: ChatProvider, memory: CampaignMemory):
        self.provider = provider
        self.memory = memory

    def extract(
        self,
        campaign_id: str,
        session_id: str,
        chunk: TranscriptChunk,
        context: ContextPackage,
        *,
        source_path: str,
        source_hash: str,
    ) -> ExtractionResult:
        user = (
            "Sources:\n"
            + "\n\n".join(
                f"[{s.source_id}] ({s.kind})\n{context.sections[s.source_id]}"
                for s in context.sources
            )
            + "\n\nExtract the facts from the transcript source [T1]."
        )
        try:
            payload, _ = call_role(
                self.provider,
                system=_EXTRACT_SYSTEM,
                user=user,
                validator=_validate_extraction_payload,
                max_tokens=3000,
            )
        except ValidationFailed as exc:
            review = ReviewItem(
                campaign_id=campaign_id,
                session_id=session_id,
                item_type="fact",
                subject=f"chunk {chunk.chunk_number} extraction failed validation",
                reason=str(exc),
                evidence=[chunk.text[:500]],
            )
            self.memory.enqueue_review(review, actor="engine-extractor")
            return ExtractionResult(facts=[], rejected=[], review_items=[review],
                                    context=context)

        facts: list[Fact] = []
        rejected: list[dict[str, Any]] = []
        review_items: list[ReviewItem] = []
        entry_by_line = {e.line_number: e for e in chunk.entries}

        for item in payload["facts"]:
            quote = str(item["quote"]).strip()
            # Deterministic evidence gate: the quote must exist in the package.
            if not context.contains_quote(quote):
                rejected.append({"item": item, "reason": "quote not found in sources"})
                continue
            line_start = int(item.get("line_start") or chunk.line_start)
            line_end = int(item.get("line_end") or line_start)
            line_start = max(chunk.line_start, min(line_start, chunk.line_end))
            line_end = max(line_start, min(line_end, chunk.line_end))

            status = EpistemicStatus(item["status"])
            # Epistemic ceiling from the deterministic cue classifier.
            anchor = entry_by_line.get(line_start)
            speaker_mode = SpeakerMode(item.get("speaker_mode") or "unclear")
            if anchor is not None:
                ceiling = assess_entry(anchor).status
                if ceiling.rank > status.rank:
                    status = ceiling
                detected_mode = assess_speaker_mode(anchor)
                if speaker_mode is SpeakerMode.UNCLEAR:
                    speaker_mode = detected_mode
                elif speaker_mode in WORLD_FACT_MODES and detected_mode not in WORLD_FACT_MODES:
                    # The model claimed DM-narration authority the line does
                    # not have; the deterministic classifier wins.
                    speaker_mode = detected_mode

            # NPC dialogue confirms only what the NPC claims — never a world
            # fact on its own.
            if speaker_mode is SpeakerMode.NPC_DIALOGUE and status.rank < EpistemicStatus.CHARACTER_BELIEF.rank:
                status = EpistemicStatus.CHARACTER_BELIEF

            # Time status: the deterministic cue classifier overrides a model
            # that turned a plan or an abandoned action into an event.
            time_status = TimeStatus(item.get("time_status") or "unknown")
            detected_time = assess_time_status(quote)
            if detected_time in (TimeStatus.NEGATED, TimeStatus.PLANNED, TimeStatus.HYPOTHETICAL):
                time_status = detected_time
            elif time_status is TimeStatus.UNKNOWN:
                time_status = detected_time
            if time_status in (TimeStatus.PLANNED, TimeStatus.HYPOTHETICAL, TimeStatus.NEGATED):
                if status.rank < EpistemicStatus.UNCONFIRMED_THEORY.rank:
                    status = EpistemicStatus.UNCONFIRMED_THEORY

            fact = Fact(
                statement=str(item["statement"]).strip(),
                category=FactCategory(item["category"]),
                change_type=ChangeType(item["change_type"]),
                entities=[str(e).strip() for e in item.get("entities", []) if str(e).strip()],
                status=status,
                confidence=float(item["confidence"]),
                subject=str(item.get("subject") or "").strip(),
                relationship=str(item.get("relationship") or "").strip(),
                object_=str(item.get("object") or "").strip(),
                time_status=time_status,
                speaker_mode=speaker_mode,
                importance=str(item.get("importance", "medium")),
                provenance=Provenance(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    source_path=source_path,
                    source_hash=source_hash,
                    line_start=line_start,
                    line_end=line_end,
                    speaker=str(item.get("speaker") or "") or None,
                    quote=quote,
                    extractor=EXTRACTOR_VERSION,
                    extractor_version="1",
                    context_manifest_hash=context.manifest_hash,
                ),
            )
            facts.append(fact)
        return ExtractionResult(facts=facts, rejected=rejected,
                                review_items=review_items, context=context)


@dataclass
class VerificationResult:
    verified: list[Fact]                 # verdict==supported
    disputed: list[Fact]                 # contradicted/uncertain -> review
    review_items: list[ReviewItem] = field(default_factory=list)


def _validate_verification_payload(expected_ids: set[str]):
    def validator(payload: Any) -> list[str]:
        problems: list[str] = []
        if not isinstance(payload, dict) or not isinstance(payload.get("verdicts"), list):
            return ["top level must be an object with a 'verdicts' array"]
        seen: set[str] = set()
        for index, item in enumerate(payload["verdicts"]):
            where = f"verdicts[{index}]"
            if not isinstance(item, dict):
                problems.append(f"{where} must be an object")
                continue
            fact_id = item.get("fact_id")
            if fact_id not in expected_ids:
                problems.append(f"{where}.fact_id unknown")
            elif fact_id in seen:
                problems.append(f"{where}.fact_id duplicated")
            else:
                seen.add(fact_id)
            if item.get("verdict") not in {v.value for v in Verdict}:
                problems.append(f"{where}.verdict invalid")
            confidence = item.get("confidence")
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                problems.append(f"{where}.confidence must be 0..1")
        missing = expected_ids - seen
        if missing and not problems:
            problems.append(f"missing verdicts for: {', '.join(sorted(missing))}")
        return problems

    return validator


class FactVerifier:
    def __init__(self, provider: ChatProvider, memory: CampaignMemory):
        self.provider = provider
        self.memory = memory

    def verify(
        self,
        campaign_id: str,
        session_id: str,
        facts: list[Fact],
        context: ContextPackage,
    ) -> VerificationResult:
        if not facts:
            return VerificationResult(verified=[], disputed=[])
        listing = "\n".join(
            f'- fact_id {f.fact_id}: "{f.statement}" | quote: "{f.provenance.quote}"'
            for f in facts
        )
        user = (
            "Sources:\n"
            + "\n\n".join(
                f"[{s.source_id}] ({s.kind})\n{context.sections[s.source_id]}"
                for s in context.sources
            )
            + f"\n\nFacts to verify:\n{listing}"
        )
        try:
            payload, _ = call_role(
                self.provider,
                system=_VERIFY_SYSTEM,
                user=user,
                validator=_validate_verification_payload({f.fact_id for f in facts}),
                max_tokens=2000,
            )
        except ValidationFailed as exc:
            items = []
            for fact in facts:
                fact.needs_review = True
                fact.review_reason = f"verifier output invalid: {exc}"
                items.append(self._review_item(campaign_id, session_id, fact))
            return VerificationResult(verified=[], disputed=list(facts), review_items=items)

        verdict_by_id = {v["fact_id"]: v for v in payload["verdicts"]}
        verified: list[Fact] = []
        disputed: list[Fact] = []
        review_items: list[ReviewItem] = []
        for fact in facts:
            verdict = verdict_by_id[fact.fact_id]
            fact.verdict = Verdict(verdict["verdict"])
            if fact.verdict is Verdict.SUPPORTED:
                verified.append(fact)
                continue
            fact.needs_review = True
            fact.review_reason = str(verdict.get("reason") or fact.verdict.value)
            if fact.status.rank < EpistemicStatus.UNCONFIRMED_THEORY.rank:
                fact.status = EpistemicStatus.UNCONFIRMED_THEORY
            disputed.append(fact)
            review_items.append(self._review_item(campaign_id, session_id, fact))
        return VerificationResult(verified=verified, disputed=disputed,
                                  review_items=review_items)

    def _review_item(self, campaign_id: str, session_id: str, fact: Fact) -> ReviewItem:
        item = ReviewItem(
            campaign_id=campaign_id,
            session_id=session_id,
            item_type="fact",
            subject=fact.statement,
            reason=fact.review_reason or "verification unresolved",
            evidence=[fact.provenance.quote],
            confidence=fact.confidence,
            suggestions=[{"fact_id": fact.fact_id, "verdict": fact.verdict.value if fact.verdict else None}],
        )
        self.memory.enqueue_review(item, actor=VERIFIER_VERSION)
        return item


def detect_memory_contradictions(
    memory: CampaignMemory, campaign_id: str, new_facts: list[Fact], *, actor: str
) -> list[tuple[str, str]]:
    """Flag verified new facts that clash with remembered ones.

    Heuristic pass: same entity, both facts carry death/status semantics with
    opposite polarity, or an explicit negation pair. Links are recorded for
    human resolution — nothing is merged or deleted.
    """
    linked: list[tuple[str, str]] = []
    for fact in new_facts:
        if fact.category not in (FactCategory.DEATH_OR_STATUS, FactCategory.STORY_EVENT):
            continue
        statement = fact.statement.casefold()
        for entity in fact.entities:
            for old in memory.facts_for_entity(campaign_id, entity):
                if old.fact_id == fact.fact_id:
                    continue
                old_statement = old.statement.casefold()
                if _oppose(statement, old_statement):
                    memory.record_contradiction(
                        campaign_id, old.fact_id, fact.fact_id,
                        note=f"possible conflict about {entity}", actor=actor,
                    )
                    linked.append((old.fact_id, fact.fact_id))
    return linked


_OPPOSITES = (
    ({"dead", "died", "dies", "killed", "slain"}, {"alive", "lives", "survived", "living"}),
    ({"destroyed", "ruined"}, {"intact", "standing", "rebuilt"}),
    ({"lost", "missing", "vanished"}, {"found", "returned", "recovered"}),
)


def _oppose(a: str, b: str) -> bool:
    words_a = set(a.replace(".", " ").split())
    words_b = set(b.replace(".", " ").split())
    for left, right in _OPPOSITES:
        if (words_a & left and words_b & right) or (words_a & right and words_b & left):
            return True
    return False
