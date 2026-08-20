"""Session processing orchestrator.

Ties the components into the end-to-end run for one session:

    parse -> chunk -> per chunk: [context -> extract -> verify] (resumable)
          -> contradiction check -> memory writes -> summary -> report

Facts are written to memory as soon as their chunk completes, and each chunk's
completion is checkpointed against the chunk content hash, so an interrupted
run resumes where it stopped and an edited transcript reprocesses only what
changed. The source file is read once and never written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .detector import detect_names
from .dnd5e import is_registrable, kind_for, mentions_in_entries, nearest_official
from .extraction import (
    ExtractionResult,
    FactExtractor,
    FactVerifier,
    detect_memory_contradictions,
)
from .memory import CampaignMemory
from .native_extractor import extract_native_facts
from .native_summarizer import build_native_summary
from .providers import RoleRegistry
from .resolver import BandAction, NameResolver
from .retrieval import ContextBuilder
from .scenes import Scene, segment_scenes
from .schemas import (
    EntityRecord,
    EpistemicStatus,
    Fact,
    ReviewItem,
    SessionSummary,
    text_sha256,
)
from .session_context import SessionContext
from .spellcheck import is_ordinary_word_candidate
from .summarizer import SessionSummarizer, render_markdown
from .transcript import TranscriptEntry, chunk_transcript, parse_transcript, validate_chunks

PIPELINE_VERSION = "engine-pipeline-v1"


@dataclass
class SessionReport:
    campaign_id: str
    session_id: str
    chunks_total: int
    chunks_processed: int
    chunks_skipped_resume: int
    facts_verified: list[Fact] = field(default_factory=list)
    facts_disputed: list[Fact] = field(default_factory=list)
    facts_rejected: int = 0
    contradictions: list[tuple[str, str]] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)
    summary: SessionSummary | None = None
    summary_markdown: str = ""
    scenes: list[Scene] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class SessionPipeline:
    def __init__(
        self,
        memory: CampaignMemory,
        registry: RoleRegistry | None = None,
        *,
        context_budget_chars: int = 24_000,
    ):
        self.memory = memory
        self.registry = registry
        self.context_builder = ContextBuilder(memory, budget_chars=context_budget_chars)

    def process_session(
        self,
        *,
        campaign_id: str,
        session_id: str,
        transcript_path: str | Path,
        game_name: str,
        session_date: str,
        session_context: SessionContext | None = None,
    ) -> SessionReport:
        if self.registry is None:
            raise RuntimeError(
                "provider-backed processing needs a RoleRegistry; "
                "use process_session_native() for the self-contained path"
            )
        path = Path(transcript_path)
        text = path.read_text(encoding="utf-8-sig")
        parsed = parse_transcript(text, source_path=str(path))
        chunks = chunk_transcript(parsed)
        validate_chunks(parsed, chunks)

        # The mapping file is the strongest PC source: register mapped PCs
        # before anything else so detection/resolution can rely on them.
        if session_context is not None:
            if session_context.campaign_id != campaign_id:
                raise ValueError("session context belongs to a different campaign")
            session_context.register_pcs(self.memory, actor="human:mapping-file")

        extractor = FactExtractor(self.registry.provider_for("extractor"), self.memory)
        verifier = FactVerifier(self.registry.provider_for("verifier"), self.memory)

        report = SessionReport(
            campaign_id=campaign_id,
            session_id=session_id,
            chunks_total=len(chunks),
            chunks_processed=0,
            chunks_skipped_resume=0,
            scenes=segment_scenes(parsed.entries),
        )
        if parsed.unparsed_lines:
            report.warnings.append(
                f"{len(parsed.unparsed_lines)} line(s) did not parse as speaker turns"
            )
        if not chunks:
            report.warnings.append(
                "no parseable speaker turns; nothing processed, no summary"
            )
            return report

        all_verified: list[Fact] = []
        for chunk in chunks:
            chunk_hash = text_sha256(chunk.text)
            if self.memory.chunk_state(campaign_id, session_id, chunk.chunk_number,
                                       chunk_hash) == "done":
                report.chunks_skipped_resume += 1
                # Only genuinely verified facts re-enter the ledger on resume;
                # disputed/needs-review facts were remembered too but must not
                # be laundered into "verified" by a second run.
                all_verified.extend(
                    f for f in self.memory.facts_for_session(campaign_id, session_id)
                    if f.provenance.line_start >= chunk.line_start
                    and f.provenance.line_end <= chunk.line_end
                    and not f.needs_review
                )
                continue

            context = self.context_builder.build(
                campaign_id, session_id, chunk, task="extractor"
            )
            extraction: ExtractionResult = extractor.extract(
                campaign_id, session_id, chunk, context,
                source_path=str(path), source_hash=parsed.source_hash,
            )
            report.facts_rejected += len(extraction.rejected)
            report.review_items.extend(extraction.review_items)

            verification = verifier.verify(
                campaign_id, session_id, extraction.facts, context
            )
            report.review_items.extend(verification.review_items)
            report.facts_disputed.extend(verification.disputed)

            for fact in verification.verified:
                self.memory.remember_fact(fact, actor=PIPELINE_VERSION)
            # Disputed facts are remembered too — flagged, never silently kept
            # out of the record; the review queue decides their future.
            for fact in verification.disputed:
                self.memory.remember_fact(fact, actor=PIPELINE_VERSION)

            all_verified.extend(verification.verified)
            report.chunks_processed += 1
            self.memory.set_chunk_state(
                campaign_id, session_id, chunk.chunk_number, chunk_hash, "done",
                detail=f"verified={len(verification.verified)} disputed={len(verification.disputed)}",
            )

        report.facts_verified = all_verified
        report.contradictions = detect_memory_contradictions(
            self.memory, campaign_id, all_verified, actor=PIPELINE_VERSION
        )
        return self._finish_with_summary(report, campaign_id, session_id,
                                         game_name, session_date, parsed,
                                         chunks, all_verified)

    def _finish_with_summary(self, report, campaign_id, session_id, game_name,
                             session_date, parsed, chunks, all_verified):

        # Summary context: reuse the first chunk plus memory-derived sources.
        summarizer = SessionSummarizer(self.registry.provider_for("summarizer"), self.memory)
        summary_context = self.context_builder.build(
            campaign_id, session_id, chunks[0], task="summarizer"
        )
        summary, summary_reviews = summarizer.summarize(
            campaign_id,
            session_id,
            game_name=game_name,
            session_date=session_date,
            parsed=parsed,
            verified_facts=all_verified,
            context=summary_context,
        )
        report.review_items.extend(summary_reviews)
        if summary is not None:
            report.summary = summary
            report.summary_markdown = render_markdown(
                summary, npcs=summary.npcs_and_groups, locations=summary.locations
            )
            self.memory.remember_summary(
                campaign_id, session_id, report.summary_markdown,
                summary.manifest_hash, actor=PIPELINE_VERSION,
            )
        else:
            report.warnings.append("summary failed validation; queued for review")
        return report

    # ------------------------------------------------------------------
    # Self-contained path: no external AI of any kind.
    # ------------------------------------------------------------------

    def _register_official_mentions(
        self, campaign_id: str, session_id: str,
        entries: list[TranscriptEntry], source_path: str,
    ) -> list[str]:
        """Official 5e names mentioned in the session become kind-correct
        entities without human review — the reference list is a closed
        vocabulary, so there is nothing to invent and nothing to ask.
        Whether the party HAS the item stays a fact-level, evidence-gated
        question; this only teaches the engine the term and its kind."""
        registered: list[str] = []
        seen: set[str] = set()
        for mention in mentions_in_entries(entries):
            folded = mention.name.casefold()
            if folded in seen or not is_registrable(mention.name):
                continue
            seen.add(folded)
            self.memory.upsert_entity(
                EntityRecord(
                    name=mention.name,
                    kind=kind_for(mention.name),
                    campaign_id=campaign_id,
                    status=EpistemicStatus.STRONGLY_SUPPORTED,
                    description=f"Official D&D 5e {mention.category}",
                    attributes={
                        "official_5e": mention.category,
                        "first_seen_session": session_id,
                        "first_seen_line": mention.line,
                        "source_path": source_path,
                    },
                ),
                actor="engine:5e-reference",
            )
            registered.append(mention.name)
        return registered

    def _propose_entity_candidates(
        self,
        campaign_id: str,
        session_id: str,
        entries: list[TranscriptEntry],
        known: frozenset[str],
    ) -> list[ReviewItem]:
        """New-name candidates from this session, queued for human review.

        The engine never creates entities on its own: a detected name that
        is not already a known entity/alias (or an ordinary English word)
        becomes a review item carrying the resolver's suggestions, so the
        human decides with the six review actions. AUTO_LINK-grade matches
        are already known and are skipped — nothing to ask.
        """
        resolver = NameResolver(self.memory)
        items: list[ReviewItem] = []
        for candidate in detect_names(
            entries, known_names=frozenset(n.casefold() for n in known)
        ):
            reasons = set(candidate.reasons)
            if reasons == {"known_entity_mention"}:
                continue                     # already in memory; nothing to ask
            if is_ordinary_word_candidate(candidate.text, reasons):
                continue
            resolution = resolver.resolve(campaign_id, candidate.text)
            best = resolution.best
            if best is not None and best.band is BandAction.AUTO_LINK:
                continue                     # confidently known under another name
            suggestions = [
                {"canonical": s.canonical, "score": round(s.score, 3),
                 "explanation": s.explanation}
                for s in resolution.suggestions
            ]
            official = nearest_official(candidate.text)
            if official is not None and not any(
                s["canonical"].casefold() == official.official.casefold()
                for s in suggestions
            ):
                suggestions.append({
                    "canonical": official.official,
                    "score": official.score,
                    "explanation": f"official 5e {official.category}",
                })
            item = ReviewItem(
                campaign_id=campaign_id,
                session_id=session_id,
                item_type="entity",
                subject=candidate.text,
                reason=(f"new name candidate ({', '.join(sorted(reasons))}; "
                        f"{candidate.mentions} mention(s), "
                        f"first at line {candidate.entry_line})"),
                evidence=[candidate.context],
                suggestions=suggestions,
                confidence=min(0.9, 0.45 + 0.1 * candidate.mentions),
            )
            self.memory.enqueue_review(item, actor="native-pipeline")
            items.append(item)
        return items

    def process_session_native(
        self,
        *,
        campaign_id: str,
        session_id: str,
        transcript_path: str | Path,
        game_name: str,
        session_date: str,
        session_context: SessionContext | None = None,
    ) -> SessionReport:
        """Process a session with the engine's own intelligence only.

        Pattern-extracted facts are evidence-correct by construction; the
        deterministic verifier here is the contradiction detector plus the
        confidence gate (low-confidence facts flow to the review queue rather
        than memory as verified). The summary is extractive, so it cannot
        contain anything unsourced.
        """
        path = Path(transcript_path)
        text = path.read_text(encoding="utf-8-sig")
        parsed = parse_transcript(text, source_path=str(path))
        chunks = chunk_transcript(parsed)
        validate_chunks(parsed, chunks)
        scenes = segment_scenes(parsed.entries)

        if session_context is not None:
            if session_context.campaign_id != campaign_id:
                raise ValueError("session context belongs to a different campaign")
            session_context.register_pcs(self.memory, actor="human:mapping-file")

        report = SessionReport(
            campaign_id=campaign_id,
            session_id=session_id,
            chunks_total=len(chunks),
            chunks_processed=len(chunks),
            chunks_skipped_resume=0,
            scenes=scenes,
        )
        if parsed.unparsed_lines:
            report.warnings.append(
                f"{len(parsed.unparsed_lines)} line(s) did not parse as speaker turns"
            )
        if not chunks:
            report.warnings.append(
                "no parseable speaker turns; nothing processed, no summary"
            )
            return report

        self._register_official_mentions(campaign_id, session_id,
                                         parsed.entries, str(path))
        # Exact stored casing — the normaliser must never reconstruct names.
        # (Freshly registered official names join it, so recasing works.)
        known = frozenset(e.name for e in self.memory.entities(campaign_id))
        facts = extract_native_facts(
            parsed.entries,
            campaign_id=campaign_id,
            session_id=session_id,
            source_path=str(path),
            source_hash=parsed.source_hash,
            known_entities=known,
        )

        verified: list[Fact] = []
        for fact in facts:
            if fact.confidence < 0.6:
                fact.needs_review = True
                fact.review_reason = "native extraction below confidence gate"
                item = ReviewItem(
                    campaign_id=campaign_id,
                    session_id=session_id,
                    item_type="fact",
                    subject=fact.statement,
                    reason=fact.review_reason,
                    evidence=[fact.provenance.quote],
                    confidence=fact.confidence,
                )
                self.memory.enqueue_review(item, actor="native-pipeline")
                report.review_items.append(item)
                report.facts_disputed.append(fact)
            else:
                verified.append(fact)
            self.memory.remember_fact(fact, actor=PIPELINE_VERSION)

        report.facts_verified = verified
        report.review_items.extend(
            self._propose_entity_candidates(
                campaign_id, session_id, parsed.entries, known
            )
        )
        report.contradictions = detect_memory_contradictions(
            self.memory, campaign_id, verified, actor=PIPELINE_VERSION
        )

        summary_context = self.context_builder.build(
            campaign_id, session_id, chunks[0], task="summarizer"
        )
        summary = build_native_summary(
            self.memory,
            campaign_id=campaign_id,
            session_id=session_id,
            game_name=game_name,
            session_date=session_date,
            parsed=parsed,
            facts=facts,
            scenes=scenes,
            manifest_hash=summary_context.manifest_hash,
        )
        report.summary = summary
        report.summary_markdown = render_markdown(
            summary, npcs=summary.npcs_and_groups, locations=summary.locations
        )
        self.memory.remember_summary(
            campaign_id, session_id, report.summary_markdown,
            summary.manifest_hash, actor=PIPELINE_VERSION,
        )
        return report
