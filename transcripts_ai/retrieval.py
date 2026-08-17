"""Campaign-context retrieval.

Builds the budgeted ContextPackage for each AI task instead of dumping the
whole vault into every request. Priority order: the transcript slice itself,
then approved aliases for names seen in it, then memory facts about the
entities involved, then previous session summaries. Every included source is
recorded in the package manifest; downstream validation rejects any AI claim
whose evidence is not inside the manifest.
"""
from __future__ import annotations

from .detector import detect_names
from .memory import CampaignMemory
from .schemas import ContextPackage, ContextSource, text_sha256
from .transcript import TranscriptChunk

DEFAULT_BUDGET_CHARS = 24_000


class ContextBuilder:
    def __init__(self, memory: CampaignMemory, *, budget_chars: int = DEFAULT_BUDGET_CHARS):
        self.memory = memory
        self.budget_chars = budget_chars

    def build(
        self,
        campaign_id: str,
        session_id: str,
        chunk: TranscriptChunk,
        *,
        task: str,
    ) -> ContextPackage:
        sources: list[ContextSource] = []
        sections: dict[str, str] = {}
        remaining = self.budget_chars

        def add(source_id: str, kind: str, path: str, text: str) -> bool:
            nonlocal remaining
            if not text or len(text) > remaining:
                return False
            sources.append(
                ContextSource(
                    source_id=source_id,
                    kind=kind,
                    path=path,
                    content_hash=text_sha256(text),
                    chars=len(text),
                )
            )
            sections[source_id] = text
            remaining -= len(text)
            return True

        # 1. The transcript slice — always first, never trimmed.
        chunk_text = chunk.text
        if len(chunk_text) > remaining:
            raise ValueError(
                f"chunk of {len(chunk_text)} chars exceeds context budget {self.budget_chars}"
            )
        add("T1", "transcript",
            f"session:{session_id}:chunk:{chunk.chunk_number}", chunk_text)

        # 2. Which known entities does this slice mention?
        known = frozenset(
            e.name.casefold() for e in self.memory.entities(campaign_id)
        )
        detected = detect_names(chunk.entries, known_names=known)
        mentioned = [d.text for d in detected]

        # 3. Approved aliases relevant to the slice.
        alias_lines = []
        for alias in self.memory.aliases(campaign_id):
            if alias.observed.casefold() in {m.casefold() for m in mentioned} or any(
                alias.canonical.casefold() in m.casefold() for m in mentioned
            ):
                alias_lines.append(
                    f'- "{alias.observed}" means "{alias.canonical}"'
                    + (f" ({alias.reason})" if alias.reason else "")
                )
        if alias_lines:
            add("A1", "alias", f"aliases:{campaign_id}",
                "Approved aliases for this campaign:\n" + "\n".join(alias_lines))

        # 4. Memory facts about the mentioned entities (strongest first).
        fact_lines: list[str] = []
        seen_fact_ids: set[str] = set()
        for name in mentioned:
            for fact in self.memory.facts_for_entity(campaign_id, name):
                if fact.fact_id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact.fact_id)
                fact_lines.append(
                    f"- [{fact.status.value}] {fact.statement} "
                    f"(session {fact.provenance.session_id}, fact {fact.fact_id})"
                )
        if fact_lines:
            fact_lines.sort()  # deterministic packaging
            text = "Known campaign facts (with reliability):\n" + "\n".join(fact_lines)
            if len(text) > remaining:
                kept = []
                total = len("Known campaign facts (with reliability):\n")
                for line in fact_lines:
                    if total + len(line) + 1 > remaining:
                        break
                    kept.append(line)
                    total += len(line) + 1
                text = "Known campaign facts (with reliability):\n" + "\n".join(kept)
            add("F1", "fact", f"facts:{campaign_id}", text)

        # 5. Previous session summaries, most recent first, whatever fits.
        for index, (prev_id, summary_md) in enumerate(
            self.memory.previous_summaries(campaign_id, before_session=session_id), start=1
        ):
            add(f"P{index}", "summary", f"summary:{prev_id}",
                f"Summary of previous session {prev_id}:\n{summary_md}")

        return ContextPackage(
            campaign_id=campaign_id,
            task=task,
            sources=sources,
            sections=sections,
            budget_chars=self.budget_chars,
        )
