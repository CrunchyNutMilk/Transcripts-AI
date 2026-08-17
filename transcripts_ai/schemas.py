"""Core data contracts for the transcript intelligence engine.

Everything the engine stores or exchanges between components is one of these
dataclasses. They are deliberately plain (stdlib only) so the engine has no
runtime dependencies, and every type knows how to validate itself and how to
round-trip through JSON for storage in SQLite.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class SchemaError(ValueError):
    """Raised when a payload does not satisfy a schema contract."""


# ---------------------------------------------------------------------------
# Epistemic model
# ---------------------------------------------------------------------------

class EpistemicStatus(str, Enum):
    """How trustworthy a stored statement is, strongest first.

    Only human decisions or facts reconciled against the final reviewed
    transcript may hold CONFIRMED_CANON. AI output alone can never exceed
    STRONGLY_SUPPORTED.
    """

    CONFIRMED_CANON = "confirmed_canon"
    STRONGLY_SUPPORTED = "strongly_supported"
    DM_HINT = "dm_hint"
    CHARACTER_BELIEF = "character_belief"
    PLAYER_ASSUMPTION = "player_assumption"
    UNCONFIRMED_THEORY = "unconfirmed_theory"
    TABLE_TALK = "table_talk"
    CONFLICTING = "conflicting"
    RETCONNED = "retconned"

    @property
    def rank(self) -> int:
        return _EPISTEMIC_RANK[self]

    def stronger_than(self, other: "EpistemicStatus") -> bool:
        return self.rank < other.rank


_EPISTEMIC_RANK = {status: i for i, status in enumerate(EpistemicStatus)}

# Statuses an AI component may assign on its own. Anything stronger requires a
# human decision or final-transcript reconciliation.
AI_ASSIGNABLE_STATUSES = frozenset(
    s for s in EpistemicStatus if s is not EpistemicStatus.CONFIRMED_CANON
)


class EntityKind(str, Enum):
    PC = "pc"
    NPC = "npc"
    PLAYER = "player"
    LOCATION = "location"
    FACTION = "faction"
    QUEST = "quest"
    ITEM = "item"
    WEAPON = "weapon"
    ARMOUR = "armour"
    POTION = "potion"
    SPELL = "spell"
    CREATURE = "creature"
    DEITY = "deity"
    LORE = "lore"
    ORGANISATION = "organisation"
    OTHER = "other"


class FactCategory(str, Enum):
    STORY_EVENT = "story_event"
    NPC = "npc"
    PC = "pc"
    LOCATION = "location"
    FACTION = "faction"
    QUEST = "quest"
    LOOT = "loot"
    COMBAT = "combat"
    CONDITION = "condition"
    RULING = "ruling"
    RELATIONSHIP = "relationship"
    PLAN = "plan"
    SECRET = "secret"
    DEATH_OR_STATUS = "death_or_status"
    ALIAS = "alias"
    LORE = "lore"
    DEITY = "deity"
    BESTIARY = "bestiary"
    OTHER = "other"


class ChangeType(str, Enum):
    INTRODUCED = "introduced"
    UPDATED = "updated"
    PROPOSED = "proposed"      # quest offered but not yet accepted
    ACCEPTED = "accepted"      # quest explicitly accepted
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    DISCOVERED = "discovered"
    AWARDED = "awarded"
    TRANSFERRED = "transferred"
    SPENT = "spent"
    LOST = "lost"
    IDENTIFIED = "identified"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    CONFIRMED = "confirmed"
    MENTIONED = "mentioned"


class TimeStatus(str, Enum):
    """Did it actually happen? Plans and negated actions are not events."""

    HAPPENED = "happened"
    CURRENT = "current"
    PLANNED = "planned"
    NEGATED = "negated"          # explicitly did not happen / was abandoned
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


class SpeakerMode(str, Enum):
    """Who is really speaking — the authority hierarchy for a statement.

    DM narration can confirm world facts. An NPC speaking *through* the DM
    only confirms that the NPC claims it — the NPC may be lying.
    """

    DM_NARRATION = "dm_narration"
    MECHANICAL_RESULT = "mechanical_result"
    NPC_DIALOGUE = "npc_dialogue"
    PC_DIALOGUE = "pc_dialogue"
    PLAYER_STATEMENT = "player_statement"
    TABLE_TALK = "table_talk"
    UNCLEAR = "unclear"


class Verdict(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNCERTAIN = "uncertain"


# ---------------------------------------------------------------------------
# Provenance — every remembered thing knows exactly where it came from
# ---------------------------------------------------------------------------

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class Provenance:
    """Where a statement came from, precisely enough to audit or delete it."""

    campaign_id: str
    session_id: str
    source_path: str            # logical path of the source document
    source_hash: str            # sha256 of the source document text
    line_start: int             # 1-based inclusive
    line_end: int               # 1-based inclusive
    speaker: str | None = None  # mapped speaker label, e.g. "DM" or a PC name
    quote: str = ""             # exact supporting text from the source
    extractor: str = "human"    # producer identity, e.g. "engine-extractor"
    extractor_version: str = "0"
    context_manifest_hash: str | None = None  # hash of the ContextPackage used

    def validate(self) -> None:
        if not self.campaign_id:
            raise SchemaError("provenance requires campaign_id")
        if not self.session_id:
            raise SchemaError("provenance requires session_id")
        if not self.source_path:
            raise SchemaError("provenance requires source_path")
        if not _SHA256_RE.fullmatch(self.source_hash):
            raise SchemaError("provenance source_hash must be sha256 hex")
        if self.line_start < 1 or self.line_end < self.line_start:
            raise SchemaError("provenance line span is invalid")


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A single extracted statement with evidence and epistemic standing.

    ``subject`` / ``relationship`` / ``object_`` form the structured triple
    ("Cloud" / "possesses" / "Potion of Greater Healing"); ``statement`` is
    the readable sentence. ``time_status`` separates events that happened
    from plans, hypotheticals and things that explicitly did not happen.
    """

    statement: str
    category: FactCategory
    change_type: ChangeType
    entities: list[str]
    status: EpistemicStatus
    confidence: float            # 0.0 - 1.0
    provenance: Provenance
    fact_id: str = ""
    subject: str = ""
    relationship: str = ""
    object_: str = ""
    time_status: TimeStatus = TimeStatus.UNKNOWN
    speaker_mode: SpeakerMode = SpeakerMode.UNCLEAR
    importance: str = "medium"   # low | medium | high | critical
    verdict: Verdict | None = None
    needs_review: bool = False
    review_reason: str | None = None

    IMPORTANCE_LEVELS = ("low", "medium", "high", "critical")

    def __post_init__(self) -> None:
        self.validate()
        if not self.fact_id:
            self.fact_id = stable_digest(
                {
                    "statement": self.statement,
                    "campaign": self.provenance.campaign_id,
                    "session": self.provenance.session_id,
                    "lines": [self.provenance.line_start, self.provenance.line_end],
                }
            )[:24]

    def validate(self) -> None:
        if not self.statement.strip():
            raise SchemaError("fact statement is empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaError(f"fact confidence {self.confidence} outside [0,1]")
        if self.importance not in self.IMPORTANCE_LEVELS:
            raise SchemaError(f"unknown importance {self.importance!r}")
        self.provenance.validate()

    def to_json(self) -> str:
        payload = asdict(self)
        payload["category"] = self.category.value
        payload["change_type"] = self.change_type.value
        payload["status"] = self.status.value
        payload["time_status"] = self.time_status.value
        payload["speaker_mode"] = self.speaker_mode.value
        payload["verdict"] = self.verdict.value if self.verdict else None
        return canonical_json(payload)

    @classmethod
    def from_json(cls, raw: str) -> "Fact":
        data = json.loads(raw)
        prov = Provenance(**data.pop("provenance"))
        verdict = data.pop("verdict", None)
        return cls(
            provenance=prov,
            category=FactCategory(data.pop("category")),
            change_type=ChangeType(data.pop("change_type")),
            status=EpistemicStatus(data.pop("status")),
            time_status=TimeStatus(data.pop("time_status", "unknown")),
            speaker_mode=SpeakerMode(data.pop("speaker_mode", "unclear")),
            verdict=Verdict(verdict) if verdict else None,
            **data,
        )


# ---------------------------------------------------------------------------
# Entities and aliases
# ---------------------------------------------------------------------------

@dataclass
class EntityRecord:
    """A campaign entity as remembered by the engine."""

    name: str
    kind: EntityKind
    campaign_id: str
    entity_id: str = ""
    description: str = ""
    vault_path: str | None = None
    status: EpistemicStatus = EpistemicStatus.STRONGLY_SUPPORTED
    attributes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise SchemaError("entity name is empty")
        if not self.campaign_id:
            raise SchemaError("entity requires campaign_id")
        if not self.entity_id:
            self.entity_id = stable_digest(
                {"campaign": self.campaign_id, "name": self.name.casefold(), "kind": self.kind.value}
            )[:24]


@dataclass
class AliasRecord:
    """An approved mapping from an observed name to a canonical entity."""

    campaign_id: str
    observed: str
    canonical: str
    entity_id: str | None
    approved_by: str            # "human:<id>" or "engine:<component>"
    reason: str = ""
    alias_id: str = ""

    def __post_init__(self) -> None:
        if not self.observed.strip() or not self.canonical.strip():
            raise SchemaError("alias requires observed and canonical names")
        if self.observed.strip().casefold() == self.canonical.strip().casefold():
            raise SchemaError("a name cannot alias itself")
        if not self.alias_id:
            self.alias_id = stable_digest(
                {"campaign": self.campaign_id, "observed": self.observed.casefold(),
                 "canonical": self.canonical.casefold()}
            )[:24]

    @property
    def human_approved(self) -> bool:
        return self.approved_by.startswith("human:")


# ---------------------------------------------------------------------------
# Review queue
# ---------------------------------------------------------------------------

class ReviewAction(str, Enum):
    CORRECT = "correct"
    ALIAS = "alias"
    NEW = "new"
    SAVE_FOR_REVIEW = "save_for_review"
    LET_AI_PICK = "let_ai_pick"
    DONT_KNOW_YET = "dont_know_yet"


@dataclass
class ReviewItem:
    """Something the engine is not confident enough to act on alone."""

    campaign_id: str
    session_id: str
    item_type: str               # "spelling" | "entity" | "fact" | "summary_claim"
    subject: str                 # the text under review
    reason: str                  # plain-English why this needs a human
    evidence: list[str]          # supporting quotes shown to the reviewer
    suggestions: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    item_id: str = ""
    resolved: bool = False
    resolution_action: ReviewAction | None = None
    resolution_detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise SchemaError("review confidence outside [0,1]")
        if not self.item_id:
            self.item_id = stable_digest(
                {"campaign": self.campaign_id, "session": self.session_id,
                 "type": self.item_type, "subject": self.subject}
            )[:24]


# ---------------------------------------------------------------------------
# Context packages (retrieval output, also the anti-hallucination boundary)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextSource:
    source_id: str      # short handle used in prompts, e.g. "S1"
    kind: str           # "transcript" | "summary" | "fact" | "vault_page" | "alias"
    path: str           # logical path or record id
    content_hash: str   # sha256 of the included text
    chars: int

    def validate(self) -> None:
        if not _SHA256_RE.fullmatch(self.content_hash):
            raise SchemaError("context source hash must be sha256 hex")


@dataclass
class ContextPackage:
    """Exactly what a provider call was shown, recorded for validation."""

    campaign_id: str
    task: str                            # role name, e.g. "extractor"
    sources: list[ContextSource]
    sections: dict[str, str]             # source_id -> included text
    budget_chars: int
    manifest_hash: str = ""

    def __post_init__(self) -> None:
        ids = [s.source_id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise SchemaError("duplicate source ids in context package")
        for source in self.sources:
            source.validate()
            if source.source_id not in self.sections:
                raise SchemaError(f"source {source.source_id} has no section text")
        total = sum(len(t) for t in self.sections.values())
        if total > self.budget_chars:
            raise SchemaError(
                f"context package exceeds budget: {total} > {self.budget_chars}"
            )
        if not self.manifest_hash:
            self.manifest_hash = stable_digest(
                [[s.source_id, s.kind, s.path, s.content_hash] for s in self.sources]
            )

    def combined_text(self) -> str:
        return "\n".join(self.sections[s.source_id] for s in self.sources)

    def contains_quote(self, quote: str) -> bool:
        """Deterministic evidence check: the quote must appear in a source."""
        needle = _normalise_for_quote_match(quote)
        if not needle:
            return False
        return needle in _normalise_for_quote_match(self.combined_text())


_WS_RE = re.compile(r"\s+")


def _normalise_for_quote_match(text: str) -> str:
    # Case- and whitespace-insensitive so line wrapping can't fail a real quote,
    # but nothing fuzzier than that: the words themselves must match exactly.
    return _WS_RE.sub(" ", text).strip().casefold()


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

@dataclass
class SummarySection:
    title: str
    confirmed: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)


@dataclass
class SessionSummary:
    campaign_id: str
    session_id: str
    game_name: str
    session_date: str
    party: list[str]
    sections: list[SummarySection]
    manifest_hash: str
    generator: str = "engine-summarizer"
    generator_version: str = "1"

    def section(self, title: str) -> SummarySection | None:
        for sec in self.sections:
            if sec.title == title:
                return sec
        return None


# ---------------------------------------------------------------------------
# Vault update planning (plan-only; the engine never writes vault files)
# ---------------------------------------------------------------------------

@dataclass
class VaultUpdatePlan:
    campaign_id: str
    session_id: str
    target_path: str
    reason: str
    fact_ids: list[str]
    proposed_block: str
    requires_human: bool = True
