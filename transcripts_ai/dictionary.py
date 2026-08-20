"""The game dictionary: table-driven name fixes, from Discord or anywhere.

The operations behind a ``/game_dictionary`` command: look a name up,
fix a wrong spelling, undo a fix, list recent changes. Every fix made
here is a HUMAN decision made at the table — it lands as a
human-approved alias plus a feedback-ledger row, which means it is
simultaneously:

- an instant resolver upgrade (the wrong spelling auto-links from now on),
- a training record for the local model (via ``lab export-records``),
- a scorecard question (via ``lab make-bank``),
- and one fewer item in the after-game review queue.

Safety: the engine's rules hold. Fixing a spelling toward a KNOWN name
just links it. Fixing toward an unknown name creates the entity at
STRONGLY_SUPPORTED (humans said so, but pages aren't canon until the
reviewer promotes them), fully audited and reversible with ``undo`` /
``forget``. Campaign isolation is inherited from memory.

The Discord cog in ``integrations/discord_game_dictionary.py`` is a thin
render layer over this module; the same operations work from any UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .memory import CampaignMemory
from .resolver import NameResolver
from .schemas import AliasRecord, EntityKind, EntityRecord, EpistemicStatus

FIX_KINDS = {kind.value: kind for kind in
              (EntityKind.PC, EntityKind.NPC, EntityKind.LOCATION,
               EntityKind.FACTION, EntityKind.ITEM, EntityKind.CREATURE,
               EntityKind.QUEST, EntityKind.DEITY, EntityKind.LORE)}


class DictionaryError(ValueError):
    """Raised on invalid dictionary operations (safe to show the user)."""


@dataclass
class LookupResult:
    query: str
    found: bool
    canonical: str = ""
    kind: str = ""
    status: str = ""
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    suggestions: list[tuple[str, float, str]] = field(default_factory=list)


@dataclass
class FixResult:
    wrong: str
    canonical: str
    created_entity: bool
    kind: str
    message: str


class GameDictionary:
    def __init__(self, memory: CampaignMemory, campaign_id: str):
        self.memory = memory
        self.campaign_id = campaign_id
        self.resolver = NameResolver(memory)

    # -- read ---------------------------------------------------------------

    def lookup(self, name: str) -> LookupResult:
        name = name.strip()
        if not name:
            raise DictionaryError("give me a name to look up")
        entity = self.memory.find_entity(self.campaign_id, name)
        if entity is None:
            alias = self.memory.resolve_alias(self.campaign_id, name)
            if alias is not None:
                entity = self.memory.find_entity(self.campaign_id, alias.canonical)
        if entity is not None:
            aliases = sorted(
                a.observed for a in self.memory.aliases(self.campaign_id)
                if a.canonical.casefold() == entity.name.casefold())
            return LookupResult(
                query=name, found=True, canonical=entity.name,
                kind=entity.kind.value, status=entity.status.value,
                description=(entity.description or "")[:300], aliases=aliases)
        resolution = self.resolver.resolve(self.campaign_id, name)
        return LookupResult(
            query=name, found=False,
            suggestions=[(s.canonical, s.score, s.explanation)
                         for s in resolution.suggestions[:3]])

    def recent(self, limit: int = 10) -> list[dict]:
        """Latest dictionary decisions (newest first) for a 'what changed'
        view — only rows a human created."""
        rows = [r for r in self.memory.all_feedback(self.campaign_id)
                if str(r.get("actor", "")).startswith("human:")
                and r["kind"] in ("correction_accepted", "alias_added",
                                  "entity_confirmed", "correction_rejected")]
        return list(reversed(rows))[:limit]

    # -- write (always a human decision) ------------------------------------

    def fix(self, wrong: str, right: str, *, actor: str,
            kind: str = "npc", session_id: str = "game-dictionary") -> FixResult:
        """'{wrong}' should be '{right}' — the table said so.

        ``actor`` must identify the human ("human:discord:<user>"). When
        ``right`` is a known name (or an alias of one), the fix links to
        its canonical form; otherwise a new entity of ``kind`` is created
        first. Both paths record the alias and the feedback-ledger row.
        """
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right:
            raise DictionaryError("both the wrong and the right spelling are needed")
        if wrong.casefold() == right.casefold():
            raise DictionaryError("those are the same name")
        if not actor.startswith("human:"):
            raise DictionaryError("dictionary fixes are human decisions; "
                                  "actor must start with 'human:'")
        if kind not in FIX_KINDS:
            raise DictionaryError(
                f"unknown kind {kind!r}; pick one of {sorted(FIX_KINDS)}")

        entity = self.memory.find_entity(self.campaign_id, right)
        if entity is None:
            alias = self.memory.resolve_alias(self.campaign_id, right)
            if alias is not None:
                entity = self.memory.find_entity(self.campaign_id, alias.canonical)
        created = False
        if entity is None:
            entity = self.memory.upsert_entity(
                EntityRecord(name=right, kind=FIX_KINDS[kind],
                             campaign_id=self.campaign_id,
                             status=EpistemicStatus.STRONGLY_SUPPORTED,
                             description=f"Added via game dictionary by {actor}"),
                actor=actor)
            created = True
            self.memory.record_feedback(
                self.campaign_id, session_id, kind="entity_confirmed",
                subject=right, accepted=True, actor=actor)

        self.memory.add_alias(
            AliasRecord(campaign_id=self.campaign_id, observed=wrong,
                        canonical=entity.name, entity_id=entity.entity_id,
                        approved_by=actor, reason="game dictionary fix"),
            actor=actor)
        self.memory.record_feedback(
            self.campaign_id, session_id, kind="correction_accepted",
            subject=f"{wrong}->{entity.name}", accepted=True, actor=actor)
        return FixResult(
            wrong=wrong, canonical=entity.name, created_entity=created,
            kind=entity.kind.value,
            message=(f'"{wrong}" now resolves to the '
                     + ("new " if created else "")
                     + f'{entity.kind.value} "{entity.name}"'))

    def undo(self, wrong: str, *, actor: str,
             session_id: str = "game-dictionary") -> bool:
        """Remove a recorded spelling and remember NOT to re-propose it."""
        if not actor.startswith("human:"):
            raise DictionaryError("dictionary changes are human decisions")
        alias = self.memory.resolve_alias(self.campaign_id, wrong)
        removed = self.memory.remove_alias(
            self.campaign_id, wrong, actor=actor,
            reason="game dictionary undo")
        if removed and alias is not None:
            self.memory.record_feedback(
                self.campaign_id, session_id, kind="correction_rejected",
                subject=f"{wrong}->{alias.canonical}", accepted=False,
                actor=actor)
        return removed
