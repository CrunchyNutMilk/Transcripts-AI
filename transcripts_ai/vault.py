"""Vault → memory sync: the engine is born knowing the campaign wiki.

The Obsidian vault is the human-curated authority on who and what exists
in a campaign. This module reads its entity pages — PCs, NPCs, factions,
gods, places, quests, creatures — and seeds campaign memory with their
canonical names, kinds, and (crucially) their ``aliases`` lists, which
already record every mishearing the table has ever corrected
("Vel Nadar: Bel Nadar, Val Nadar, Vel'Nadar"). Those become
human-approved aliases, so the resolver auto-links them on sight.

Read-only on the vault, always: pages are parsed, never written.
Tracker/index pages (session loot trackers, directory notes, rosters,
transcripts, summaries) are skipped by their frontmatter ``type`` or
shape. Everything imported carries its vault path as provenance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .memory import CampaignMemory
from .schemas import AliasRecord, EntityKind, EntityRecord, EpistemicStatus

# frontmatter `type` values that are content pages, not entities
SKIP_TYPES = frozenset({
    "loot-tracker", "transcript-mapped", "transcript", "party-roster",
    "summary", "session-summary", "map-note", "campaign", "folder-note",
    "index", "directory",
})

TYPE_TO_KIND = {
    "pc": EntityKind.PC,
    "npc": EntityKind.NPC,
    "npc-group": EntityKind.NPC,
    "faction": EntityKind.FACTION,
    "god": EntityKind.DEITY,
    "deity": EntityKind.DEITY,
    "location": EntityKind.LOCATION,
    "settlement": EntityKind.LOCATION,
    "region": EntityKind.LOCATION,
    "place": EntityKind.LOCATION,
    "plane": EntityKind.LOCATION,
    "quest": EntityKind.QUEST,
    "item": EntityKind.ITEM,
    "weapon": EntityKind.WEAPON,
    "creature": EntityKind.CREATURE,
    "monster": EntityKind.CREATURE,
    "bestiary": EntityKind.CREATURE,
    "lore": EntityKind.LORE,
}

TITLE_PREFIXES = [
    ("NPC Group - ", EntityKind.NPC),
    ("NPC - ", EntityKind.NPC),
    ("Faction - ", EntityKind.FACTION),
    ("God - ", EntityKind.DEITY),
    ("Quest - ", EntityKind.QUEST),
    ("Bestiary - ", EntityKind.CREATURE),
    ("Location - ", EntityKind.LOCATION),
    ("Settlement - ", EntityKind.LOCATION),
    ("Item - ", EntityKind.ITEM),
]

FOLDER_HINTS = [
    ("03 - PCs", EntityKind.PC),
    ("Gods", EntityKind.DEITY),
    ("Factions", EntityKind.FACTION),
    ("Places", EntityKind.LOCATION),
    ("Settlements", EntityKind.LOCATION),
    ("Regions", EntityKind.LOCATION),
    ("Plane", EntityKind.LOCATION),
    ("Unknown Locations", EntityKind.LOCATION),
    ("09 - Bestiary", EntityKind.CREATURE),
    ("06 - Quests", EntityKind.QUEST),
    ("05 - Lore", EntityKind.LORE),
    ("Allies", EntityKind.NPC),
    ("BBEG", EntityKind.NPC),
    ("Magic Beings", EntityKind.NPC),
    ("Unknown NPC", EntityKind.NPC),
]

DEFAULT_SECTIONS = ("03 - PCs", "04 - NPCs & Locations", "06 - Quests",
                    "09 - Bestiary")

_LIST_ITEM = re.compile(r"^\s+-\s+(.*)$")


def parse_frontmatter(text: str) -> dict:
    """Minimal YAML-subset frontmatter reader: scalars and string lists.

    Deliberately tolerant — anything it cannot understand is skipped, not
    fatal, because vault pages are hand-maintained.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    data: dict = {}
    current_list: str | None = None
    for line in lines[1:]:
        if line.strip() == "---":
            break
        item = _LIST_ITEM.match(line)
        if item and current_list is not None:
            data[current_list].append(item.group(1).strip().strip("'\""))
            continue
        if ":" in line and not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            if value:
                data[key] = value
                current_list = None
            else:
                data[key] = []
                current_list = key
    return data


@dataclass
class VaultEntity:
    name: str
    kind: EntityKind
    aliases: list[str] = field(default_factory=list)
    status_note: str = ""
    notes: str = ""
    campaign: str = ""
    path: str = ""


def _clean_title(raw: str) -> tuple[str, EntityKind | None]:
    for prefix, kind in TITLE_PREFIXES:
        if raw.startswith(prefix):
            return raw[len(prefix):].strip(), kind
    return raw.strip(), None


def _kind_for(front: dict, title_kind: EntityKind | None,
              path: Path) -> EntityKind | None:
    front_type = str(front.get("type", "")).strip().lower()
    if front_type in TYPE_TO_KIND:
        return TYPE_TO_KIND[front_type]
    if title_kind is not None:
        return title_kind
    parts = {p for p in path.parts}
    for folder, kind in FOLDER_HINTS:
        if folder in parts:
            return kind
    return None


def _is_structural(stem: str, path: Path) -> bool:
    if stem.endswith("Directory") or stem in ("PCs", "Summaries", "Mapping"):
        return True
    if re.match(r"^\d\d - ", stem):
        return True
    return stem == path.parent.name        # folder note ("Kyrenia/Kyrenia.md"
                                           # is the PC page — see caller)


def scan_vault(root: str | Path, *, campaign_id: str,
               sections: tuple[str, ...] = DEFAULT_SECTIONS
               ) -> tuple[list[VaultEntity], list[str]]:
    """(entities, skipped-page notes). Read-only."""
    root = Path(root)
    entities: list[VaultEntity] = []
    skipped: list[str] = []
    for section in sections:
        base = root / section
        if not base.is_dir():
            skipped.append(f"section missing: {section}")
            continue
        # Obsidian folder-note convention: a folder whose same-named page
        # is itself entity-typed IS that entity ("Kyrenia/Kyrenia.md",
        # type: pc); its other pages (Abilities, History, Personal
        # Loot, ...) are sub-pages, never entities. Same-named pages that
        # are NOT entity-typed ("Factions/Factions.md") are section notes
        # and must not swallow their siblings.
        entity_folders = set()
        for note in base.rglob("*.md"):
            if note.stem != note.parent.name:
                continue
            note_type = str(parse_frontmatter(
                note.read_text(encoding="utf-8", errors="replace")
            ).get("type", "")).strip().lower()
            if note_type in TYPE_TO_KIND:
                entity_folders.add(note.parent)
        for path in sorted(base.rglob("*.md")):
            rel = str(path.relative_to(root))
            stem = path.stem
            if path.parent in entity_folders and stem != path.parent.name:
                continue
            front = parse_frontmatter(path.read_text(encoding="utf-8",
                                                     errors="replace"))
            front_type = str(front.get("type", "")).strip().lower()
            if front_type in SKIP_TYPES:
                continue
            # A PC page lives at "03 - PCs/<Name>/<Name>.md": same-stem-as-
            # folder is the page itself there, but a structural note at the
            # section root ("PCs.md"). Real entity pages always carry a type
            # or a recognised title prefix; structural pages carry neither.
            title, title_kind = _clean_title(str(front.get("title", stem)))
            kind = _kind_for(front, title_kind, path.relative_to(root))
            if kind is None or (_is_structural(stem, path)
                                and front_type not in TYPE_TO_KIND):
                if kind is not None or front_type:
                    skipped.append(f"structural/unknown: {rel}")
                continue
            page_campaign = str(front.get("campaign", "")).strip()
            if page_campaign and page_campaign != campaign_id:
                skipped.append(f"other campaign ({page_campaign}): {rel}")
                continue
            raw_aliases = front.get("aliases")
            aliases = [a for a in raw_aliases
                       if isinstance(a, str) and a.strip()] \
                if isinstance(raw_aliases, list) else []
            entities.append(VaultEntity(
                name=title,
                kind=kind,
                aliases=[a for a in aliases
                         if a.casefold() != title.casefold()],
                status_note=str(front.get("status", "")).strip(),
                notes=str(front.get("notes", "")).strip(),
                campaign=page_campaign or campaign_id,
                path=rel,
            ))
    # The same name can page in two places (an NPC and its bestiary
    # statblock); one entity per name, the richest page wins.
    by_name: dict[str, VaultEntity] = {}
    for entity in entities:
        key = entity.name.casefold()
        existing = by_name.get(key)
        if existing is None or len(entity.aliases) > len(existing.aliases):
            if existing is not None:
                skipped.append(f"duplicate name, kept richer page: "
                               f"{existing.path if existing else ''}")
            by_name[key] = entity
        else:
            skipped.append(f"duplicate name, kept richer page: {entity.path}")
    return list(by_name.values()), skipped


def sync_vault(memory: CampaignMemory, campaign_id: str,
               entities: list[VaultEntity],
               *, actor: str = "human:vault") -> tuple[int, int]:
    """Seed memory with vault canon. Returns (entities, aliases) written.

    The vault is human-curated, so pages land as STRONGLY_SUPPORTED
    entities under a human-lineage actor and their alias lists become
    human-approved aliases (the resolver then auto-links them). The
    engine still never edits the vault — this is one-way, vault → memory.
    """
    entity_count = alias_count = 0
    for vault_entity in entities:
        record = memory.upsert_entity(
            EntityRecord(
                name=vault_entity.name,
                kind=vault_entity.kind,
                campaign_id=campaign_id,
                status=EpistemicStatus.STRONGLY_SUPPORTED,
                description=(vault_entity.notes
                             or f"Vault page: {vault_entity.path}"),
                attributes={
                    "vault_path": vault_entity.path,
                    **({"vault_status": vault_entity.status_note}
                       if vault_entity.status_note else {}),
                },
            ),
            actor=actor,
        )
        entity_count += 1
        for alias in vault_entity.aliases:
            memory.add_alias(
                AliasRecord(
                    campaign_id=campaign_id,
                    observed=alias,
                    canonical=vault_entity.name,
                    entity_id=record.entity_id,
                    approved_by=actor,
                    reason=f"vault aliases list ({vault_entity.path})",
                ),
                actor=actor,
            )
            alias_count += 1
    return entity_count, alias_count
