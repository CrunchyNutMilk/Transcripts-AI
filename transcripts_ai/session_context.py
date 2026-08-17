"""Session context: who is playing whom, in this campaign, this session.

The player→character mapping is the strongest source for identifying PCs.
Mappings are campaign-scoped (the same Discord user may play different
characters in different campaigns) and support session-specific overrides
(a player temporarily controlling another character). Nothing here ever
falls back to another campaign's mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import EntityKind, EntityRecord, EpistemicStatus, SchemaError
from .memory import CampaignMemory


@dataclass(frozen=True)
class PlayerMapping:
    player_id: str          # Discord user id (string form)
    player_label: str       # speaker label as it appears in transcripts
    character_name: str


@dataclass
class SessionContext:
    campaign_id: str
    session_id: str
    game_name: str
    session_date: str
    dm_labels: frozenset[str]                  # speaker labels that are the DM
    mappings: list[PlayerMapping]
    overrides: list[PlayerMapping] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.campaign_id or not self.session_id:
            raise SchemaError("session context requires campaign and session ids")
        if not self.dm_labels:
            raise SchemaError("session context requires at least one DM label")

    # -- resolution ---------------------------------------------------------

    def character_for_speaker(self, speaker_label: str) -> str | None:
        """Session overrides beat default mappings; DM labels return 'DM'."""
        folded = speaker_label.strip().casefold()
        if folded in {label.casefold() for label in self.dm_labels} or folded in {
            "dm", "gm", "dungeon master"
        }:
            return "DM"
        for mapping in self.overrides:
            if mapping.player_label.casefold() == folded:
                return mapping.character_name
        for mapping in self.mappings:
            if mapping.player_label.casefold() == folded:
                return mapping.character_name
        return None

    def is_dm(self, speaker_label: str) -> bool:
        return self.character_for_speaker(speaker_label) == "DM"

    @property
    def pc_names(self) -> list[str]:
        """Current-session PCs, overrides included, mapping-authoritative."""
        seen: dict[str, None] = {}
        for mapping in [*self.overrides, *self.mappings]:
            seen.setdefault(mapping.character_name, None)
        return list(seen)

    def player_for_character(self, character_name: str) -> str | None:
        folded = character_name.casefold()
        for mapping in [*self.overrides, *self.mappings]:
            if mapping.character_name.casefold() == folded:
                return mapping.player_id
        return None

    # -- memory sync --------------------------------------------------------

    def register_pcs(self, memory: CampaignMemory, *, actor: str) -> list[EntityRecord]:
        """Ensure every mapped PC exists in campaign memory as a PC entity.

        The mapping file is human-maintained configuration, so mapped PCs are
        canon. Also records the player attribute so 'players and their
        Discord identities' is remembered per campaign.
        """
        records = []
        for mapping in [*self.mappings, *self.overrides]:
            records.append(
                memory.upsert_entity(
                    EntityRecord(
                        name=mapping.character_name,
                        kind=EntityKind.PC,
                        campaign_id=self.campaign_id,
                        status=EpistemicStatus.CONFIRMED_CANON,
                        attributes={
                            "player_id": mapping.player_id,
                            "player_label": mapping.player_label,
                        },
                    ),
                    actor=actor,
                )
            )
        return records
