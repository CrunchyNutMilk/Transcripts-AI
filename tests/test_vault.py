"""Vault → memory sync: frontmatter parsing, page classification, seeding."""
import pytest

from transcripts_ai.cli import main as engine_main
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.schemas import EntityKind
from transcripts_ai.vault import parse_frontmatter, scan_vault, sync_vault

CAMPAIGN = "Heckuva Side Quest"


def build_vault(root):
    pcs = root / "03 - PCs" / "Kyrenia"
    pcs.mkdir(parents=True)
    (pcs / "Kyrenia.md").write_text(
        "---\n"
        "title: Kyrenia\n"
        "aliases:\n  - Kyrenia\n  - Althea\n  - Althena\n"
        "type: pc\n"
        f"campaign: {CAMPAIGN}\n"
        "status: active\n"
        "---\n\n# Kyrenia\n", encoding="utf-8")
    (root / "03 - PCs" / "PCs.md").write_text("---\ntype: folder-note\n---\n",
                                              encoding="utf-8")
    bbeg = root / "04 - NPCs & Locations" / "BBEG"
    bbeg.mkdir(parents=True)
    (bbeg / "NPC - Vel Nadar.md").write_text(
        "---\n"
        'title: "NPC - Vel Nadar"\n'
        "aliases:\n  - Vel Nadar\n  - Bel Nadar\n  - Val Nadar\n  - Vel'Nadar\n"
        "type: npc\n"
        f"campaign: {CAMPAIGN}\n"
        "status: active-threat\n"
        "notes: Major sealed threat\n"
        "---\n", encoding="utf-8")
    gods = root / "04 - NPCs & Locations" / "Gods"
    gods.mkdir(parents=True)
    (gods / "God - Earth Mother.md").write_text(
        f"---\ntitle: God - Earth Mother\ncampaign: {CAMPAIGN}\n---\n",
        encoding="utf-8")
    (gods / "Gods Directory.base").write_text("{}", encoding="utf-8")
    loot = root / "07 - Loot"
    loot.mkdir(parents=True)
    (loot / "2025-11-09 - Session Loot.md").write_text(
        "---\ntype: loot-tracker\n---\n", encoding="utf-8")
    other = root / "04 - NPCs & Locations" / "Allies"
    other.mkdir(parents=True)
    (other / "NPC - Visitor.md").write_text(
        "---\ntitle: NPC - Visitor\ntype: npc\ncampaign: Other Game\n---\n",
        encoding="utf-8")
    # folder-note convention: sub-pages of an entity folder are not entities
    (pcs / "Abilities.md").write_text(
        f"---\ntitle: Kyrenia Abilities\ntype: pc\ncampaign: {CAMPAIGN}\n---\n",
        encoding="utf-8")
    # duplicate name in a second section: the richer page (more aliases) wins
    bestiary = root / "09 - Bestiary"
    bestiary.mkdir(parents=True)
    (bestiary / "Vel Nadar.md").write_text(
        f"---\ntitle: Vel Nadar\ntype: bestiary\ncampaign: {CAMPAIGN}\n---\n",
        encoding="utf-8")
    return root


class TestFrontmatter:
    def test_scalars_lists_and_quotes(self):
        data = parse_frontmatter(
            '---\ntitle: "NPC - X"\naliases:\n  - A\n  - "B"\nlevel: 6\n---\nbody')
        assert data["title"] == "NPC - X"
        assert data["aliases"] == ["A", "B"]
        assert data["level"] == "6"

    def test_no_frontmatter_is_empty(self):
        assert parse_frontmatter("# Just a page\n") == {}


class TestScan:
    def test_entities_kinds_aliases(self, tmp_path):
        entities, skipped = scan_vault(build_vault(tmp_path),
                                       campaign_id=CAMPAIGN)
        by_name = {e.name: e for e in entities}
        assert by_name["Kyrenia"].kind is EntityKind.PC
        assert by_name["Kyrenia"].aliases == ["Althea", "Althena"]
        assert by_name["Vel Nadar"].kind is EntityKind.NPC
        assert "Vel'Nadar" in by_name["Vel Nadar"].aliases
        assert by_name["Earth Mother"].kind is EntityKind.DEITY
        assert "Visitor" not in by_name          # other campaign
        assert any("other campaign" in s for s in skipped)

    def test_trackers_and_folder_notes_skipped(self, tmp_path):
        entities, _ = scan_vault(build_vault(tmp_path), campaign_id=CAMPAIGN)
        names = {e.name for e in entities}
        assert not any("Loot" in n or n == "PCs" for n in names)

    def test_entity_subpages_are_not_entities(self, tmp_path):
        entities, _ = scan_vault(build_vault(tmp_path), campaign_id=CAMPAIGN)
        names = {e.name for e in entities}
        assert "Kyrenia Abilities" not in names
        assert "Kyrenia" in names

    def test_duplicate_names_keep_the_richer_page(self, tmp_path):
        entities, skipped = scan_vault(build_vault(tmp_path),
                                       campaign_id=CAMPAIGN)
        vel = [e for e in entities if e.name == "Vel Nadar"]
        assert len(vel) == 1
        assert vel[0].kind is EntityKind.NPC     # 4-alias BBEG page wins
        assert any("duplicate name" in s for s in skipped)

    def test_loot_section_not_scanned_by_default(self, tmp_path):
        entities, _ = scan_vault(build_vault(tmp_path), campaign_id=CAMPAIGN)
        assert all("07 - Loot" not in e.path for e in entities)


class TestSync:
    def test_seeds_entities_and_human_approved_aliases(self, tmp_path):
        entities, _ = scan_vault(build_vault(tmp_path), campaign_id=CAMPAIGN)
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            entity_count, alias_count = sync_vault(memory, CAMPAIGN, entities)
            assert entity_count == 3 and alias_count == 5
            vel = memory.find_entity(CAMPAIGN, "Vel Nadar")
            assert vel is not None and vel.kind is EntityKind.NPC
            assert vel.attributes["vault_status"] == "active-threat"
            alias = memory.resolve_alias(CAMPAIGN, "Vel'Nadar")
            assert alias is not None and alias.canonical == "Vel Nadar"
            assert alias.human_approved     # the vault is human-curated
            # the resolver now auto-links historic mishearings on sight
            from transcripts_ai.resolver import BandAction, NameResolver
            best = NameResolver(memory).resolve(CAMPAIGN, "Bel Nadar").best
            assert best.canonical == "Vel Nadar"
            assert best.band is BandAction.AUTO_LINK
        finally:
            memory.close()

    def test_sync_is_idempotent(self, tmp_path):
        entities, _ = scan_vault(build_vault(tmp_path), campaign_id=CAMPAIGN)
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            sync_vault(memory, CAMPAIGN, entities)
            sync_vault(memory, CAMPAIGN, entities)
            assert len(memory.entities(CAMPAIGN)) == 3
            assert len(memory.aliases(CAMPAIGN)) == 5
        finally:
            memory.close()


class TestMultiCampaign:
    def test_two_campaigns_in_one_db_never_cross(self, tmp_path):
        """The engine is campaign-generic: knowledge is data keyed by
        campaign id, and nothing leaks between games."""
        vault_a = build_vault(tmp_path / "vault_a")
        vault_b = tmp_path / "vault_b"
        npc_dir = vault_b / "04 - NPCs & Locations" / "BBEG"
        npc_dir.mkdir(parents=True)
        (npc_dir / "NPC - Strahd.md").write_text(
            "---\ntitle: NPC - Strahd\ntype: npc\ncampaign: Curse of Strahd\n"
            "aliases:\n  - Strahd\n  - The Devil Strahd\n---\n",
            encoding="utf-8")
        memory = CampaignMemory(tmp_path / "shared.sqlite")
        try:
            entities_a, _ = scan_vault(vault_a, campaign_id=CAMPAIGN)
            sync_vault(memory, CAMPAIGN, entities_a)
            entities_b, _ = scan_vault(vault_b, campaign_id="Curse of Strahd")
            sync_vault(memory, "Curse of Strahd", entities_b)

            assert memory.find_entity("Curse of Strahd", "Vel Nadar") is None
            assert memory.find_entity(CAMPAIGN, "Strahd") is None
            assert memory.resolve_alias(CAMPAIGN, "The Devil Strahd") is None
            from transcripts_ai.resolver import NameResolver
            best = NameResolver(memory).resolve("Curse of Strahd",
                                                "Bel Nadar").best
            assert best is None or best.canonical != "Vel Nadar"
        finally:
            memory.close()


class TestCli:
    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        build_vault(tmp_path / "vault")
        db = tmp_path / "m.sqlite"
        assert engine_main(["--db", str(db), "vault-sync",
                            "--vault", str(tmp_path / "vault"),
                            "--campaign", CAMPAIGN, "--dry-run"]) == 0
        out = capsys.readouterr().out
        assert "Vel Nadar" in out and "dry run" in out
        memory = CampaignMemory(db)
        try:
            assert memory.entities(CAMPAIGN) == []
        finally:
            memory.close()

    def test_sync_end_to_end(self, tmp_path, capsys):
        build_vault(tmp_path / "vault")
        db = tmp_path / "m.sqlite"
        assert engine_main(["--db", str(db), "vault-sync",
                            "--vault", str(tmp_path / "vault"),
                            "--campaign", CAMPAIGN]) == 0
        assert "synced 3" in capsys.readouterr().out

    def test_wrong_root_fails_clearly(self, tmp_path, capsys):
        (tmp_path / "empty").mkdir()
        assert engine_main(["--db", str(tmp_path / "m.sqlite"), "vault-sync",
                            "--vault", str(tmp_path / "empty"),
                            "--campaign", CAMPAIGN]) == 1