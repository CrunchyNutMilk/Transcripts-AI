"""The game dictionary: lookup, fix, undo — every fix a human decision."""
import pytest

from transcripts_ai.dictionary import DictionaryError, GameDictionary
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.resolver import BandAction, NameResolver
from transcripts_ai.schemas import EntityKind, EntityRecord

CAMPAIGN = "heckuva"
ACTOR = "human:discord:crunchynutmilk"


@pytest.fixture
def dictionary(tmp_path):
    memory = CampaignMemory(tmp_path / "m.sqlite")
    memory.upsert_entity(
        EntityRecord(name="Vel Nadar", kind=EntityKind.NPC,
                     campaign_id=CAMPAIGN,
                     description="Major sealed threat"),
        actor="human:harry")
    yield GameDictionary(memory, CAMPAIGN)
    memory.close()


class TestLookup:
    def test_known_entity(self, dictionary):
        result = dictionary.lookup("Vel Nadar")
        assert result.found and result.kind == "npc"
        assert "sealed threat" in result.description

    def test_alias_resolves_to_canonical(self, dictionary):
        dictionary.fix("Valadar", "Vel Nadar", actor=ACTOR)
        result = dictionary.lookup("Valadar")
        assert result.found and result.canonical == "Vel Nadar"
        assert "Valadar" in result.aliases

    def test_unknown_offers_suggestions(self, dictionary):
        result = dictionary.lookup("Vel Nadur")
        assert not result.found
        assert result.suggestions
        assert result.suggestions[0][0] == "Vel Nadar"

    def test_empty_query_rejected(self, dictionary):
        with pytest.raises(DictionaryError):
            dictionary.lookup("  ")


class TestFix:
    def test_fix_to_known_name_links_and_learns(self, dictionary):
        result = dictionary.fix("Valadar", "Vel Nadar", actor=ACTOR)
        assert not result.created_entity
        assert result.canonical == "Vel Nadar"
        # resolver upgrade is immediate
        best = NameResolver(dictionary.memory).resolve(CAMPAIGN, "Valadar").best
        assert best.canonical == "Vel Nadar"
        assert best.band is BandAction.AUTO_LINK
        # and it is a training-grade human decision in the ledger
        rows = dictionary.memory.all_feedback(CAMPAIGN)
        assert any(r["kind"] == "correction_accepted"
                   and r["subject"] == "Valadar->Vel Nadar"
                   and r["actor"] == ACTOR for r in rows)

    def test_fix_via_existing_alias_lands_on_canonical(self, dictionary):
        dictionary.fix("Valadar", "Vel Nadar", actor=ACTOR)
        result = dictionary.fix("Veladar", "Valadar", actor=ACTOR)
        assert result.canonical == "Vel Nadar"    # chains to the real name

    def test_fix_to_new_name_creates_entity(self, dictionary):
        result = dictionary.fix("latone", "Lathone", actor=ACTOR, kind="npc")
        assert result.created_entity
        entity = dictionary.memory.find_entity(CAMPAIGN, "Lathone")
        assert entity is not None and entity.kind is EntityKind.NPC
        assert entity.status.value != "confirmed_canon"   # humans promote later

    def test_guards(self, dictionary):
        with pytest.raises(DictionaryError, match="same name"):
            dictionary.fix("Ghomra", "ghomra", actor=ACTOR)
        with pytest.raises(DictionaryError, match="human"):
            dictionary.fix("a", "b", actor="engine")
        with pytest.raises(DictionaryError, match="kind"):
            dictionary.fix("a", "b", actor=ACTOR, kind="spaceship")


class TestUndo:
    def test_undo_removes_and_blocks_reproposal(self, dictionary):
        dictionary.fix("Valadar", "Vel Nadar", actor=ACTOR)
        assert dictionary.undo("Valadar", actor=ACTOR)
        assert dictionary.memory.resolve_alias(CAMPAIGN, "Valadar") is None
        assert dictionary.memory.was_rejected_before(
            CAMPAIGN, "correction_rejected", "Valadar->Vel Nadar")
        # the resolver will not re-surface the rejected pairing
        resolution = NameResolver(dictionary.memory).resolve(CAMPAIGN, "Valadar")
        assert all(s.canonical != "Vel Nadar" for s in resolution.suggestions)

    def test_undo_unknown_is_false(self, dictionary):
        assert not dictionary.undo("never-recorded", actor=ACTOR)


class TestRecent:
    def test_recent_lists_human_changes_newest_first(self, dictionary):
        dictionary.fix("Valadar", "Vel Nadar", actor=ACTOR)
        dictionary.fix("Zigshul", "Zig Shul", actor=ACTOR, kind="npc")
        rows = dictionary.recent(limit=5)
        assert rows
        assert rows[0]["subject"] in ("Zigshul->Zig Shul", "Zig Shul")
        assert all(r["actor"].startswith("human:") for r in rows)
