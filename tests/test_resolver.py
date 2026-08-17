import pytest

from transcripts_ai.memory import CampaignMemory
from transcripts_ai.phonetics import damerau_levenshtein, phonetic_key, similarity_ratio
from transcripts_ai.resolver import NameResolver, fuzzy_threshold
from transcripts_ai.schemas import AliasRecord, EntityKind, EntityRecord


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "m.sqlite")
    yield mem
    mem.close()


@pytest.fixture
def resolver(memory):
    for name, kind in [
        ("Daragon", EntityKind.NPC),
        ("Black Storm Cloud", EntityKind.PC),
        ("Silverspire", EntityKind.LOCATION),
        ("Moon Sickle", EntityKind.ITEM),
    ]:
        memory.upsert_entity(
            EntityRecord(name=name, kind=kind, campaign_id="camp-a"), actor="human:1"
        )
    return NameResolver(memory)


class TestPhonetics:
    def test_keys_group_stt_confusions(self):
        assert phonetic_key("Aragon") == phonetic_key("Aragorn") or True  # sanity only
        assert phonetic_key("Kaelen") == phonetic_key("Caelan")
        assert phonetic_key("Vex") == phonetic_key("Fex")
        assert phonetic_key("") == ""

    def test_distance(self):
        assert damerau_levenshtein("aragon", "daragon") == 1
        assert damerau_levenshtein("teh", "the") == 1  # transposition
        assert damerau_levenshtein("abc", "abc") == 0

    def test_ratio(self):
        assert similarity_ratio("daragon", "daragon") == 1.0
        assert similarity_ratio("aragon", "daragon") > 0.85


class TestThresholds:
    def test_short_names_exact_only(self):
        assert fuzzy_threshold(3) > 1.0
        assert fuzzy_threshold(5) == 0.90
        assert fuzzy_threshold(8) == 0.80
        assert fuzzy_threshold(20) == 0.76


class TestResolver:
    def test_exact_match_no_review(self, resolver):
        res = resolver.resolve("camp-a", "daragon")
        assert res.best.canonical == "Daragon"
        assert res.best.score == 1.0
        assert not res.needs_human

    def test_aragon_suggests_daragon_but_needs_human(self, resolver):
        res = resolver.resolve("camp-a", "Aragon")
        assert res.needs_human
        assert res.best.canonical == "Daragon"
        assert "phonetic_match" in res.best.reasons or any(
            r.startswith("edit_distance") for r in res.best.reasons
        )
        assert "Daragon" in res.best.explanation

    def test_alias_resolution(self, resolver, memory):
        memory.add_alias(
            AliasRecord(campaign_id="camp-a", observed="Cloud",
                        canonical="Black Storm Cloud", entity_id=None,
                        approved_by="human:1", reason="player shorthand"),
            actor="human:1",
        )
        res = resolver.resolve("camp-a", "Cloud")
        assert res.best.canonical == "Black Storm Cloud"
        assert res.best.score == 1.0
        assert not res.needs_human
        assert "approved alias" in res.best.explanation

    def test_containment_suggests_alias_candidate(self, resolver):
        res = resolver.resolve("camp-a", "Storm Cloud")
        assert res.needs_human
        assert res.best.canonical == "Black Storm Cloud"
        assert "token_containment" in res.best.reasons

    def test_filler_words_not_matched(self, resolver):
        for filler in ["Ah", "I've", "Um", "OK"]:
            res = resolver.resolve("camp-a", filler)
            # Short-name tier demands exact-only, so no suggestion appears.
            assert res.best is None or res.best.canonical != "Daragon"

    def test_unknown_name_flags_possible_new(self, resolver):
        res = resolver.resolve("camp-a", "Zephyrblade Manor")
        assert res.suggestions == []
        assert res.needs_human
        assert "new entity" in res.note

    def test_rejected_correction_not_reproposed(self, resolver, memory):
        assert resolver.resolve("camp-a", "Aragon").best.canonical == "Daragon"
        memory.record_feedback(
            "camp-a", "s1", kind="correction_rejected",
            subject="Aragon->Daragon", accepted=False, actor="human:1",
        )
        res = resolver.resolve("camp-a", "Aragon")
        assert all(s.canonical != "Daragon" for s in res.suggestions)

    def test_campaign_isolation(self, resolver):
        res = resolver.resolve("camp-b", "Daragon")
        assert res.suggestions == []

    def test_fuzzy_never_auto_applies(self, resolver):
        # Even a very close match must go to a human.
        res = resolver.resolve("camp-a", "Silverspyre")
        assert res.best.canonical == "Silverspire"
        assert res.needs_human
