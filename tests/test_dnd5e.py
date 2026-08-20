"""Official 5e names: data, scanning, and the knowledge layer in the engine."""
import pytest

from transcripts_ai.dnd5e import (
    is_registrable,
    kind_for,
    nearest_official,
    official_names,
    protected_tokens,
    scan_text,
)
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.pipeline import SessionPipeline
from transcripts_ai.schemas import EntityKind


def t(*lines):
    return "\n\n".join(f"DM: {text}" for text in lines) + "\n"


class TestData:
    def test_the_sessions_items_are_known(self):
        names = official_names()
        for name in ("Horn of Blasting", "Staff of Swarming Insects",
                     "Necklace of Prayer Beads", "Wand of Fear",
                     "Pearl of Power"):
            assert names[name] == "item"
        assert names["Insect Plague"] == "spell"
        assert names["Cure Wounds"] == "spell"
        assert names["Pelor"] == "deity"
        assert names["Aboleth"] == "monster"

    def test_reasonable_size(self):
        assert len(official_names()) > 900


class TestScan:
    def test_exact_mentions_any_casing(self):
        mentions, suspects = scan_text(t(
            "you find a necklace of prayer beads and a wand of fear",
            "he casts insect plague on the bridge",
        ))
        assert mentions["Necklace of Prayer Beads"] == 1
        assert mentions["Wand of Fear"] == 1
        assert mentions["Insect Plague"] == 1

    def test_mangled_item_is_suspected(self):
        _, suspects = scan_text(t("i raise the steph of swarming insects"))
        assert any(s.official == "Staff of Swarming Insects" for s in suspects)

    def test_exact_matches_are_not_suspects(self):
        _, suspects = scan_text(t("the horn of blasting explodes"))
        assert not [s for s in suspects if s.official == "Horn of Blasting"]

    def test_common_words_never_anchor(self):
        mentions, suspects = scan_text(t(
            "of the and the of the", "we walk to the town of the king"))
        assert not mentions and not suspects

    def test_subset_names_do_not_double_count(self):
        # "Staff of Fire" must not also fire inside longer exact windows.
        mentions, _ = scan_text(t("he holds the staff of fire high"))
        assert mentions["Staff of Fire"] == 1

    def test_speaker_names_are_not_scanned(self):
        mentions, _ = scan_text("Aboleth: hello there\n")
        assert not mentions


class TestKnowledge:
    def test_kinds(self):
        assert kind_for("Insect Plague") is EntityKind.SPELL
        assert kind_for("Aboleth") is EntityKind.CREATURE
        assert kind_for("Pelor") is EntityKind.DEITY
        assert kind_for("Necklace of Prayer Beads") is EntityKind.ITEM
        assert kind_for("Potion of Healing") is EntityKind.POTION
        assert kind_for("Dragon Scale Mail") is EntityKind.ARMOUR
        assert kind_for("Vorpal Sword") is EntityKind.WEAPON

    def test_registrable(self):
        assert is_registrable("Horn of Blasting")     # multi-word: safe
        assert is_registrable("Aboleth")              # rare single word: safe
        assert not is_registrable("Command")          # everyday word: never
        assert not is_registrable("Light")

    def test_nearest_official(self):
        near = nearest_official("steph of swarming insects")
        assert near and near.official == "Staff of Swarming Insects"
        exact = nearest_official("wand of fear")
        assert exact and exact.score == 1.0
        assert nearest_official("Ghomra") is None

    def test_protected_tokens_are_rare_rulebook_words(self):
        protected = protected_tokens()
        assert "thunderous" in protected and "aboleth" in protected
        assert "the" not in protected and "of" not in protected


class TestEngineIntegration:
    def _run(self, tmp_path, text):
        transcript = tmp_path / "s1 Mapped.md"
        transcript.write_text(text, encoding="utf-8")
        memory = CampaignMemory(tmp_path / "m.sqlite")
        report = SessionPipeline(memory).process_session_native(
            campaign_id="camp", session_id="s1",
            transcript_path=transcript, game_name="G", session_date="s1")
        return memory, report

    def test_official_mention_registers_kind_correct_entity(self, tmp_path):
        memory, report = self._run(
            tmp_path,
            "DM: you find a necklace of prayer beads in the chest.\n")
        try:
            entity = memory.find_entity("camp", "Necklace of Prayer Beads")
            assert entity is not None and entity.kind is EntityKind.ITEM
            assert entity.attributes["official_5e"] == "item"
            # closed vocabulary: nothing to ask a human about the name
            assert not [i for i in report.review_items
                        if i.subject.casefold() == "necklace of prayer beads"]
        finally:
            memory.close()

    def test_lowercase_official_loot_becomes_a_fact(self, tmp_path):
        # The classic loot rules need capitalised items; Craig transcripts
        # are lowercase. The official list closes that gap.
        memory, report = self._run(
            tmp_path,
            "DM: you find a necklace of prayer beads in the chest.\n")
        try:
            loot = [f for f in report.facts_verified
                    if f.category.value == "loot"]
            assert any("Necklace of Prayer Beads" in f.statement for f in loot)
            assert all(f.provenance.quote for f in loot)
        finally:
            memory.close()

    def test_contracted_award_is_loot(self, tmp_path):
        memory, report = self._run(
            tmp_path,
            "DM: so you'll find a necklace of prayer beads in the hoard.\n")
        try:
            assert any("Necklace of Prayer Beads" in f.statement
                       for f in report.facts_verified
                       if f.category.value == "loot")
        finally:
            memory.close()

    def test_banter_about_an_item_is_not_loot(self, tmp_path):
        # A cue elsewhere on the line must not attach to the item.
        memory, report = self._run(
            tmp_path,
            "Birch: if you use a horn of blasting in a combat and it does "
            "them 14 damage and then you get 41, like you know what i mean\n")
        try:
            assert not [f for f in report.facts_verified
                        if f.category.value == "loot"]
        finally:
            memory.close()

    def test_mere_mention_without_acquisition_is_not_loot(self, tmp_path):
        memory, report = self._run(
            tmp_path,
            "DM: a horn of blasting once destroyed this whole valley.\n")
        try:
            assert not [f for f in report.facts_verified
                        if f.category.value == "loot"]
            # ...but the term itself is still learned, kind-correct
            assert memory.find_entity("camp", "Horn of Blasting") is not None
        finally:
            memory.close()

    def test_mangled_candidate_gets_official_suggestion(self, tmp_path):
        memory, report = self._run(
            tmp_path,
            "DM: He lifts the Steph Of Swarming Insects, and the Steph Of "
            "Swarming Insects hums.\n")
        try:
            items = [i for i in report.review_items if i.item_type == "entity"]
            match = [i for i in items if "steph" in i.subject.casefold()]
            assert match, [i.subject for i in items]
            assert any(s["canonical"] == "Staff of Swarming Insects"
                       and "official 5e" in s["explanation"]
                       for s in match[0].suggestions)
        finally:
            memory.close()

    def test_panel_prompt_carries_official_hint(self, tmp_path):
        from transcripts_ai.lab.panel import build_panel_prompt
        from transcripts_ai.schemas import ReviewItem
        item = ReviewItem(campaign_id="camp", session_id="s1",
                          item_type="entity", subject="steph of swarming insects",
                          reason="r", evidence=["q"])
        prompt = build_panel_prompt(item, [])
        assert "OFFICIAL 5e REFERENCE" in prompt
        assert "Staff of Swarming Insects" in prompt

    def test_spellchecker_never_corrects_rulebook_words(self, tmp_path):
        from transcripts_ai.spellcheck import SpellChecker
        from transcripts_ai.transcript import parse_transcript
        memory = CampaignMemory(tmp_path / "m.sqlite")
        try:
            parsed = parse_transcript(
                "DM: the aboleth casts a thunderous wave\n",
                source_path="x")
            corrections = SpellChecker(memory).find_corrections(parsed, "camp")
            corrected = {c.wrong.casefold() for c in corrections}
            assert "aboleth" not in corrected
            assert "thunderous" not in corrected
        finally:
            memory.close()
