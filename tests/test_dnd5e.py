"""Official 5e names: data file integrity, exact mentions, near-miss catches."""
from transcripts_ai.dnd5e import official_names, scan_text


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
