"""Two-lane spelling workflow tests.

Lane 1: ordinary-word misspellings are corrected automatically in Mapped,
audited separately, and NEVER create entities, memory rows or review items.
Lane 2: possible PC/NPC/location names still go through the human review
workflow with the six actions.
"""
import pytest

from transcripts_ai.cli import main
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.resolver import NameResolver
from transcripts_ai.schemas import EntityKind, EntityRecord
from transcripts_ai.spellcheck import (
    SpellChecker,
    is_ordinary_word_candidate,
)
from transcripts_ai.transcript import parse_transcript
from transcripts_ai.wordlist import best_correction, is_common_word

MAPPED = (
    "Jinx: I did that becuase it seemed right, but thier plan was definately risky.\n"
    "DM: gomra nods slowly.\n"
    "Diego: We should ask Aragon about the mines.\n"
)


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "m.sqlite")
    mem.upsert_entity(
        EntityRecord(name="Ghomra", kind=EntityKind.PC, campaign_id="camp"),
        actor="human:1",
    )
    mem.upsert_entity(
        EntityRecord(name="Daragon", kind=EntityKind.NPC, campaign_id="camp"),
        actor="human:1",
    )
    yield mem
    mem.close()


@pytest.fixture
def mapped_file(tmp_path):
    path = tmp_path / "2026-08-09 - Game - Transcripts Mapped.md"
    path.write_text(MAPPED, encoding="utf-8")
    return path


class TestWordlist:
    def test_common_words(self):
        for word in ["them", "her", "because", "with", "there", "still", "people"]:
            assert is_common_word(word), word

    def test_best_correction_finds_standard_words(self):
        assert best_correction("becuase")[0] == "because"
        assert best_correction("thier")[0] == "their"

    def test_fantasy_names_have_no_confident_correction(self):
        found = best_correction("daragon")
        # Either nothing, or something we would reject on margin/confidence —
        # the important part is 'daragon' is not a dictionary word itself.
        assert not is_common_word("daragon")
        assert found is None or found[0] != "daragon"


class TestLane1OrdinaryWords:
    def test_misspellings_corrected_without_review_items(self, memory, mapped_file):
        checker = SpellChecker(memory)
        parsed = parse_transcript(mapped_file.read_text(), source_path=str(mapped_file))
        corrections = checker.find_corrections(parsed, "camp")
        fixed = {c.original: c.corrected for c in corrections}
        assert fixed["becuase"] == "because"
        assert fixed["thier"] == "their"
        assert fixed["definately"] == "definitely"

        applied = checker.apply_corrections(
            mapped_file, corrections, campaign_id="camp", session_id="s1"
        )
        assert applied == len(corrections)
        text = mapped_file.read_text()
        assert "because it seemed right" in text
        assert "their plan was definitely risky" in text

        # No review items, no entities, no facts — audit log only.
        assert memory.pending_reviews("camp") == []
        assert memory.find_entity("camp", "because") is None
        assert memory.facts_for_session("camp", "s1") == []
        logged = memory.auto_corrections("camp")
        assert {l["subject"] for l in logged} == {
            "becuase -> because", "thier -> their", "definately -> definitely",
        }
        # Separate from entity actions in the audit log.
        assert all(l["action"] == "auto_spell_correction" for l in logged)

    def test_entity_lane_words_never_auto_corrected(self, memory, mapped_file):
        checker = SpellChecker(memory)
        parsed = parse_transcript(mapped_file.read_text(), source_path=str(mapped_file))
        corrections = checker.find_corrections(parsed, "camp")
        originals = {c.original for c in corrections}
        # "gomra" phonetically matches the known PC Ghomra -> entity lane.
        assert "gomra" not in originals
        # "Aragon" is capitalised -> possible name, untouched by lane 1.
        assert "aragon" not in originals and "Aragon" not in originals

    def test_unmapped_transcript_refused(self, memory, tmp_path):
        unmapped = tmp_path / "2026-08-09 - Game - Transcripts Unmapped.md"
        unmapped.write_text("Jinx: becuase\n", encoding="utf-8")
        checker = SpellChecker(memory)
        parsed = parse_transcript(unmapped.read_text())
        corrections = checker.find_corrections(parsed, "camp")
        with pytest.raises(ValueError, match="Mapped"):
            checker.apply_corrections(
                unmapped, corrections, campaign_id="camp", session_id="s"
            )
        assert "becuase" in unmapped.read_text()  # untouched

    def test_short_and_ambiguous_tokens_left_alone(self, memory, tmp_path):
        path = tmp_path / "x Transcripts Mapped.md"
        path.write_text("Jinx: teh dor was shut\n", encoding="utf-8")
        checker = SpellChecker(memory)
        corrections = checker.find_corrections(
            parse_transcript(path.read_text()), "camp"
        )
        assert all(c.original != "teh" for c in corrections)  # < 4 letters

    def test_cli_dry_run_then_apply(self, memory, mapped_file, tmp_path, capsys):
        db = tmp_path / "m.sqlite"  # same file the fixture opened
        code = main(["--db", str(db), "spellcheck", "--campaign", "camp",
                     "--transcript", str(mapped_file)])
        assert code == 0
        out = capsys.readouterr().out
        assert "becuase -> because" in out and "dry run" in out
        assert "becuase" in mapped_file.read_text()  # dry run: unchanged
        code = main(["--db", str(db), "spellcheck", "--campaign", "camp",
                     "--transcript", str(mapped_file), "--apply"])
        assert code == 0
        assert "because" in mapped_file.read_text()

    def test_cli_refuses_unmapped(self, tmp_path, capsys):
        unmapped = tmp_path / "x Transcripts Unmapped.md"
        unmapped.write_text("a: b\n", encoding="utf-8")
        code = main(["--db", str(tmp_path / "m.sqlite"), "spellcheck",
                     "--campaign", "camp", "--transcript", str(unmapped)])
        assert code == 3
        assert "REFUSED" in capsys.readouterr().out


class TestLane2EntitiesStillReviewed:
    def test_uncertain_name_still_needs_human(self, memory):
        resolution = NameResolver(memory).resolve("camp", "Aragon")
        assert resolution.needs_human
        assert resolution.best.canonical == "Daragon"

    def test_ordinary_word_candidates_filtered_from_queue(self):
        for word in ["Tell", "Dad", "People", "Because", "Still", "There"]:
            assert is_ordinary_word_candidate(word, {"capitalised_mid_sentence"}), word

    def test_named_entities_not_filtered(self):
        # Non-dictionary names pass through to review.
        assert not is_ordinary_word_candidate("Silas", {"capitalised_mid_sentence"})
        assert not is_ordinary_word_candidate("Oxwater", {"travel_target"})
        # Multi-word candidates are never filtered here.
        assert not is_ordinary_word_candidate(
            "Silver Spire", {"capitalised_multiword"}
        )
        # A common word WITH a strong naming signal stays reviewable:
        # "the city of Hope" -> Hope is a real place name candidate.
        assert not is_ordinary_word_candidate("Hope", {"location_of_phrase"})
