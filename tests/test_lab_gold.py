"""Gold-transcript layer: WER, name accuracy, mined corrections, whisper prompt."""
import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.gold import (
    compare_transcripts,
    known_names,
    records_from_gold,
    render_markdown,
    whisper_prompt,
)
from transcripts_ai.lab.records import read_records
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.schemas import AliasRecord, EntityKind, EntityRecord

CAMPAIGN = "heckuva"


def t(*lines):
    """Build a machine-format transcript from bare texts."""
    stamped = []
    for index, text in enumerate(lines):
        start = f"00:00:{index * 5:02d}.000"
        end = f"00:00:{index * 5 + 4:02d}.000"
        stamped.append(f"[{start} - {end}] - Unknown: {text}")
    return "\n\n".join(stamped) + "\n"


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "memory.sqlite")
    for name, kind in [("Jinx", EntityKind.PC), ("Ghomra", EntityKind.NPC),
                       ("Silver Spire", EntityKind.LOCATION)]:
        mem.upsert_entity(
            EntityRecord(name=name, kind=kind, campaign_id=CAMPAIGN),
            actor="human:harry",
        )
    mem.add_alias(
        AliasRecord(campaign_id=CAMPAIGN, observed="Gomra", canonical="Ghomra",
                    entity_id=None, approved_by="human:harry"),
        actor="human:harry",
    )
    yield mem
    mem.close()


class TestCompare:
    def test_identical_is_zero_wer(self):
        text = t("The party enters the Silver Spire")
        report = compare_transcripts(text, text)
        assert report.wer == 0.0 and not report.pairs

    def test_substitution_mined_as_pair(self):
        gold = t("Ghomra swings the axe")
        hyp = t("Gomra swings the axe")
        report = compare_transcripts(hyp, gold)
        assert report.substitutions == 1 and report.wer == pytest.approx(0.25)
        assert (report.pairs[0].heard, report.pairs[0].truth) == ("gomra", "ghomra")
        assert "Ghomra swings" in report.pairs[0].quote

    def test_dropped_and_invented_words(self):
        gold = t("the goblin runs away fast")
        hyp = t("the um goblin runs away")
        report = compare_transcripts(hyp, gold)
        assert report.deletions == 1      # "fast" dropped
        assert report.insertions == 1     # "um" invented
        assert report.substitutions == 0

    def test_unequal_replace_not_mined_as_pairs(self):
        gold = t("they meet Boblin Thee Seventh at dawn")
        hyp = t("they meet bob at dawn")
        report = compare_transcripts(hyp, gold)
        assert not report.pairs           # 1-vs-3 words: not a confident pair
        assert report.errors >= 2

    def test_speaker_names_are_not_compared(self):
        gold = "[00:00:00.000 - 00:00:04.000] - Sarah: hello there\n"
        hyp = "[00:00:00.000 - 00:00:04.000] - Unknown: hello there\n"
        assert compare_transcripts(hyp, gold).wer == 0.0


class TestNameScoring:
    NAMES = {"ghomra": "Ghomra", "gomra": "Ghomra", "silver spire": "Silver Spire"}

    def test_hit_and_miss_counted(self):
        gold = t("Ghomra waves", "Ghomra leaves")
        hyp = t("Ghomra waves", "Gondra leaves")
        report = compare_transcripts(hyp, gold, names=self.NAMES)
        assert (report.name_total, report.name_hits) == (2, 1)
        miss = report.name_misses[0]
        assert miss.name == "Ghomra" and miss.heard == "gondra"

    def test_multiword_name_needs_every_word(self):
        gold = t("they reach the Silver Spire tonight")
        hyp = t("they reach the silver spare tonight")
        report = compare_transcripts(hyp, gold, names=self.NAMES)
        assert (report.name_total, report.name_hits) == (1, 0)
        assert report.name_misses[0].name == "Silver Spire"

    def test_alias_spelling_in_gold_counts_for_canonical(self):
        gold = t("Gomra nods")
        hyp = t("Gomra nods")
        report = compare_transcripts(hyp, gold, names=self.NAMES)
        assert (report.name_total, report.name_hits) == (1, 1)

    def test_dropped_name_recorded_as_dropped(self):
        gold = t("then Ghomra spoke")
        hyp = t("then spoke")
        report = compare_transcripts(hyp, gold, names=self.NAMES)
        assert report.name_misses[0].heard == ""

    def test_known_names_canonical_wins_over_alias(self, memory):
        names = known_names(memory, CAMPAIGN)
        assert names["ghomra"] == "Ghomra"
        assert names["gomra"] == "Ghomra"
        assert names["silver spire"] == "Silver Spire"


class TestGoldOutputs:
    def test_records_are_human_and_valid(self):
        gold = t("Ghomra swings the axe")
        hyp = t("Gomra swings the axe")
        report = compare_transcripts(hyp, gold)
        records = records_from_gold(report, campaign_id=CAMPAIGN,
                                    session_id="2026-08-16")
        assert len(records) == 1
        record = records[0]
        record.validate()
        assert record.item_type == "transcription"
        assert record.actor.startswith("human:")
        assert (record.subject, record.choice) == ("gomra", "ghomra")
        assert record.accepted

    def test_markdown_report(self):
        gold = t("Ghomra swings the axe")
        hyp = t("Gomra swings the axe")
        report = compare_transcripts(hyp, gold, names={"ghomra": "Ghomra"})
        text = render_markdown(report, session_id="2026-08-16")
        assert "WER" in text and "Ghomra" in text and "gomra" in text


class TestWhisperPrompt:
    def test_pcs_come_first_and_all_names_present(self, memory):
        prompt = whisper_prompt(memory, CAMPAIGN)
        assert prompt.index("Jinx") < prompt.index("Ghomra")
        assert "Silver Spire" in prompt
        assert "Gomra" not in prompt.replace("Ghomra", "")   # aliases left out

    def test_truncation_never_cuts_a_name_in_half(self, memory):
        prompt = whisper_prompt(memory, CAMPAIGN, max_chars=60)
        assert len(prompt) <= 61            # body + closing period
        assert "Jinx" in prompt             # most important name survives

    def test_empty_campaign_falls_back(self, memory):
        prompt = whisper_prompt(memory, "empty-campaign")
        assert "fantasy names" in prompt


class TestGoldCli:
    def _write(self, tmp_path):
        gold = tmp_path / "gold.md"
        hyp = tmp_path / "machine.md"
        gold.write_text(t("Ghomra swings the axe", "the party rests"),
                        encoding="utf-8")
        hyp.write_text(t("Gomra swings the axe", "the party rests"),
                       encoding="utf-8")
        return gold, hyp

    def test_gold_score_end_to_end(self, tmp_path, memory, capsys):
        gold, hyp = self._write(tmp_path)
        records = tmp_path / "mined.jsonl"
        report = tmp_path / "gold.md.report.md"
        assert lab_main(["gold-score", "--gold", str(gold), "--hyp", str(hyp),
                         "--db", str(memory.db_path), "--campaign", CAMPAIGN,
                         "--session", "2026-08-16",
                         "--out-records", str(records),
                         "--report", str(report)]) == 0
        out = capsys.readouterr().out
        assert "WER" in out and "known-name accuracy" in out
        mined = read_records(records)
        assert mined and mined[0].choice == "ghomra"
        assert "Gold transcript report" in report.read_text(encoding="utf-8")

    def test_gold_score_without_db_still_scores(self, tmp_path, capsys):
        gold, hyp = self._write(tmp_path)
        assert lab_main(["gold-score", "--gold", str(gold),
                         "--hyp", str(hyp)]) == 0
        assert "WER" in capsys.readouterr().out

    def test_records_need_session(self, tmp_path, capsys):
        gold, hyp = self._write(tmp_path)
        assert lab_main(["gold-score", "--gold", str(gold), "--hyp", str(hyp),
                         "--campaign", CAMPAIGN,
                         "--out-records", str(tmp_path / "r.jsonl")]) == 2

    def test_db_and_campaign_go_together(self, tmp_path, memory):
        gold, hyp = self._write(tmp_path)
        assert lab_main(["gold-score", "--gold", str(gold), "--hyp", str(hyp),
                         "--db", str(memory.db_path)]) == 2

    def test_whisper_prompt_to_file(self, tmp_path, memory, capsys):
        out = tmp_path / "prompt.txt"
        assert lab_main(["whisper-prompt", "--db", str(memory.db_path),
                         "--campaign", CAMPAIGN, "--out", str(out)]) == 0
        assert "Jinx" in out.read_text(encoding="utf-8")
