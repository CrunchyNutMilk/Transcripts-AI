"""Training-lab Layer 1: question banks, deterministic scoring, baseline answerer, CLI."""
import pytest

from transcripts_ai.lab.__main__ import main as lab_main
from transcripts_ai.lab.answerers import memory_answerer
from transcripts_ai.lab.records import RecordError
from transcripts_ai.lab.scorecard import (
    NOT_IN_RECORD,
    Question,
    questions_from_feedback,
    read_bank,
    read_history,
    render_markdown,
    run_scorecard,
    score_answer,
    write_bank,
)
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.schemas import (
    AliasRecord,
    ChangeType,
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    Fact,
    FactCategory,
    Provenance,
    text_sha256,
)

CAMPAIGN = "heckuva"


@pytest.fixture
def memory(tmp_path):
    mem = CampaignMemory(tmp_path / "memory.sqlite")
    yield mem
    mem.close()


def make_question(qtype="fact", question="Who guards the Silver Spire?",
                  expected=("Silas",), **overrides):
    base = dict(question_id="q1", campaign_id=CAMPAIGN, qtype=qtype,
                question=question, expected=list(expected))
    base.update(overrides)
    return Question(**base)


class TestQuestionValidation:
    def test_unknown_type_rejected(self):
        with pytest.raises(RecordError, match="unknown question type"):
            make_question(qtype="riddle").validate()

    def test_non_trick_needs_expected(self):
        with pytest.raises(RecordError, match="needs expected"):
            make_question(expected=()).validate()

    def test_trick_may_have_no_expected(self):
        make_question(qtype="trick", expected=()).validate()


class TestScoreAnswer:
    def test_expected_name_inside_sentence(self):
        q = make_question(expected=["Ghomra"])
        assert score_answer(q, "The guard is Ghomra, the goblin.")

    def test_wrong_answer_fails(self):
        q = make_question(expected=["Silas"])
        assert not score_answer(q, "It was Argon.")

    def test_empty_answer_fails(self):
        assert not score_answer(make_question(), "")

    def test_no_never_matches_inside_known(self):
        # The substring bug this scorer was rewritten to kill.
        q = make_question(qtype="trick", expected=["no"])
        assert not score_answer(q, "Ghomra is a known name in this campaign.")
        assert score_answer(q, "No.")

    def test_trick_accepts_refusals(self):
        q = make_question(qtype="trick", expected=[])
        assert score_answer(q, NOT_IN_RECORD)
        assert score_answer(q, "There is no record of that happening.")
        assert not score_answer(q, "Boblin died fighting the dragon.")

    def test_short_answer_matches_longer_expected_only_when_tight(self):
        q = make_question(expected=["Boblin Thee 7enth"])
        assert score_answer(q, "Boblin Thee")      # 2 of 3 tokens: close enough
        assert not score_answer(q, "Boblin")       # 1 of 3: too loose

    def test_articles_and_punctuation_ignored(self):
        q = make_question(expected=["the Silver Spire"])
        assert score_answer(q, "Silver Spire!")


class TestBankFiles:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "bank.jsonl"
        questions = [make_question(), make_question(question_id="q2", qtype="trick",
                                                    expected=())]
        assert write_bank(path, questions) == 2
        assert read_bank(path) == questions

    def test_bad_line_reported_with_number(self, tmp_path):
        path = tmp_path / "bank.jsonl"
        path.write_text('{"nope": true}\n', encoding="utf-8")
        with pytest.raises(RecordError, match="line 1"):
            read_bank(path)


class TestRunScorecard:
    def test_totals_by_type_and_failures(self, tmp_path):
        bank = [
            make_question(question_id="q1", expected=["Silas"]),
            make_question(question_id="q2", qtype="alias",
                          question='Which known name does "Gomra" refer to?',
                          expected=["Ghomra"]),
            make_question(question_id="q3", qtype="trick", expected=[]),
        ]
        canned = {"q1": "Silas", "q2": "Argon", "q3": NOT_IN_RECORD}
        entry = run_scorecard(bank, lambda q: canned[q.question_id],
                              answerer_name="canned")
        assert (entry.total, entry.correct) == (3, 2)
        assert entry.by_type["alias"] == {"total": 1, "correct": 0}
        assert [f["question_id"] for f in entry.failures] == ["q2"]
        report = render_markdown(entry)
        assert "2/3" in report and "Argon" in report

    def test_history_round_trip(self, tmp_path):
        from transcripts_ai.lab.scorecard import append_history
        path = tmp_path / "history.jsonl"
        assert read_history(path) == []  # missing file is an empty history
        entry = run_scorecard([make_question()], lambda q: "Silas",
                              answerer_name="canned")
        append_history(path, entry)
        history = read_history(path)
        assert len(history) == 1 and history[0]["correct"] == 1
        report = render_markdown(entry, history)
        assert "Trend" in report


class TestQuestionsFromFeedback:
    def test_human_ledger_becomes_questions(self, memory):
        memory.record_feedback(CAMPAIGN, "s1", kind="correction_accepted",
                               subject="Gomra->Ghomra", accepted=True,
                               actor="human:harry")
        memory.record_feedback(CAMPAIGN, "s1", kind="entity_misclassified",
                               subject="gonf", accepted=True, actor="human:harry")
        memory.record_feedback(CAMPAIGN, "s2", kind="correction_rejected",
                               subject="jink->link", accepted=False,
                               actor="human:harry")
        # engine rows never become questions
        memory.record_feedback(CAMPAIGN, "s2", kind="alias_added",
                               subject="Althena->Althea", accepted=True,
                               actor="engine")

        questions = questions_from_feedback(memory, CAMPAIGN)
        assert sorted(q.qtype for q in questions) == ["alias", "trick", "trick"]
        alias = next(q for q in questions if q.qtype == "alias")
        assert '"Gomra"' in alias.question and alias.expected == ["Ghomra"]
        for question in questions:
            question.validate()


class TestMemoryAnswerer:
    def _seed(self, memory):
        ghomra = memory.upsert_entity(
            EntityRecord(name="Ghomra", kind=EntityKind.NPC, campaign_id=CAMPAIGN),
            actor="human:harry",
        )
        memory.add_alias(
            AliasRecord(campaign_id=CAMPAIGN, observed="Gomra", canonical="Ghomra",
                        entity_id=ghomra.entity_id, approved_by="human:harry"),
            actor="human:harry",
        )
        statement = "Silas guards the Silver Spire"
        memory.remember_fact(
            Fact(
                statement=statement,
                category=FactCategory.STORY_EVENT,
                change_type=ChangeType.MENTIONED,
                entities=["Silas", "Silver Spire"],
                status=EpistemicStatus.STRONGLY_SUPPORTED,
                confidence=0.9,
                provenance=Provenance(
                    campaign_id=CAMPAIGN, session_id="s1",
                    source_path="Transcript Mapped/s1.md",
                    source_hash=text_sha256("src"), line_start=1, line_end=1,
                    quote=statement,
                ),
            ),
            actor="engine",
        )

    def test_alias_question_resolves(self, memory):
        self._seed(memory)
        answer = memory_answerer(memory, CAMPAIGN)(
            make_question(qtype="alias",
                          question='Which known name does "Gomra" refer to?',
                          expected=["Ghomra"]))
        assert answer == "Ghomra"

    def test_unknown_name_refused(self, memory):
        self._seed(memory)
        answer = memory_answerer(memory, CAMPAIGN)(
            make_question(qtype="trick",
                          question='Which character, place or thing is "gonf"?',
                          expected=[]))
        assert answer == NOT_IN_RECORD

    def test_fact_question_searches_memory(self, memory):
        self._seed(memory)
        answerer = memory_answerer(memory, CAMPAIGN)
        question = make_question(expected=["Silas"])
        assert score_answer(question, answerer(question))

    def test_unknown_fact_refused(self, memory):
        self._seed(memory)
        answer = memory_answerer(memory, CAMPAIGN)(
            make_question(question="Who forged the Moon Sickle?",
                          expected=["nobody"]))
        assert answer == NOT_IN_RECORD


class TestLabCli:
    def test_full_layer1_smoke(self, tmp_path, capsys):
        db = tmp_path / "memory.sqlite"
        memory = CampaignMemory(db)
        memory.record_feedback(CAMPAIGN, "s1", kind="correction_accepted",
                               subject="Gomra->Ghomra", accepted=True,
                               actor="human:harry")
        memory.record_feedback(CAMPAIGN, "s2", kind="correction_rejected",
                               subject="jink->link", accepted=False,
                               actor="human:harry")
        memory.close()

        records = tmp_path / "records.jsonl"
        bank = tmp_path / "bank.jsonl"
        history = tmp_path / "history.jsonl"
        report = tmp_path / "scorecard.md"

        assert lab_main(["export-records", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out", str(records)]) == 0
        assert lab_main(["tally", "--records", str(records)]) == 0
        assert lab_main(["split", "--records", str(records), "--frozen", "s2",
                         "--out-train", str(tmp_path / "train.jsonl"),
                         "--out-eval", str(tmp_path / "eval.jsonl")]) == 0
        assert lab_main(["make-bank", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out", str(bank)]) == 0
        assert lab_main(["score", "--db", str(db), "--campaign", CAMPAIGN,
                         "--bank", str(bank), "--history", str(history),
                         "--report", str(report)]) == 0

        assert read_history(history)  # the run was recorded
        assert "Scorecard" in report.read_text(encoding="utf-8")
        out = capsys.readouterr().out
        assert "exported 2 training record(s)" in out
        assert "frozen: s2" in out

    def test_make_bank_empty_ledger_exits_nonzero(self, tmp_path):
        db = tmp_path / "empty.sqlite"
        CampaignMemory(db).close()
        assert lab_main(["make-bank", "--db", str(db), "--campaign", CAMPAIGN,
                         "--out", str(tmp_path / "bank.jsonl")]) == 1
