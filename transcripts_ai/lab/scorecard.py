"""Question banks and the nightly scorecard.

The honesty layer of docs/TRAINING.md: a fixed set of questions with known
answers, scored the same way every night, appended to a history file so
"getting better" is a curve rather than a feeling.

Three question types, because they measure different failure modes:

- ``fact``  — a real question with a real answer ("Who gave the party the
  Moon Sickle?"). Measures recall/retrieval.
- ``alias`` — a spelling variant that must resolve to a canonical name
  ("Which known name does 'Gomra' refer to?"). Measures name resolution.
- ``trick`` — unanswerable or false-premise ("When did Boblin die?" — he
  didn't). The correct answer is *not in the record*; measures the
  no-invention property, which is the one stock LLMs fail worst.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..memory import CampaignMemory
from .records import RecordError

QUESTION_TYPES = ("fact", "alias", "trick")
NOT_IN_RECORD = "not in the record"

# Phrases that count as a correct refusal on trick questions.
_REFUSAL_PATTERNS = (
    "not in the record", "no record", "not mentioned", "not stated",
    "no such", "never happened", "did not happen", "didn't happen",
    "does not exist", "doesn't exist", "unknown", "i don't know",
    "cannot answer", "can't answer", "no answer", "not an entity",
    "no evidence",
)


@dataclass
class Question:
    question_id: str
    campaign_id: str
    qtype: str                       # fact | alias | trick
    question: str
    expected: list[str]              # acceptable answers (any-match)
    session_id: str = ""             # session the answer comes from, if any
    notes: str = ""

    def validate(self) -> None:
        if self.qtype not in QUESTION_TYPES:
            raise RecordError(f"unknown question type {self.qtype!r}")
        if self.qtype != "trick" and not self.expected:
            raise RecordError(f"{self.question_id}: non-trick question needs expected answers")


def write_bank(path: str | Path, questions: Iterable[Question]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for question in questions:
            question.validate()
            f.write(json.dumps(asdict(question), sort_keys=True, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_bank(path: str | Path) -> list[Question]:
    questions: list[Question] = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                question = Question(**json.loads(line))
                question.validate()
            except (json.JSONDecodeError, TypeError, RecordError) as exc:
                raise RecordError(f"{path} line {number}: {exc}") from exc
            questions.append(question)
    return questions


_NORMALISE = re.compile(r"[^a-z0-9' ]+")


def _normalise(text: str) -> str:
    text = _NORMALISE.sub(" ", text.casefold())
    return " ".join(w for w in text.split() if w not in {"the", "a", "an"})


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_normalise(text).split())


def _phrase_matches(phrase: str, answer_tokens: tuple[str, ...]) -> bool:
    """Word-level containment: every token of the phrase appears in the
    answer (so 'no' never matches inside 'known')."""
    phrase_tokens = _tokens(phrase)
    return bool(phrase_tokens) and set(phrase_tokens) <= set(answer_tokens)


def score_answer(question: Question, answer: str) -> bool:
    """Deterministic scoring — no judge model, so the score can't drift."""
    answer_tokens = _tokens(answer)
    if question.qtype == "trick":
        return any(
            _phrase_matches(phrase, answer_tokens)
            for phrase in (*_REFUSAL_PATTERNS, *question.expected)
        )
    if not answer_tokens:
        return False
    for expected in question.expected:
        expected_tokens = _tokens(expected)
        if not expected_tokens:
            continue
        # Expected inside answer ("Ghomra" within a full sentence), or the
        # answer is a tight subset of a longer expected form.
        if set(expected_tokens) <= set(answer_tokens):
            return True
        if set(answer_tokens) <= set(expected_tokens) and \
                len(answer_tokens) * 2 >= len(expected_tokens):
            return True
    return False


# ---------------------------------------------------------------------------
# Running a scorecard
# ---------------------------------------------------------------------------

Answerer = Callable[[Question], str]


@dataclass
class ScorecardEntry:
    answerer: str
    campaign_id: str
    run_at: float
    total: int
    correct: int
    by_type: dict[str, dict[str, int]]     # qtype -> {total, correct}
    failures: list[dict[str, str]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0


def run_scorecard(
    questions: list[Question], answerer: Answerer, *, answerer_name: str,
    max_failures_kept: int = 50,
) -> ScorecardEntry:
    totals: Counter[str] = Counter()
    corrects: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    campaign = questions[0].campaign_id if questions else ""
    for question in questions:
        answer = answerer(question)
        good = score_answer(question, answer)
        totals[question.qtype] += 1
        if good:
            corrects[question.qtype] += 1
        elif len(failures) < max_failures_kept:
            failures.append({
                "question_id": question.question_id,
                "qtype": question.qtype,
                "question": question.question,
                "expected": " | ".join(question.expected) or NOT_IN_RECORD,
                "answer": answer[:300],
            })
    return ScorecardEntry(
        answerer=answerer_name,
        campaign_id=campaign,
        run_at=time.time(),
        total=sum(totals.values()),
        correct=sum(corrects.values()),
        by_type={t: {"total": totals[t], "correct": corrects[t]} for t in totals},
        failures=failures,
    )


def append_history(path: str | Path, entry: ScorecardEntry) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry), sort_keys=True, ensure_ascii=False) + "\n")


def read_history(path: str | Path) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def render_markdown(entry: ScorecardEntry, history: list[dict[str, Any]] | None = None) -> str:
    lines = [
        f"# Scorecard — {entry.answerer}",
        "",
        f"Overall: **{entry.correct}/{entry.total}** ({entry.accuracy:.0%})",
        "",
        "| Type | Correct | Total | Accuracy |",
        "|---|---|---|---|",
    ]
    for qtype in QUESTION_TYPES:
        stats = entry.by_type.get(qtype)
        if not stats:
            continue
        share = stats["correct"] / stats["total"] if stats["total"] else 0.0
        lines.append(f"| {qtype} | {stats['correct']} | {stats['total']} | {share:.0%} |")
    if entry.failures:
        lines += ["", f"## Failures ({len(entry.failures)} shown)", ""]
        for failure in entry.failures:
            lines.append(
                f"- [{failure['qtype']}] {failure['question']}\n"
                f"  expected: {failure['expected']}\n"
                f"  answered: {failure['answer']}"
            )
    if history:
        lines += ["", "## Trend (same answerer)", ""]
        for past in history[-10:]:
            if past.get("answerer") != entry.answerer:
                continue
            accuracy = past["correct"] / past["total"] if past["total"] else 0.0
            lines.append(f"- {time.strftime('%Y-%m-%d %H:%M', time.localtime(past['run_at']))}: "
                         f"{past['correct']}/{past['total']} ({accuracy:.0%})")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Seeding a bank from what already exists
# ---------------------------------------------------------------------------

def questions_from_feedback(memory: CampaignMemory, campaign_id: str) -> list[Question]:
    """Convert the human-decision archive into scored questions.

    Every "checker proposed X, human decided Y" becomes a question with a
    known answer — the existing archive seeds the scorecard with no new
    labelling work. Only human decisions are used.
    """
    questions: list[Question] = []
    seen: set[str] = set()
    for row in memory.all_feedback(campaign_id):
        if not str(row.get("actor", "")).startswith("human:"):
            continue
        kind, subject = row["kind"], row["subject"]
        observed, target = (subject.partition("->")[0].strip(),
                            subject.partition("->")[2].strip() or None)
        qid = f"fb-{row['feedback_id']}"
        if qid in seen:
            continue
        if kind in ("correction_accepted", "alias_added") and target and row["accepted"]:
            questions.append(Question(
                question_id=qid, campaign_id=campaign_id, qtype="alias",
                question=f'In this campaign, which known name does "{observed}" refer to?',
                expected=[target], session_id=row["session_id"],
                notes=f"from {kind}",
            ))
        elif kind == "entity_misclassified" and row["accepted"]:
            questions.append(Question(
                question_id=qid, campaign_id=campaign_id, qtype="trick",
                question=f'Which character, place or thing is "{observed}" in this campaign?',
                expected=["not an entity", NOT_IN_RECORD],
                session_id=row["session_id"], notes="human said: ordinary word",
            ))
        elif kind == "correction_rejected" and target and not row["accepted"]:
            questions.append(Question(
                question_id=qid, campaign_id=campaign_id, qtype="trick",
                question=f'Is "{observed}" another spelling of "{target}" in this campaign?',
                expected=["no", NOT_IN_RECORD],
                session_id=row["session_id"], notes="human rejected this correction",
            ))
        else:
            continue
        seen.add(qid)
    return questions
