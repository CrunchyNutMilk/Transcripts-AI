"""Post-game quiz: recap trivia where every answer is a free gold label.

docs/TRAINING.md's cheapest labelling trick. After a session is
processed, its verified facts convert mechanically into quiz questions —
post them to Discord as recap trivia, and the table's answers (plus the
DM's vetoes of bad premises) are exactly the human decisions the
flywheel wants, gathered as a game instead of a chore.

Two outputs from the same generation:

- a **question bank** (the scorecard's own JSONL format, so the same
  questions also measure models forever), and
- a **Discord-ready Markdown** post with the answer key at the bottom.

Every question is evidence-backed: real questions carry the line the
answer came from; trick questions are built from entities the session's
facts explicitly do NOT support, so a "gotcha" can always be justified.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..memory import CampaignMemory
from ..schemas import EntityKind, Fact, FactCategory
from .scorecard import NOT_IN_RECORD, Question

DEFAULT_COUNT = 10
_TRICK_SHARE = 0.3       # roughly this share of the quiz is no-invention bait

# The number ADJACENT to "damage", never every digit in the statement
# ("rolled a 12 and took 24 damage" must answer 24, not 1224).
_DAMAGE_AMOUNT = re.compile(r"(\d+)\s+(?:\w+\s+)?damage\b", re.IGNORECASE)


def _qid(campaign_id: str, session_id: str, seed: str) -> str:
    digest = hashlib.sha256(
        f"{campaign_id}:{session_id}:{seed}".encode()).hexdigest()[:12]
    return f"quiz-{session_id}-{digest}"


@dataclass
class QuizItem:
    question: Question
    source_line: int | None = None    # None for trick questions


def _fact_question(fact: Fact, campaign_id: str, session_id: str) -> QuizItem | None:
    line = fact.provenance.line_start
    if fact.category is FactCategory.LOOT and fact.object_:
        return QuizItem(Question(
            question_id=_qid(campaign_id, session_id, f"loot:{fact.object_}"),
            campaign_id=campaign_id, qtype="fact",
            question="What item did the party obtain this session?",
            expected=[fact.object_], session_id=session_id,
            notes=f"line {line}"), source_line=line)
    if fact.category is FactCategory.COMBAT and fact.subject:
        # object_ is often empty on took_damage facts; the statement's
        # damage-adjacent number is the answer. Exactly one such number,
        # or no question — ambiguity must not reach the answer key.
        amounts = _DAMAGE_AMOUNT.findall(fact.statement)
        took_it = re.search(r"\b(?:took|takes?)\b", fact.statement,
                            re.IGNORECASE)
        if (len(set(amounts)) == 1 and took_it
                and fact.subject.casefold() not in ("party", "the party")):
            amount = amounts[0]
            return QuizItem(Question(
                question_id=_qid(campaign_id, session_id,
                                 f"combat:{fact.subject}:{amount}"),
                campaign_id=campaign_id, qtype="fact",
                question=f"How much damage did {fact.subject} take?",
                expected=[amount], session_id=session_id,
                notes=f"line {line}"), source_line=line)
    if fact.subject and fact.relationship and fact.object_ and \
            fact.subject.casefold() not in ("party", "the party"):
        verb = fact.relationship.replace("_", " ")
        # Relationship slugs make clumsy English; phrase the common ones.
        phrasing = {
            "cast spell": f"Which spell did {fact.subject} cast this session?",
            "obtained": f"What did {fact.subject} obtain this session?",
            "travelled to": f"Where did {fact.subject} travel this session?",
        }
        question = phrasing.get(
            verb, f"This session, what did {fact.subject} {verb}?")
        return QuizItem(Question(
            question_id=_qid(campaign_id, session_id,
                             f"triple:{fact.subject}:{verb}:{fact.object_}"),
            campaign_id=campaign_id, qtype="fact",
            question=question,
            expected=[fact.object_], session_id=session_id,
            notes=f"line {line}"), source_line=line)
    return None


def _ever_died(memory: CampaignMemory, campaign_id: str, name: str) -> bool:
    """Campaign-wide death check: a character who died in ANY session must
    never be 'never happened' bait — the record does say they died."""
    return any(f.category is FactCategory.DEATH_OR_STATUS
               for f in memory.facts_for_entity(campaign_id, name))


def _trick_questions(memory: CampaignMemory, campaign_id: str,
                     session_id: str, facts: list[Fact],
                     count: int, *, off_limits: set[str]) -> list[QuizItem]:
    """No-invention bait with justifiable 'gotcha's.

    False-death: a PC/NPC the CAMPAIGN's facts never kill (checked across
    all sessions, not just this one). False-loot: a known item this
    session's facts never award. ``off_limits`` holds names that are
    answers to the quiz's real questions — bait must never leak them into
    the question body. Correct answer is always the refusal.
    """
    looted = {f.object_.casefold() for f in facts
              if f.category is FactCategory.LOOT and f.object_}
    session_names = {e.casefold() for f in facts for e in f.entities}
    banned = off_limits | looted | session_names
    tricks: list[QuizItem] = []
    people = [e for e in memory.entities(campaign_id)
              if e.kind in (EntityKind.PC, EntityKind.NPC)
              and e.name.casefold() not in banned
              and not _ever_died(memory, campaign_id, e.name)]
    for entity in sorted(people, key=lambda e: e.name)[: max(1, count // 2)]:
        tricks.append(QuizItem(Question(
            question_id=_qid(campaign_id, session_id, f"death:{entity.name}"),
            campaign_id=campaign_id, qtype="trick",
            question=f"When did {entity.name} die?",
            expected=[], session_id=session_id,
            notes="no death anywhere in the record — correct answer is a refusal")))
    items = [e for e in memory.entities(campaign_id)
             if e.kind in (EntityKind.ITEM, EntityKind.WEAPON,
                           EntityKind.POTION, EntityKind.ARMOUR)
             and e.name.casefold() not in banned]
    for entity in sorted(items, key=lambda e: e.name)[: max(1, count // 2)]:
        tricks.append(QuizItem(Question(
            question_id=_qid(campaign_id, session_id, f"loot:{entity.name}"),
            campaign_id=campaign_id, qtype="trick",
            question=f"Who obtained the {entity.name} this session?",
            expected=[], session_id=session_id,
            notes="not awarded this session — correct answer is a refusal")))
    return tricks[:count]


def generate_quiz(memory: CampaignMemory, campaign_id: str, session_id: str,
                  *, count: int = DEFAULT_COUNT) -> list[QuizItem]:
    facts = [f for f in memory.facts_for_session(campaign_id, session_id)
             if not f.needs_review]
    real: list[QuizItem] = []
    seen_ids: set[str] = set()
    by_text: dict[str, QuizItem] = {}
    for fact in sorted(facts, key=lambda f: f.provenance.line_start):
        item = _fact_question(fact, campaign_id, session_id)
        if item is None or item.question.question_id in seen_ids:
            continue
        seen_ids.add(item.question.question_id)
        # The same question text twice with different answers would make
        # the quiz self-contradictory ("What item did the party obtain?"
        # x2). Merge into one any-match question instead.
        existing = by_text.get(item.question.question)
        if existing is not None:
            for answer in item.question.expected:
                if answer.casefold() not in {
                        e.casefold() for e in existing.question.expected}:
                    existing.question.expected.append(answer)
            existing.question.notes += f"; also line {item.source_line}"
            continue
        by_text[item.question.question] = item
        real.append(item)
    # At least one real question always survives when any exist; bait
    # fills the rest of the requested share.
    trick_count = min(max(1, int(count * _TRICK_SHARE)),
                      max(0, count - 1)) if real else 0
    real = real[: count - trick_count]
    off_limits = {answer.casefold() for item in real
                  for answer in item.question.expected}
    tricks = _trick_questions(memory, campaign_id, session_id, facts,
                              trick_count, off_limits=off_limits)
    return real + tricks


def render_quiz_markdown(items: list[QuizItem], *, campaign_id: str,
                         session_id: str) -> str:
    lines = [
        f"# {campaign_id} — recap trivia for {session_id}",
        "",
        "Answer from memory; \"that never happened\" is a legal answer",
        "(some questions are bait).",
        "",
    ]
    for number, item in enumerate(items, start=1):
        lines.append(f"{number}. {item.question.question}")
    lines += ["", "---", "", "## Answer key (DM eyes only)", ""]
    for number, item in enumerate(items, start=1):
        if item.question.qtype == "trick":
            answer = f"never happened — {item.question.notes}"
        else:
            answer = " / ".join(item.question.expected)
            if item.source_line is not None:
                answer += f"  (transcript line {item.source_line})"
        lines.append(f"{number}. {answer}")
    lines += [
        "", "_Every table answer here is a training label: mark disputes in "
        "the review queue and the engine learns from them._",
    ]
    return "\n".join(lines) + "\n"
