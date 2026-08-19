"""Baseline answerers for the scorecard.

The point of a baseline is honesty, not brilliance: ``memory_answerer`` uses
only campaign memory (resolver for name questions, fact search otherwise) and
says NOT_IN_RECORD when it finds nothing. Every future answerer — local
Llama, cloud teacher, fine-tuned model — is scored against the same bank, so
"is the model actually adding anything over plain retrieval?" is always one
scorecard away from being answered.

``teacher_answerer`` wraps a cloud/local teacher: it hands the model the
same retrieval the memory baseline gets (candidate names, matching facts)
and lets it phrase the answer — so a teacher's score measures reading
comprehension over the record, not trivia knowledge it cannot have.
"""
from __future__ import annotations

import re

from ..memory import CampaignMemory
from ..providers import ProviderError, ValidationFailed, call_role
from ..resolver import BandAction, NameResolver
from .scorecard import NOT_IN_RECORD, Answerer, Question
from .teachers import Teacher

_QUOTED = re.compile(r'["“\']([^"”\']{1,60})["”\']')
_CAPITALISED = re.compile(r"\b[A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+){0,3}\b")
_STOPWORDS = frozenset(
    "in this campaign which known name does refer to who what where when is "
    "the a an of another spelling character place thing".split()
)


def _subjects_of(question: Question) -> list[str]:
    """Names the question is about: quoted spans first, then capitalised."""
    subjects = _QUOTED.findall(question.question)
    if subjects:
        return subjects
    return [
        m for m in _CAPITALISED.findall(question.question)
        if m.casefold() not in _STOPWORDS
    ]


def memory_answerer(memory: CampaignMemory, campaign_id: str) -> Answerer:
    resolver = NameResolver(memory)

    def answer(question: Question) -> str:
        subjects = _subjects_of(question)

        if question.qtype in ("alias", "trick"):
            for subject in subjects:
                resolution = resolver.resolve(campaign_id, subject)
                best = resolution.best
                if best is not None and best.band in (BandAction.AUTO_LINK,
                                                      BandAction.SUGGEST):
                    return best.canonical
            return NOT_IN_RECORD

        # fact questions: search remembered facts with the question's terms.
        terms = [w for w in re.findall(r"[\w'\-]{4,}", question.question)
                 if w.casefold() not in _STOPWORDS]
        for probe in (subjects + [" ".join(terms[:4])] if terms else subjects):
            if not probe:
                continue
            facts = memory.search_facts(campaign_id, probe, limit=3)
            if facts:
                return facts[0].statement
        return NOT_IN_RECORD

    return answer


# ---------------------------------------------------------------------------
# Teachers as answerers
# ---------------------------------------------------------------------------

ANSWER_SYSTEM = f"""\
You answer questions about ONE Dungeons & Dragons campaign using ONLY the
campaign record provided with each question. Rules:
- If the record does not contain the answer, your answer must be exactly:
  {NOT_IN_RECORD}
- Never guess and never use outside knowledge; questions may be traps whose
  correct answer is that there is no answer.
- Reply with ONLY a JSON object: {{"answer": "<your answer>"}}
"""


def _answer_validator(payload: object) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        return ['reply must be a JSON object like {"answer": "..."}']
    return []


def _record_pack(memory: CampaignMemory, campaign_id: str,
                 question: Question) -> str:
    """The campaign record shown to a teacher: same retrieval the memory
    baseline uses, rendered as text. Deterministic per (memory, question)."""
    resolver = NameResolver(memory)
    names: list[str] = []
    facts: list[str] = []
    seen_names: set[str] = set()
    seen_facts: set[str] = set()
    for subject in _subjects_of(question):
        for suggestion in resolver.resolve(campaign_id, subject).suggestions:
            folded = suggestion.canonical.casefold()
            if folded not in seen_names:
                seen_names.add(folded)
                names.append(f"- {suggestion.canonical} ({suggestion.explanation})"
                             if suggestion.explanation else f"- {suggestion.canonical}")
        for fact in memory.search_facts(campaign_id, subject, limit=3):
            if fact.statement not in seen_facts:
                seen_facts.add(fact.statement)
                facts.append(f"- {fact.statement}")
    lines = ["CAMPAIGN RECORD", "", "Known names matching the question:"]
    lines += names or ["- (none)"]
    lines += ["", "Recorded facts matching the question:"]
    lines += facts or ["- (none)"]
    return "\n".join(lines)


def teacher_answerer(memory: CampaignMemory, campaign_id: str,
                     teacher: Teacher) -> Answerer:
    def answer(question: Question) -> str:
        user = (f"{_record_pack(memory, campaign_id, question)}\n\n"
                f"QUESTION: {question.question}")
        try:
            payload, _ = call_role(
                teacher.provider,
                system=ANSWER_SYSTEM,
                user=user,
                validator=_answer_validator,
                max_tokens=teacher.max_tokens,
                usage_sink=teacher.record_usage,
            )
        except (ProviderError, ValidationFailed):
            return NOT_IN_RECORD   # a broken teacher refuses; it never guesses
        return payload["answer"].strip() or NOT_IN_RECORD

    return answer
