"""Baseline answerers for the scorecard.

The point of a baseline is honesty, not brilliance: ``memory_answerer`` uses
only campaign memory (resolver for name questions, fact search otherwise) and
says NOT_IN_RECORD when it finds nothing. Every future answerer — local
Llama, cloud teacher, fine-tuned model — is scored against the same bank, so
"is the model actually adding anything over plain retrieval?" is always one
scorecard away from being answered.
"""
from __future__ import annotations

import re

from ..memory import CampaignMemory
from ..resolver import BandAction, NameResolver
from .scorecard import NOT_IN_RECORD, Answerer, Question

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
