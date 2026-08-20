"""Gold transcripts: turn one hand-corrected session into a pile of labels.

The workflow docs/TRAINING.md calls "perfect transcripts": record the game,
let Whisper produce a machine transcript, fix it by hand once, and this
module mines the difference —

- **WER** (word error rate) per session: the transcription-layer metric on
  the three-layer scoreboard, comparable run over run.
- **Substitution pairs** ("heard → truth"): every fix made becomes a
  training record (``source="gold-transcript"``), fuel for the misspelling
  map, Whisper prompt tuning, and eventually fine-tuning data.
- **Name accuracy**: of every place a known campaign name appears in the
  gold text, how often did the machine get it right — and what did it
  write instead? Names are what break fact extraction, so this number
  matters more than overall WER.

Alignment is text-only (speakers are ignored): machine transcripts from
``scripts/transcribe_session.py`` are all "Unknown" while gold transcripts
carry real speakers, and diarization is not what is being measured here.

Also here: ``whisper_prompt`` — the cheapest big win. It builds an
``initial_prompt`` from campaign memory (PCs first, then NPCs, places,
the rest) so Whisper has seen "Ghomra" and "Nwen'sua" before it ever
decodes the audio. Feed it to ``scripts/transcribe_session.py
--initial-prompt-file``.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from ..memory import CampaignMemory
from ..transcript import parse_transcript
from .records import TrainingRecord

_WORD = re.compile(r"[a-z0-9]+(?:['\-][a-z0-9]+)*")

# Whisper conditions on roughly the last 224 tokens of the prompt; keep the
# generated prompt comfortably inside that so no name falls off the front.
PROMPT_MAX_CHARS = 700

_PROMPT_KIND_ORDER = ("pc", "npc", "location", "faction", "creature", "deity",
                      "item", "weapon", "armour", "potion", "spell", "quest",
                      "player", "lore")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


@dataclass
class _Word:
    token: str
    line: int          # source line in its transcript
    raw_line: str      # the raw entry text, for evidence quotes


def _words_of(transcript_text: str, *, source_path: str) -> list[_Word]:
    parsed = parse_transcript(transcript_text, source_path=source_path)
    words: list[_Word] = []
    for entry in parsed.entries:
        for token in _tokens(entry.text):
            words.append(_Word(token=token, line=entry.line_number,
                               raw_line=entry.text))
    return words


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class Substitution:
    heard: str          # what the machine wrote
    truth: str          # what the human corrected it to
    gold_line: int
    quote: str          # the corrected line, as evidence


@dataclass
class NameMiss:
    name: str           # the known name as stored in memory
    heard: str          # what the machine wrote in that spot ("" if dropped)
    gold_line: int


@dataclass
class GoldReport:
    gold_words: int
    substitutions: int
    deletions: int      # words the machine dropped
    insertions: int     # words the machine invented
    pairs: list[Substitution] = field(default_factory=list)
    name_total: int = 0
    name_hits: int = 0
    name_misses: list[NameMiss] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.gold_words if self.gold_words else 0.0

    @property
    def name_accuracy(self) -> float:
        return self.name_hits / self.name_total if self.name_total else 0.0


def known_names(memory: CampaignMemory, campaign_id: str) -> dict[str, str]:
    """observed-form (casefolded) -> canonical name, entities and aliases."""
    names: dict[str, str] = {}
    for alias in memory.aliases(campaign_id):
        names[alias.observed.casefold()] = alias.canonical
    for entity in memory.entities(campaign_id):
        names[entity.name.casefold()] = entity.name   # canonical wins
    return names


def compare_transcripts(
    hypothesis_text: str,
    gold_text: str,
    *,
    names: dict[str, str] | None = None,
) -> GoldReport:
    """Word-level alignment of machine output against the corrected truth."""
    hyp = _words_of(hypothesis_text, source_path="<hypothesis>")
    gold = _words_of(gold_text, source_path="<gold>")
    matcher = SequenceMatcher(a=[w.token for w in hyp],
                              b=[w.token for w in gold], autojunk=False)

    report = GoldReport(gold_words=len(gold), substitutions=0,
                        deletions=0, insertions=0)
    # For name scoring: whether each gold word survived unchanged, and what
    # sat in its place when it did not.
    gold_ok = [False] * len(gold)
    gold_heard = [""] * len(gold)

    for op, a1, a2, b1, b2 in matcher.get_opcodes():
        if op == "equal":
            for offset in range(b2 - b1):
                gold_ok[b1 + offset] = True
                gold_heard[b1 + offset] = gold[b1 + offset].token
        elif op == "replace":
            span_a, span_b = a2 - a1, b2 - b1
            report.substitutions += min(span_a, span_b)
            if span_a > span_b:
                report.insertions += span_a - span_b
            else:
                report.deletions += span_b - span_a
            if span_a == span_b:
                # one-to-one pairs: the confident, minable corrections
                for offset in range(span_b):
                    g = gold[b1 + offset]
                    report.pairs.append(Substitution(
                        heard=hyp[a1 + offset].token, truth=g.token,
                        gold_line=g.line, quote=g.raw_line))
                    gold_heard[b1 + offset] = hyp[a1 + offset].token
            else:
                heard_phrase = " ".join(w.token for w in hyp[a1:a2])
                for offset in range(span_b):
                    gold_heard[b1 + offset] = heard_phrase
        elif op == "delete":       # in hypothesis only: machine invented words
            report.insertions += a2 - a1
        elif op == "insert":       # in gold only: machine dropped words
            report.deletions += b2 - b1

    if names:
        _score_names(gold, gold_ok, gold_heard, names, report)
    return report


def _score_names(gold: list[_Word], gold_ok: list[bool], gold_heard: list[str],
                 names: dict[str, str], report: GoldReport) -> None:
    """Every occurrence of a known name in the gold token stream: did the
    machine transcribe the whole name correctly?"""
    name_tokens = {form: tuple(_tokens(form)) for form in names}
    max_len = max((len(t) for t in name_tokens.values() if t), default=0)
    index = 0
    while index < len(gold):
        matched = None
        for length in range(min(max_len, len(gold) - index), 0, -1):
            window = tuple(w.token for w in gold[index:index + length])
            for form, toks in name_tokens.items():
                if toks == window:
                    matched = (form, length)
                    break
            if matched:
                break
        if not matched:
            index += 1
            continue
        form, length = matched
        report.name_total += 1
        if all(gold_ok[index:index + length]):
            report.name_hits += 1
        else:
            heard = " ".join(dict.fromkeys(
                gold_heard[index + o] for o in range(length))).strip()
            report.name_misses.append(NameMiss(
                name=names[form], heard=heard, gold_line=gold[index].line))
        index += length


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def records_from_gold(report: GoldReport, *, campaign_id: str,
                      session_id: str) -> list[TrainingRecord]:
    """One record per confident one-to-one correction. These ARE human
    decisions — the human wrote the gold transcript — so they carry a
    human actor and feed the same flywheel as review answers."""
    records = []
    for pair in report.pairs:
        records.append(TrainingRecord(
            campaign_id=campaign_id,
            session_id=session_id,
            item_type="transcription",
            subject=pair.heard,
            proposal=pair.heard,
            decision="substitute",
            choice=pair.truth,
            accepted=True,
            evidence_quote=pair.quote[:300],
            source="gold-transcript",
            actor="human:gold-transcript",
            extras={"gold_line": pair.gold_line},
        ))
    return records


def render_markdown(report: GoldReport, *, session_id: str = "") -> str:
    top = Counter((p.heard, p.truth) for p in report.pairs).most_common(30)
    lines = [
        f"# Gold transcript report{' — ' + session_id if session_id else ''}",
        "",
        f"- Gold words: **{report.gold_words}**",
        f"- WER: **{report.wer:.1%}**  "
        f"({report.substitutions} substituted, {report.deletions} dropped, "
        f"{report.insertions} invented)",
    ]
    if report.name_total:
        lines.append(f"- Known-name accuracy: **{report.name_accuracy:.1%}** "
                     f"({report.name_hits}/{report.name_total})")
    if report.name_misses:
        lines += ["", "## Names the machine got wrong", ""]
        by_name = Counter((m.name, m.heard) for m in report.name_misses)
        for (name, heard), count in by_name.most_common(40):
            lines.append(f'- **{name}** heard as "{heard or "(dropped)"}" '
                         f"×{count}")
    if top:
        lines += ["", "## Most common corrections", "",
                  "| Machine wrote | Truth | Count |", "|---|---|---|"]
        for (heard, truth), count in top:
            lines.append(f"| {heard} | {truth} | {count} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Whisper prompt seeding
# ---------------------------------------------------------------------------

def whisper_prompt(memory: CampaignMemory, campaign_id: str,
                   *, max_chars: int = PROMPT_MAX_CHARS) -> str:
    """A Whisper ``initial_prompt`` that has already met the campaign.

    Important names first (PCs, then NPCs, places, …) so truncation drops
    the least valuable ones. Distinct alias spellings are left out: the
    prompt should pull Whisper toward canonical spellings, not variants.
    """
    order = {kind: rank for rank, kind in enumerate(_PROMPT_KIND_ORDER)}
    entities = sorted(
        memory.entities(campaign_id),
        key=lambda e: (order.get(e.kind.value, len(order)), e.name.casefold()),
    )
    seen: set[str] = set()
    names: list[str] = []
    for entity in entities:
        folded = entity.name.casefold()
        if folded not in seen:
            seen.add(folded)
            names.append(entity.name)
    prefix = "Tabletop D&D session. Names in this campaign: "
    body = ""
    for name in names:
        extended = f"{body}, {name}" if body else name
        if len(prefix) + len(extended) + 1 > max_chars:
            break
        body = extended
    return f"{prefix}{body}." if body else \
        "Tabletop D&D session with dice rolls and fantasy names."
