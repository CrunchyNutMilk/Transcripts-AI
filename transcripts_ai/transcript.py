"""Mapped-transcript parsing, cleaning and chunking.

The Summurizer bot writes transcript lines as:

    [HH:MM:SS.mmm - HH:MM:SS.mmm] - <speaker>: <text>

This module parses that format (plus tolerated variants without millis or
with plain "Speaker: text" lines), never mutates the source text, and chunks
entries on speaker turns and time gaps so downstream AI calls receive
coherent, budget-bounded slices whose offsets reconstruct the original
exactly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import SchemaError, text_sha256

TIMESTAMP = r"(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d{1,3}))?"
LINE_RE = re.compile(
    rf"^\[{TIMESTAMP}\s*-\s*{TIMESTAMP}\]\s*-\s*(?P<speaker>[^:]+?):\s?(?P<text>.*)$"
)
BARE_LINE_RE = re.compile(r"^(?P<speaker>[A-Za-z0-9 _'\-\.]{1,64}?):\s(?P<text>.+)$")


def _ms(h: str, m: str, s: str, frac: str | None) -> int:
    millis = int((frac or "0").ljust(3, "0"))
    return ((int(h) * 60 + int(m)) * 60 + int(s)) * 1000 + millis


@dataclass
class TranscriptEntry:
    line_number: int           # 1-based line in the source file
    speaker: str
    text: str
    start_ms: int | None = None
    end_ms: int | None = None
    raw: str = ""

    @property
    def is_dm(self) -> bool:
        return self.speaker.strip().casefold() in {"dm", "gm", "dungeon master"}


@dataclass
class ParsedTranscript:
    source_path: str
    source_hash: str
    entries: list[TranscriptEntry]
    unparsed_lines: list[tuple[int, str]]
    total_lines: int

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.speaker, None)
        return list(seen)


def parse_transcript(text: str, *, source_path: str = "<memory>") -> ParsedTranscript:
    """Parse a Mapped transcript without modifying anything.

    Lines that match neither format are preserved in ``unparsed_lines`` (blank
    lines are ignored) so nothing silently disappears; continuation lines are
    appended to the previous entry's text.
    """
    entries: list[TranscriptEntry] = []
    unparsed: list[tuple[int, str]] = []
    lines = text.splitlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        match = LINE_RE.match(line)
        if match:
            groups = match.groups()
            entries.append(
                TranscriptEntry(
                    line_number=number,
                    speaker=match.group("speaker").strip(),
                    text=match.group("text").strip(),
                    start_ms=_ms(*groups[0:4]),
                    end_ms=_ms(*groups[4:8]),
                    raw=line,
                )
            )
            continue
        bare = BARE_LINE_RE.match(line)
        if bare:
            entries.append(
                TranscriptEntry(
                    line_number=number,
                    speaker=bare.group("speaker").strip(),
                    text=bare.group("text").strip(),
                    raw=line,
                )
            )
            continue
        if entries and line.startswith((" ", "\t")):
            entries[-1].text = f"{entries[-1].text} {line.strip()}".strip()
            entries[-1].raw = f"{entries[-1].raw}\n{line}"
            continue
        unparsed.append((number, line))
    return ParsedTranscript(
        source_path=source_path,
        source_hash=text_sha256(text),
        entries=entries,
        unparsed_lines=unparsed,
        total_lines=len(lines),
    )


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

@dataclass
class TranscriptChunk:
    chunk_number: int          # 1-based
    entries: list[TranscriptEntry] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(e.raw for e in self.entries)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def line_start(self) -> int:
        return self.entries[0].line_number

    @property
    def line_end(self) -> int:
        return self.entries[-1].line_number

    @property
    def start_ms(self) -> int | None:
        return self.entries[0].start_ms

    @property
    def end_ms(self) -> int | None:
        return self.entries[-1].end_ms

    @property
    def speakers(self) -> list[str]:
        seen: dict[str, None] = {}
        for entry in self.entries:
            seen.setdefault(entry.speaker, None)
        return list(seen)


DEFAULT_TARGET_CHARS = 4000
DEFAULT_MAX_CHARS = 6000
DEFAULT_MIN_CHARS = 1500
DEFAULT_MAX_ENTRIES = 40
LARGE_GAP_MS = 5 * 60 * 1000  # a 5-minute silence always starts a new chunk


def chunk_transcript(
    parsed: ParsedTranscript,
    *,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_chars: int = DEFAULT_MIN_CHARS,
    max_entries: int = DEFAULT_MAX_ENTRIES,
    large_gap_ms: int = LARGE_GAP_MS,
) -> list[TranscriptChunk]:
    """Split entries into chunks that reconstruct the source exactly.

    Boundary preferences, strongest first: a large time gap always breaks; the
    hard limits (max_chars / max_entries) always break; otherwise we break at
    the first speaker change after target_chars, so one speaker's thought is
    not split mid-turn. Chunks smaller than min_chars are merged forward
    unless a large gap separates them.
    """
    if not (0 < min_chars <= target_chars <= max_chars):
        raise SchemaError("invalid chunker configuration")

    chunks: list[TranscriptChunk] = []
    current: list[TranscriptEntry] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if current:
            chunks.append(TranscriptChunk(chunk_number=len(chunks) + 1, entries=current))
            current, current_chars = [], 0

    previous: TranscriptEntry | None = None
    for entry in parsed.entries:
        gap_break = (
            previous is not None
            and previous.end_ms is not None
            and entry.start_ms is not None
            and entry.start_ms - previous.end_ms >= large_gap_ms
        )
        entry_len = len(entry.raw) + 1
        over_hard_limit = current and (
            current_chars + entry_len > max_chars or len(current) >= max_entries
        )
        over_target_at_turn = (
            current
            and current_chars >= target_chars
            and previous is not None
            and entry.speaker != previous.speaker
        )
        if gap_break or over_hard_limit or over_target_at_turn:
            flush()
        current.append(entry)
        current_chars += entry_len
        previous = entry
    flush()

    # Merge a trailing runt into the previous chunk when no gap separates them.
    merged: list[TranscriptChunk] = []
    for chunk in chunks:
        if (
            merged
            and chunk.char_count < min_chars
            and not _gap_between(merged[-1], chunk, large_gap_ms)
            and merged[-1].char_count + chunk.char_count + 1 <= max_chars
        ):
            merged[-1].entries.extend(chunk.entries)
        else:
            merged.append(chunk)
    for number, chunk in enumerate(merged, start=1):
        chunk.chunk_number = number
    return merged


def _gap_between(a: TranscriptChunk, b: TranscriptChunk, large_gap_ms: int) -> bool:
    return (
        a.end_ms is not None
        and b.start_ms is not None
        and b.start_ms - a.end_ms >= large_gap_ms
    )


def validate_chunks(parsed: ParsedTranscript, chunks: list[TranscriptChunk]) -> None:
    """Assert the chunk set covers every parsed entry exactly once, in order."""
    flat = [e for chunk in chunks for e in chunk.entries]
    if [e.line_number for e in flat] != [e.line_number for e in parsed.entries]:
        raise SchemaError("chunk set does not reconstruct the parsed transcript")
