"""Craig multitrack recordings: free diarization from filenames.

Craig (the Discord recording bot) exports one audio track per speaker,
named ``NN-username.ext`` inside a ``craig-….aac.zip``. Because every
track belongs to exactly one person, a speaker-labelled transcript needs
no diarization model at all: transcribe each track separately, stamp
every segment with the track's speaker, and merge by time.

This module is the pure logic (no audio decoding, no network): filename
parsing, zip/folder discovery, time-ordered merging, and mapping Discord
usernames to character names with the same ``mapping.json`` the rest of
the engine treats as authority. ``scripts/transcribe_session.py --craig``
drives it.

All timestamps are session-relative: Craig pads every track to a common
start, so segments from different tracks merge without offsets —
overlapping speech stays overlapping, which is the truth of the table.
"""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {".aac", ".flac", ".opus", ".ogg", ".mp3", ".wav", ".m4a"}

# "2-crunchynutmilk.aac", "10-NorthTitan_1234.flac" -> (track, speaker).
_TRACK_RE = re.compile(r"^(\d+)-(.+)$")
_OLD_DISCRIMINATOR = re.compile(r"_\d{4}$")


class CraigError(ValueError):
    """Raised when a Craig export cannot be understood."""


@dataclass(frozen=True)
class CraigTrack:
    number: int
    speaker: str          # Discord username as Craig recorded it
    path: Path


@dataclass
class Segment:
    start: float          # seconds, session-relative
    end: float
    speaker: str
    text: str


def parse_track_filename(name: str) -> tuple[int, str] | None:
    """``"2-crunchynutmilk_1234.aac"`` -> ``(2, "crunchynutmilk")``.

    Returns None for files that are not per-speaker tracks (info.txt,
    raw.dat, ffmpeg licences, …). The 4-digit suffix is the old Discord
    discriminator Craig used to append; it is display noise, not name.
    """
    path = Path(name)
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        return None
    match = _TRACK_RE.match(path.stem)
    if not match:
        return None
    speaker = _OLD_DISCRIMINATOR.sub("", match.group(2)) or match.group(2)
    return int(match.group(1)), speaker


def extract_craig_zip(zip_path: Path, dest: Path) -> Path:
    """Extract only the audio tracks of a Craig zip into ``dest``."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            base = Path(info.filename).name
            if info.is_dir() or parse_track_filename(base) is None:
                continue
            target = dest / base           # flatten: never trust zip paths
            with archive.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return dest


def discover_tracks(source: Path) -> list[CraigTrack]:
    """Find per-speaker tracks in an extracted folder, track order."""
    tracks: list[CraigTrack] = []
    for path in sorted(source.iterdir()):
        if not path.is_file():
            continue
        parsed = parse_track_filename(path.name)
        if parsed is None:
            continue
        number, speaker = parsed
        tracks.append(CraigTrack(number=number, speaker=speaker, path=path))
    tracks.sort(key=lambda t: t.number)
    if not tracks:
        raise CraigError(
            f"no per-speaker tracks found in {source} "
            "(expected files like '1-username.aac')"
        )
    return tracks


# ---------------------------------------------------------------------------
# Speaker mapping (same mapping.json as ingest/process)
# ---------------------------------------------------------------------------

def speaker_map_from_mapping(mapping_data: dict) -> dict[str, str]:
    """Casefolded Discord label -> transcript speaker name.

    DM labels become "DM"; players (then overrides, which win) become
    their character names. Craig usernames not in the file stay as-is —
    the caller should surface them so the mapping can be extended.
    """
    result: dict[str, str] = {}
    for label in mapping_data.get("dm_labels", []):
        result[str(label).casefold()] = "DM"
    for row in mapping_data.get("players", []) + mapping_data.get("overrides", []):
        label = str(row.get("player_label", "")).casefold()
        character = str(row.get("character_name", "")).strip()
        if label and character:
            result[label] = character
    return result


def apply_speaker_map(
    segments: list[Segment], speaker_map: dict[str, str]
) -> tuple[list[Segment], list[str]]:
    """Rename speakers via the mapping; returns (segments, unmapped names)."""
    unmapped: dict[str, None] = {}
    renamed: list[Segment] = []
    for segment in segments:
        mapped = speaker_map.get(segment.speaker.casefold())
        if mapped is None:
            unmapped.setdefault(segment.speaker)
        renamed.append(Segment(start=segment.start, end=segment.end,
                               speaker=mapped or segment.speaker,
                               text=segment.text))
    return renamed, list(unmapped)


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def merge_segments(per_track: dict[str, list[tuple[float, float, str]]]) -> list[Segment]:
    """Interleave every speaker's segments into one session timeline."""
    merged: list[Segment] = []
    for speaker, segments in per_track.items():
        for start, end, text in segments:
            text = text.strip()
            if text:
                merged.append(Segment(start=start, end=end,
                                      speaker=speaker, text=text))
    merged.sort(key=lambda s: (s.start, s.end, s.speaker))
    return merged


def _fmt(seconds: float) -> str:
    ms = int(seconds * 1000)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def render_transcript(segments: list[Segment]) -> str:
    """The engine's Mapped-transcript line format, one entry per segment."""
    lines = [
        f"[{_fmt(s.start)} - {_fmt(s.end)}] - {s.speaker}: {s.text}"
        for s in segments
    ]
    return "\n\n".join(lines) + ("\n" if lines else "")


# ---------------------------------------------------------------------------
# Relabelling Craig's text exports (Discord usernames -> character names)
# ---------------------------------------------------------------------------

def relabel_transcript_text(
    text: str, speaker_map: dict[str, str]
) -> tuple[str, list[str]]:
    """Rewrite ONLY the speaker names in a transcript, mapping-authoritative.

    Craig's text export is ``Username: text`` lines; the bot's format adds
    timestamps. Both are handled with the parser's own line patterns, and
    every other byte — text, blank lines, unparsed lines — is preserved
    exactly, so line numbers (provenance) survive relabelling.

    Returns the rewritten text and the speakers that were not in the map
    (kept unchanged; the caller should show them so the mapping grows).
    """
    from .transcript import BARE_LINE_RE, LINE_RE

    unmapped: dict[str, None] = {}
    out_lines: list[str] = []
    for line in text.splitlines():
        match = LINE_RE.match(line) or BARE_LINE_RE.match(line)
        if match:
            speaker = match.group("speaker").strip()
            mapped = speaker_map.get(speaker.casefold())
            if mapped is None:
                unmapped.setdefault(speaker)
            else:
                start, end = match.span("speaker")
                line = line[:start] + mapped + line[end:]
        out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), \
        list(unmapped)
