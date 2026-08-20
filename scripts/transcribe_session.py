#!/usr/bin/env python3
"""Transcribe session audio parts locally and feed the engine — no cloud.

Designed for the "Audio Low ... Part N.mp3" exports. Two local backends, in
preference order:

1. **whisper.cpp server** (the bot's own transcription engine). Point it at
   the server you already run (default http://127.0.0.1:8178). Uses your GPU
   and your pinned ggml model — the exact quality the bot produces today.
2. **faster-whisper** (pip install faster-whisper), CPU/GPU, model choice via
   --fw-model (e.g. base.en, small.en, large-v3).

Output: a timestamped transcript whose speakers are all "Unknown" —
diarization stays with the bot. Because speakers are unknown, --process runs
against a DISPOSABLE evaluation database, never your permanent campaign
memory: Mapped transcripts are the only allowed input for that (use
`python -m transcripts_ai process` after the bot maps speakers).

**Craig multitrack zips** (craig-….aac.zip from the Discord recorder) are
the better input when you have one: every speaker has their own track, so
the transcript comes out speaker-labelled with no diarization at all.
Pass --craig with the zip (or its extracted folder) and optionally
--mapping to turn Discord usernames into character names. Craig's .aac
tracks decode natively with --backend faster-whisper; for whisper-cpp the
script converts via ffmpeg when it is on PATH.

Examples:
    python scripts/transcribe_session.py --out session.md part1.mp3 part2.mp3
    python scripts/transcribe_session.py --backend whisper-cpp \
        --server http://127.0.0.1:8178 --out 20260809_unmapped.md "*.mp3"
    python scripts/transcribe_session.py --out s.md --process \
        --campaign "<My Campaign>" --session 2026-08-09 \
        --game "<My Campaign>" "*Part*.mp3"
    python scripts/transcribe_session.py --backend faster-whisper \
        --craig "craig-abc123.aac.zip" --mapping mapping.json \
        --initial-prompt-file prompt.txt --out 20260816_craig.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PART_RE = re.compile(r"[Pp]art[ _](\d+)")


def _fmt(ms: float) -> str:
    ms = int(ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, milli = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{milli:03d}"


def sort_parts(paths: list[str]) -> list[Path]:
    def key(p: str):
        match = PART_RE.search(Path(p).name)
        return (int(match.group(1)) if match else 0, Path(p).name)
    return [Path(p) for p in sorted(paths, key=key)]


def transcribe_whisper_cpp(path: Path, server: str,
                           initial_prompt: str = "") -> list[tuple[float, float, str]]:
    """POST one audio file to a running whisper.cpp server /inference."""
    boundary = "----transcripts-ai"
    body = b""
    data = path.read_bytes()
    fields = {"response_format": "verbose_json", "temperature": "0.0"}
    if initial_prompt:
        fields["prompt"] = initial_prompt
    for name, value in fields.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="{name}"\r\n\r\n{value}\r\n').encode()
    content_type = "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
             f'filename="{path.name}"\r\nContent-Type: {content_type}\r\n\r\n').encode()
    body += data + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        server.rstrip("/") + "/inference",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        payload = json.loads(response.read().decode("utf-8"))
    segments = []
    for segment in payload.get("segments", []):
        segments.append((float(segment["start"]), float(segment["end"]),
                         segment["text"].strip()))
    return segments


DEFAULT_PROMPT = "Tabletop D&D session with dice rolls and fantasy names."


def transcribe_faster_whisper(path: Path, model_name: str,
                              initial_prompt: str = "") -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel  # lazy: optional dependency

    model = _FW_CACHE.setdefault(
        model_name, WhisperModel(model_name, device="auto", compute_type="auto")
    )
    segments, _info = model.transcribe(
        str(path), language="en", vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt or DEFAULT_PROMPT,
    )
    return [(s.start, s.end, s.text.strip()) for s in segments]


_FW_CACHE: dict = {}


def _ensure_decodable(path: Path, backend: str, workdir: Path) -> Path:
    """whisper.cpp servers usually reject AAC/Opus; convert via ffmpeg."""
    if backend != "whisper-cpp" or path.suffix.lower() in {".wav", ".mp3", ".flac"}:
        return path
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        print(f"WARNING: {path.name} is {path.suffix} and ffmpeg is not on "
              "PATH; sending as-is (if the server rejects it, install ffmpeg "
              "or use --backend faster-whisper)")
        return path
    workdir.mkdir(parents=True, exist_ok=True)
    converted = workdir / (path.stem + ".wav")
    if not converted.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-ar", "16000", "-ac", "1", str(converted)],
            check=True,
        )
    return converted


def run_craig(args, initial_prompt: str) -> int:
    from transcripts_ai import craig

    out = Path(args.out)
    source = Path(args.craig)
    if source.is_file():
        extract_dir = out.with_name(out.stem + "_craig_tracks")
        print(f"extracting {source.name} -> {extract_dir}")
        craig.extract_craig_zip(source, extract_dir)
        source = extract_dir
    tracks = craig.discover_tracks(source)
    print(f"{len(tracks)} speaker track(s): "
          + ", ".join(f"{t.number}-{t.speaker}" for t in tracks))

    per_track: dict[str, list[tuple[float, float, str]]] = {}
    convert_dir = out.with_name(out.stem + "_wav")
    for track in tracks:
        print(f"transcribing track {track.number} ({track.speaker}) ...")
        audio_path = _ensure_decodable(track.path, args.backend, convert_dir)
        if args.backend == "whisper-cpp":
            segments = transcribe_whisper_cpp(audio_path, args.server,
                                              initial_prompt)
        else:
            segments = transcribe_faster_whisper(audio_path, args.fw_model,
                                                 initial_prompt)
        per_track.setdefault(track.speaker, []).extend(segments)
        print(f"  {len(segments)} segments")

    merged = craig.merge_segments(per_track)
    if args.mapping:
        with open(args.mapping, encoding="utf-8") as f:
            mapping_data = json.load(f)
        speaker_map = craig.speaker_map_from_mapping(mapping_data)
        merged, unmapped = craig.apply_speaker_map(merged, speaker_map)
        if unmapped:
            print("NOT IN MAPPING (kept as Discord usernames): "
                  + ", ".join(unmapped))
            print("  add them to the mapping file's players/dm_labels and "
                  "re-run to get character names")
    out.write_text(craig.render_transcript(merged), encoding="utf-8")
    print(f"wrote {out} ({len(merged)} entries, "
          f"{len(per_track)} speakers)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="*", help="audio part files, any order")
    parser.add_argument("--craig",
                        help="Craig multitrack export: the craig-*.zip itself "
                             "or its extracted folder. Produces a "
                             "speaker-labelled transcript (one track per "
                             "speaker; no diarization needed)")
    parser.add_argument("--mapping",
                        help="mapping.json to rename Discord usernames to "
                             "character names (Craig mode)")
    parser.add_argument("--out", required=True, help="output transcript .md")
    parser.add_argument("--backend", choices=["whisper-cpp", "faster-whisper"],
                        default="whisper-cpp")
    parser.add_argument("--server", default="http://127.0.0.1:8178",
                        help="whisper.cpp server base URL")
    parser.add_argument("--fw-model", default="base.en")
    parser.add_argument("--initial-prompt", default="",
                        help="seed Whisper with campaign names (generate one "
                             "with: python -m transcripts_ai.lab whisper-prompt)")
    parser.add_argument("--initial-prompt-file",
                        help="read --initial-prompt from a file")
    parser.add_argument("--process", action="store_true",
                        help="run the engine's native pipeline afterwards "
                             "(EVALUATION ONLY: writes to a disposable database)")
    parser.add_argument("--campaign")
    parser.add_argument("--session")
    parser.add_argument("--game")
    parser.add_argument("--eval-db", default=None,
                        help="disposable evaluation database "
                             "(default: <out>_eval.sqlite). This script's "
                             "transcripts have no speaker names, so they are "
                             "never allowed into permanent campaign memory — "
                             "run the Mapped transcript through "
                             "`python -m transcripts_ai process` for that.")
    args = parser.parse_args()

    initial_prompt = args.initial_prompt
    if args.initial_prompt_file:
        initial_prompt = Path(args.initial_prompt_file).read_text(
            encoding="utf-8").strip()
    if initial_prompt:
        print(f"seeding Whisper with: {initial_prompt[:120]}"
              + ("..." if len(initial_prompt) > 120 else ""))

    if args.craig:
        if args.audio:
            print("--craig replaces the positional audio files; pass one or "
                  "the other")
            return 2
        return run_craig(args, initial_prompt)
    if not args.audio:
        print("nothing to do: pass audio files or --craig")
        return 2

    # Expand wildcards ourselves: PowerShell passes quoted globs literally.
    import glob as _glob
    expanded: list[str] = []
    for raw in args.audio:
        matches = _glob.glob(raw)
        expanded.extend(matches if matches else [raw])
    missing = [p for p in expanded if not Path(p).is_file()]
    if missing:
        print(f"audio file(s) not found: {', '.join(missing)}")
        return 2

    parts = sort_parts(expanded)
    print(f"{len(parts)} part(s): {', '.join(p.name for p in parts)}")

    lines: list[str] = []
    offset = 0.0
    for part in parts:
        print(f"transcribing {part.name} ...")
        if args.backend == "whisper-cpp":
            segments = transcribe_whisper_cpp(part, args.server, initial_prompt)
        else:
            segments = transcribe_faster_whisper(part, args.fw_model,
                                                 initial_prompt)
        for start, end, text in segments:
            if not text:
                continue
            lines.append(
                f"[{_fmt((offset + start) * 1000)} - {_fmt((offset + end) * 1000)}]"
                f" - Unknown: {text}"
            )
        if segments:
            offset += segments[-1][1]
        print(f"  {len(segments)} segments (session so far: {_fmt(offset * 1000)})")

    out = Path(args.out)
    out.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines)")

    if args.process:
        if not (args.campaign and args.session and args.game):
            print("--process needs --campaign, --session and --game")
            return 2
        from transcripts_ai.memory import CampaignMemory
        from transcripts_ai.pipeline import SessionPipeline

        eval_db = args.eval_db or str(out.with_name(out.stem + "_eval.sqlite"))
        print(f"NOTE: unlabelled transcript -> evaluation database {eval_db} "
              "(permanent campaign memory is Mapped-transcripts only)")
        memory = CampaignMemory(eval_db)
        try:
            report = SessionPipeline(memory).process_session_native(
                campaign_id=args.campaign,
                session_id=args.session,
                transcript_path=out,
                game_name=args.game,
                session_date=args.session,
            )
        finally:
            memory.close()
        print(f"facts verified: {len(report.facts_verified)}  "
              f"review items: {len(report.review_items)}")
        summary_path = out.with_name(out.stem + "_summary.md")
        summary_path.write_text(report.summary_markdown, encoding="utf-8")
        print(f"summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
