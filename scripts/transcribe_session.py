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

Examples:
    python scripts/transcribe_session.py --out session.md part1.mp3 part2.mp3
    python scripts/transcribe_session.py --backend whisper-cpp \
        --server http://127.0.0.1:8178 --out 20260809_unmapped.md "*.mp3"
    python scripts/transcribe_session.py --out s.md --process \
        --campaign "Heckuva Side Quest" --session 2026-08-09 \
        --game "Heckuva Side Quest" "*Part*.mp3"
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


def transcribe_whisper_cpp(path: Path, server: str) -> list[tuple[float, float, str]]:
    """POST one audio file to a running whisper.cpp server /inference."""
    boundary = "----transcripts-ai"
    body = b""
    data = path.read_bytes()
    fields = {"response_format": "verbose_json", "temperature": "0.0"}
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


def transcribe_faster_whisper(path: Path, model_name: str) -> list[tuple[float, float, str]]:
    from faster_whisper import WhisperModel  # lazy: optional dependency

    model = _FW_CACHE.setdefault(
        model_name, WhisperModel(model_name, device="auto", compute_type="auto")
    )
    segments, _info = model.transcribe(
        str(path), language="en", vad_filter=True,
        condition_on_previous_text=False,
        initial_prompt="Tabletop D&D session with dice rolls and fantasy names.",
    )
    return [(s.start, s.end, s.text.strip()) for s in segments]


_FW_CACHE: dict = {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", help="audio part files, any order")
    parser.add_argument("--out", required=True, help="output transcript .md")
    parser.add_argument("--backend", choices=["whisper-cpp", "faster-whisper"],
                        default="whisper-cpp")
    parser.add_argument("--server", default="http://127.0.0.1:8178",
                        help="whisper.cpp server base URL")
    parser.add_argument("--fw-model", default="base.en")
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
            segments = transcribe_whisper_cpp(part, args.server)
        else:
            segments = transcribe_faster_whisper(part, args.fw_model)
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
