#!/usr/bin/env python3
"""Rewrite transcript speaker names using mapping.json — nothing else.

Turns Craig's text export (``Username: text``) or any engine-format
transcript into one whose speakers are character names, using the same
mapping file the rest of the engine treats as authority. Only the speaker
labels change; every other byte, line number and timestamp is preserved.

    python scripts/relabel_transcript.py --in craig_export.txt \
        --mapping mapping.json --out "2026-08-16 Transcript Mapped.md"

Speakers not in the mapping are kept unchanged and listed, so you can add
them to mapping.json (players / dm_labels) and run again.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcripts_ai.craig import relabel_transcript_text, speaker_map_from_mapping


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="source", required=True,
                        help="transcript to relabel (Craig txt or engine md)")
    parser.add_argument("--mapping", required=True,
                        help="mapping.json (see mapping.example.json)")
    parser.add_argument("--out", required=True, help="relabelled output file")
    args = parser.parse_args()

    source = Path(args.source)
    out = Path(args.out)
    if source.resolve() == out.resolve():
        print("refusing to overwrite the input; pick a different --out")
        return 2

    with open(args.mapping, encoding="utf-8") as f:
        mapping_data = json.load(f)
    speaker_map = speaker_map_from_mapping(mapping_data)

    text = source.read_text(encoding="utf-8")
    relabelled, unmapped = relabel_transcript_text(text, speaker_map)
    out.write_text(relabelled, encoding="utf-8")
    print(f"wrote {out}")
    if unmapped:
        print("NOT IN MAPPING (kept as-is): " + ", ".join(unmapped))
        print("  add them to the mapping file's players/dm_labels and re-run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
