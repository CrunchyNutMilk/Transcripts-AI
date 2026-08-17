#!/usr/bin/env python3
"""Deterministic campaign-memory training pass over Mapped transcripts.

Runs the offline stages of the engine over a folder of
"... Transcript Mapped.md" files (with optional YAML frontmatter):

1. Read frontmatter (campaign, session_date); order sessions by date.
2. Parse and chunk each transcript; segment scenes; collect stats.
3. Register frequent speakers as PC/DM entities (speaker labels in a Mapped
   transcript are character names by construction).
4. Detect candidate entities with the rule-based detector, growing the
   known-name set session by session.
5. Cluster spelling variants (speaker-label drift and detected names) via the
   resolver's phonetic/edit-distance signals and queue them for human review —
   never auto-merging.
6. Persist everything to campaign memory (SQLite) with audit trail, and write
   a Markdown training report.

No AI provider is needed; this is the evidence substrate the AI stages build
on. Usage:

    python scripts/ingest_campaign.py --data-dir <folder> --db <memory.sqlite> \
        --campaign "Heckuva Side Quest" --report report.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transcripts_ai.detector import detect_names
from transcripts_ai.memory import CampaignMemory
from transcripts_ai.phonetics import damerau_levenshtein, phonetic_key
from transcripts_ai.resolver import NameResolver
from transcripts_ai.scenes import SceneType, segment_scenes
from transcripts_ai.schemas import (
    EntityKind,
    EntityRecord,
    EpistemicStatus,
    ReviewItem,
)
from transcripts_ai.transcript import chunk_transcript, parse_transcript, validate_chunks

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.S)
DATE_IN_NAME = re.compile(r"(\d{4})(\d{2})(\d{2})")

# A speaker label must appear in this share of sessions (or speak this many
# times overall) before we trust it as a PC — guests still qualify via count.
PC_MIN_SESSIONS = 2
PC_MIN_TURNS = 20
DM_LABELS = {"dm", "gm", "dungeon master"}
# Labels that must never be classified as PCs, whatever their frequency.
NEVER_PC = DM_LABELS | {"unknown", "narrator", "system", "bot", "speaker"}


def read_session(path: Path) -> tuple[str, str, str]:
    """Return (campaign, session_date, body) from a Mapped transcript file."""
    text = path.read_text(encoding="utf-8-sig")
    campaign, session_date = "", ""
    match = FRONTMATTER_RE.match(text)
    body = text
    if match:
        body = text[match.end():]
        for line in match.group(1).splitlines():
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip().strip('"')
            if key == "campaign":
                campaign = value
            elif key == "session_date":
                session_date = value
    if not session_date:
        found = DATE_IN_NAME.search(path.name)
        if found:
            session_date = "-".join(found.groups())
    return campaign, session_date, body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--campaign", required=True,
                        help="campaign id; frontmatter must match when present")
    parser.add_argument("--report", default=None)
    parser.add_argument("--mapping", default=None,
                        help="player mapping JSON (see mapping.example.json). "
                             "When given, its characters are the authoritative "
                             "PCs; unmapped frequent speakers only go to review.")
    args = parser.parse_args()

    # Walk the vault layout recursively; accept anything that is a Mapped
    # transcript by filename or frontmatter. Never touch Unmapped files.
    root = Path(args.data_dir)
    files = []
    for path in sorted(root.rglob("*.md")):
        name = path.name.casefold()
        if "unmapped" in name:
            continue
        if "transcript" in name and "mapped" in name:
            files.append(path)
            continue
        head = path.read_text(encoding="utf-8-sig", errors="replace")[:300]
        if "type: transcript-mapped" in head:
            files.append(path)
    if not files:
        print(f"no Mapped transcripts under {args.data_dir}")
        return 1

    memory = CampaignMemory(args.db)
    campaign_id = args.campaign
    actor = "engine-ingest"

    sessions: list[dict] = []
    speaker_turns: Counter[str] = Counter()
    speaker_sessions: defaultdict[str, set] = defaultdict(set)
    speaker_display: dict[str, str] = {}
    name_mentions: Counter[str] = Counter()
    name_display: dict[str, str] = {}
    name_reasons: defaultdict[str, set] = defaultdict(set)
    name_sessions: defaultdict[str, set] = defaultdict(set)
    scene_totals: Counter[SceneType] = Counter()

    ordered = []
    for path in files:
        campaign, session_date, body = read_session(path)
        if campaign and campaign != campaign_id:
            print(f"SKIP {path.name}: frontmatter campaign {campaign!r} != {campaign_id!r}")
            continue
        ordered.append((session_date or path.stem, path, body))
    ordered.sort()

    for session_date, path, body in ordered:
        parsed = parse_transcript(body, source_path=path.name)
        chunks = chunk_transcript(parsed)
        validate_chunks(parsed, chunks)
        scenes = segment_scenes(parsed.entries)
        for scene in scenes:
            scene_totals[scene.scene_type] += len(scene.entries)

        for entry in parsed.entries:
            folded = entry.speaker.casefold()
            speaker_turns[folded] += 1
            speaker_sessions[folded].add(session_date)
            speaker_display.setdefault(folded, entry.speaker)

        known = frozenset(
            e.name.casefold() for e in memory.entities(campaign_id)
        )
        detected = detect_names(parsed.entries, known_names=known)
        for name in detected:
            name_mentions[name.folded] += name.mentions
            name_display.setdefault(name.folded, name.text)
            name_reasons[name.folded].update(name.reasons)
            name_sessions[name.folded].add(session_date)

        sessions.append(
            {
                "date": session_date,
                "file": path.name,
                "entries": len(parsed.entries),
                "unparsed": len(parsed.unparsed_lines),
                "chunks": len(chunks),
                "scenes": len(scenes),
                "detected": len(detected),
            }
        )
        print(f"ingested {session_date}: {len(parsed.entries)} turns, "
              f"{len(chunks)} chunks, {len(scenes)} scenes, "
              f"{len(detected)} name candidates")

    # ---- speakers -> PC / player-handle entities ---------------------------
    # A speaker label with digits or camel-cased handle shape is almost
    # certainly a Discord username, not a character name; it is registered as
    # a PLAYER and queued for review instead of asserted as a PC. The bot's
    # mapping file remains the real authority — this is a best-effort default.
    def looks_like_handle(label: str) -> bool:
        if any(ch.isdigit() for ch in label):
            return True
        compact = label.replace(" ", "")
        interior_caps = sum(1 for ch in compact[1:] if ch.isupper())
        return " " not in label and interior_caps >= 1 and not label.istitle()

    # Authoritative mapping, when provided: its characters ARE the PCs.
    mapped_characters: dict[str, str] = {}   # folded label -> character name
    mapped_dm_labels: set[str] = set()
    if args.mapping:
        import json
        with open(args.mapping, encoding="utf-8") as f:
            mapping_data = json.load(f)
        if mapping_data.get("campaign") and mapping_data["campaign"] != campaign_id:
            print(f"mapping file is for {mapping_data['campaign']!r}, not {campaign_id!r}")
            return 2
        mapped_dm_labels = {l.casefold() for l in mapping_data.get("dm_labels", [])}
        for row in mapping_data.get("players", []) + mapping_data.get("overrides", []):
            mapped_characters[row["player_label"].casefold()] = row["character_name"]
            memory.upsert_entity(
                EntityRecord(
                    name=row["character_name"],
                    kind=EntityKind.PC,
                    campaign_id=campaign_id,
                    status=EpistemicStatus.CONFIRMED_CANON,
                    attributes={"player_id": str(row["player_id"]),
                                "player_label": row["player_label"]},
                ),
                actor="human:mapping-file",
            )
            # Character names spoken as labels also resolve to themselves.
            mapped_characters.setdefault(row["character_name"].casefold(),
                                         row["character_name"])

    pc_labels: list[str] = []
    handle_labels: list[str] = []
    for folded, turns in speaker_turns.most_common():
        display = speaker_display[folded]
        if folded in NEVER_PC or folded in mapped_dm_labels:
            continue
        if len(speaker_sessions[folded]) < PC_MIN_SESSIONS and turns < PC_MIN_TURNS:
            continue
        if mapped_characters:
            # Mapping is authoritative: speakers it covers are already PCs;
            # anything else frequent goes to review, never auto-PC.
            if folded in mapped_characters:
                pc_labels.append(mapped_characters[folded])
            else:
                memory.enqueue_review(
                    ReviewItem(
                        campaign_id=campaign_id,
                        session_id="ingest",
                        item_type="entity",
                        subject=display,
                        reason=("frequent speaker not in the mapping file — "
                                "guest PC, renamed character, or noise?"),
                        evidence=[f"{turns} turns across sessions: "
                                  + ", ".join(sorted(speaker_sessions[folded]))],
                        confidence=0.6,
                    ),
                    actor=actor,
                )
                handle_labels.append(display)
            continue
        if looks_like_handle(display):
            kind, bucket = EntityKind.PLAYER, handle_labels
        else:
            kind, bucket = EntityKind.PC, pc_labels
        memory.upsert_entity(
            EntityRecord(
                name=display,
                kind=kind,
                campaign_id=campaign_id,
                status=EpistemicStatus.STRONGLY_SUPPORTED,
                description=f"speaker in {len(speaker_sessions[folded])} session(s), "
                            f"{turns} turns",
            ),
            actor=actor,
        )
        bucket.append(display)
        if kind is EntityKind.PLAYER:
            memory.enqueue_review(
                ReviewItem(
                    campaign_id=campaign_id,
                    session_id="ingest",
                    item_type="entity",
                    subject=display,
                    reason=(
                        "speaker label looks like a Discord username, recorded as a "
                        "player handle — confirm which character this player controls"
                    ),
                    evidence=[
                        f"{turns} turns across sessions: "
                        + ", ".join(sorted(speaker_sessions[folded]))
                    ],
                    confidence=0.7,
                ),
                actor=actor,
            )

    # ---- speaker-label variant clustering (the Jinx/Jenx problem) ----------
    variant_pairs: list[tuple[str, str, str]] = []
    labels = [speaker_display[f] for f in speaker_turns
              if f not in NEVER_PC and f not in mapped_dm_labels]
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            fa, fb = a.casefold(), b.casefold()
            reason = None
            if fa in fb or fb in fa:
                reason = "one label contains the other"
            elif damerau_levenshtein(fa, fb, cap=2) <= (1 if min(len(fa), len(fb)) <= 5 else 2):
                reason = "spelling distance"
            elif phonetic_key(a) and phonetic_key(a) == phonetic_key(b):
                reason = "phonetic match"
            if reason:
                variant_pairs.append((a, b, reason))

    for a, b, reason in variant_pairs:
        major, minor = (a, b) if speaker_turns[a.casefold()] >= speaker_turns[b.casefold()] else (b, a)
        item = ReviewItem(
            campaign_id=campaign_id,
            session_id="ingest",
            item_type="spelling",
            subject=minor,
            reason=(
                f'speaker label "{minor}" looks like a variant of "{major}" '
                f"({reason}); {speaker_turns[minor.casefold()]} vs "
                f"{speaker_turns[major.casefold()]} turns — approve as alias?"
            ),
            evidence=[
                f'"{minor}" appears in sessions: '
                + ", ".join(sorted(speaker_sessions[minor.casefold()])),
                f'"{major}" appears in sessions: '
                + ", ".join(sorted(speaker_sessions[major.casefold()])),
            ],
            suggestions=[{"canonical": major, "score": 0.9, "reason": reason}],
            confidence=0.85,
        )
        memory.enqueue_review(item, actor=actor)

    # ---- detected non-speaker names ----------------------------------------
    resolver = NameResolver(memory)
    strong = {"self_introduction", "introduction", "location_of_phrase",
              "travel_target"}
    new_candidates = []
    speaker_foldeds = set(speaker_turns)
    for folded, mentions in name_mentions.most_common():
        if folded in speaker_foldeds:
            continue
        display = name_display[folded]
        reasons = name_reasons[folded]
        session_count = len(name_sessions[folded])
        # Credibility gate: repeated across sessions, or a strong pattern.
        if session_count < 2 and not (strong & reasons):
            continue
        # Human previously said "not an entity" -> never propose again.
        rows = memory.feedback_for(campaign_id, "entity_misclassified", display)
        if rows and rows[-1]["accepted"]:
            continue
        resolution = resolver.resolve(campaign_id, display)
        if resolution.best is not None and resolution.best.score == 1.0:
            continue  # already known exactly / via alias
        suggestions = [
            {"canonical": s.canonical, "score": s.score, "reason": "; ".join(s.reasons)}
            for s in resolution.suggestions
        ]
        item = ReviewItem(
            campaign_id=campaign_id,
            session_id="ingest",
            item_type="entity",
            subject=display,
            reason=(
                f"detected {mentions}x across {session_count} session(s); "
                f"signals: {', '.join(sorted(reasons))}"
                + ("" if suggestions else "; no close match in memory — possibly new")
            ),
            evidence=[
                "sessions: " + ", ".join(sorted(name_sessions[folded])),
            ],
            suggestions=suggestions,
            confidence=min(0.5 + 0.1 * session_count, 0.9),
        )
        memory.enqueue_review(item, actor=actor)
        new_candidates.append((display, mentions, session_count, sorted(reasons), suggestions))

    # ---- report -------------------------------------------------------------
    lines = [
        f"# Campaign training report — {campaign_id}",
        "",
        f"Sessions ingested: {len(sessions)}",
        "",
        "| Session | Turns | Chunks | Scenes | Name candidates | Unparsed lines |",
        "|---|---|---|---|---|---|",
    ]
    for s in sessions:
        lines.append(
            f"| {s['date']} | {s['entries']} | {s['chunks']} | {s['scenes']} "
            f"| {s['detected']} | {s['unparsed']} |"
        )
    total_entries = sum(s["entries"] for s in sessions) or 1
    lines += [
        "",
        "## Scene mix (share of speaker turns)",
        "",
    ]
    for scene_type, count in scene_totals.most_common():
        lines.append(f"- {scene_type.value}: {100 * count / total_entries:.1f}%")
    pc_labels = list(dict.fromkeys(pc_labels))
    lines += [
        "",
        "## Speakers registered as PCs (mapping-grade evidence)",
        "",
    ]
    for label in pc_labels:
        folded = label.casefold()
        lines.append(
            f"- **{label}** — {speaker_turns[folded]} turns across "
            f"{len(speaker_sessions[folded])} session(s)"
        )
    lines += [
        "",
        "## Speakers that look like Discord usernames (registered as players, review queued)",
        "",
    ]
    for label in handle_labels:
        folded = label.casefold()
        lines.append(
            f"- **{label}** — {speaker_turns[folded]} turns across "
            f"{len(speaker_sessions[folded])} session(s)"
        )
    lines += [
        "",
        "## Speaker-label variants queued for alias review",
        "",
    ]
    for a, b, reason in variant_pairs:
        lines.append(f"- `{a}` ↔ `{b}` ({reason})")
    lines += [
        "",
        f"## Entity candidates queued for review ({len(new_candidates)})",
        "",
    ]
    for display, mentions, session_count, reasons, suggestions in new_candidates[:60]:
        suffix = ""
        if suggestions:
            best = suggestions[0]
            suffix = f' — closest known: "{best["canonical"]}" ({best["reason"]})'
        lines.append(
            f"- **{display}** — {mentions} mention(s), {session_count} session(s); "
            f"{', '.join(reasons)}{suffix}"
        )
    if len(new_candidates) > 60:
        lines.append(f"- … and {len(new_candidates) - 60} more in the review queue")
    report = "\n".join(lines) + "\n"
    if args.report:
        Path(args.report).write_text(report, encoding="utf-8")
        print(f"report written to {args.report}")
    print(f"\nPCs: {len(pc_labels)}  variant pairs: {len(variant_pairs)}  "
          f"entity candidates: {len(new_candidates)}  "
          f"pending reviews: {len(memory.pending_reviews(campaign_id))}")
    memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
