"""Command-line interface for the engine.

    python -m transcripts_ai process --campaign camp-a --session 2026-08-01 \
        --transcript "path/to/... - Transcripts Mapped.md" --game "<My Campaign>"
    python -m transcripts_ai ingest ...      # memory-only pass over an old session
    python -m transcripts_ai reviews --campaign camp-a
    python -m transcripts_ai facts --campaign camp-a --query "moon sickle"
    python -m transcripts_ai forget --campaign camp-a --fact-id abc123 --reason "wrong"
    python -m transcripts_ai export --campaign camp-a  # verified fine-tune dataset
    python -m transcripts_ai audit --campaign camp-a

Provider configuration comes from the environment (see README): AI_ROLE_* for
each role, OPENAI_API_KEY / OPENAI_BASE_URL for cloud, LOCAL_AI_BASE_URL for a
local OpenAI-compatible server. The DB path defaults to ./engine_memory.sqlite.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .memory import CampaignMemory
from .pipeline import SessionPipeline
from .providers import RoleRegistry


def _memory(args: argparse.Namespace) -> CampaignMemory:
    return CampaignMemory(args.db)


def cmd_vault_sync(args: argparse.Namespace) -> int:
    from collections import Counter

    from .vault import DEFAULT_SECTIONS, scan_vault, sync_vault

    sections = tuple(args.sections) if args.sections else DEFAULT_SECTIONS
    entities, skipped = scan_vault(args.vault, campaign_id=args.campaign,
                                   sections=sections)
    if not entities:
        print("no entity pages found — check --vault points at the campaign "
              "folder (the one containing '03 - PCs')")
        return 1
    by_kind = Counter(e.kind.value for e in entities)
    alias_total = sum(len(e.aliases) for e in entities)
    print(f"{len(entities)} vault entit(ies), {alias_total} alias(es): "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))
    if args.dry_run:
        for entity in entities:
            aka = f"  (aka {', '.join(entity.aliases)})" if entity.aliases else ""
            print(f"  [{entity.kind.value:8s}] {entity.name}{aka}")
        print(f"dry run: nothing written; {len(skipped)} page(s) skipped")
        return 0
    memory = _memory(args)
    try:
        entity_count, alias_count = sync_vault(memory, args.campaign, entities)
    finally:
        memory.close()
    print(f"synced {entity_count} entit(ies) and {alias_count} alias(es) "
          f"into {args.db}")
    if skipped:
        print(f"skipped {len(skipped)} page(s) (structural/other-campaign); "
              "use --dry-run to list them")
    return 0


def load_session_mapping(path: str, campaign_id: str, session_id: str,
                         game: str, session_date: str):
    """Load a mapping JSON file into a SessionContext (see mapping.example.json)."""
    import json

    from .session_context import PlayerMapping, SessionContext

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("campaign") and data["campaign"] != campaign_id:
        raise SystemExit(
            f"mapping file is for campaign {data['campaign']!r}, not {campaign_id!r}"
        )

    def rows(key):
        return [
            PlayerMapping(str(r["player_id"]), r["player_label"], r["character_name"])
            for r in data.get(key, [])
        ]

    return SessionContext(
        campaign_id=campaign_id,
        session_id=session_id,
        game_name=game,
        session_date=session_date,
        dm_labels=frozenset(data.get("dm_labels") or ["DM"]),
        mappings=rows("players"),
        overrides=rows("overrides"),
    )


def cmd_process(args: argparse.Namespace) -> int:
    import os
    from pathlib import Path

    name = Path(args.transcript).name.casefold()
    if "unmapped" in name or "unknown" in name:
        print(
            "REFUSED: permanent campaign memory only accepts Mapped transcripts.\n"
            f"{args.transcript} looks unmapped. Run it through the bot's speaker\n"
            "mapping first, or evaluate it with scripts/transcribe_session.py\n"
            "--process (disposable database)."
        )
        return 3

    session_date = args.date or args.session
    session_context = None
    if args.mapping:
        session_context = load_session_mapping(
            args.mapping, args.campaign, args.session, args.game, session_date
        )

    memory = _memory(args)
    use_native = args.native or not os.environ.get("AI_ROLE_EXTRACTOR")
    try:
        if use_native:
            pipeline = SessionPipeline(memory)
            report = pipeline.process_session_native(
                campaign_id=args.campaign,
                session_id=args.session,
                transcript_path=args.transcript,
                game_name=args.game,
                session_date=session_date,
                session_context=session_context,
            )
        else:
            pipeline = SessionPipeline(memory, RoleRegistry())
            report = pipeline.process_session(
                campaign_id=args.campaign,
                session_id=args.session,
                transcript_path=args.transcript,
                game_name=args.game,
                session_date=session_date,
                session_context=session_context,
            )
    finally:
        memory.close()
    print(f"mode: {'native (self-contained)' if use_native else 'provider-backed'}")
    print(f"chunks: {report.chunks_processed}/{report.chunks_total} "
          f"(resumed past {report.chunks_skipped_resume})")
    print(f"facts verified: {len(report.facts_verified)}  "
          f"disputed: {len(report.facts_disputed)}  rejected: {report.facts_rejected}")
    print(f"contradictions flagged: {len(report.contradictions)}")
    print(f"review items: {len(report.review_items)}")
    for warning in report.warnings:
        print(f"warning: {warning}")
    if args.summary_out and report.summary_markdown:
        with open(args.summary_out, "w", encoding="utf-8") as f:
            f.write(report.summary_markdown)
        print(f"summary written to {args.summary_out}")
    elif report.summary_markdown:
        print("\n" + report.summary_markdown)
    return 0


def cmd_reviews(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        items = memory.pending_reviews(args.campaign)
    finally:
        memory.close()
    for index, item in enumerate(items, start=1):
        print(f"{index:3d}. [{item.item_type}] {item.item_id}  {item.subject}")
        print(f"     reason: {item.reason}")
        for suggestion in item.suggestions:
            canonical = suggestion.get("canonical", "?")
            print(f"     suggestion: {canonical} "
                  f"(score {suggestion.get('score', '?')}, {suggestion.get('reason', '')})")
    print(f"{len(items)} pending review item(s)")
    print("resolve with: python -m transcripts_ai resolve --campaign <c> "
          "--item <id|number> --action correct|alias|new|not-entity|defer|dont-know ...")
    return 0


def cmd_resolve(args: argparse.Namespace) -> int:
    from .review import ReviewCoordinator
    from .schemas import EntityKind

    memory = _memory(args)
    try:
        items = memory.pending_reviews(args.campaign)
        item = None
        if args.item.isdigit() and 1 <= int(args.item) <= len(items):
            item = items[int(args.item) - 1]
        else:
            item = next((i for i in items if i.item_id == args.item), None)
        if item is None:
            print(f"no pending item {args.item!r}; run `reviews` to list")
            return 1
        actor = f"human:{args.user}"
        coordinator = ReviewCoordinator(memory)
        if args.action == "correct":
            if not args.canonical:
                print("--canonical required for correct")
                return 2
            coordinator.correct(item, canonical=args.canonical, actor=actor,
                                apply_to_all=not args.this_occurrence_only)
            if args.reject_others:
                for suggestion in item.suggestions:
                    other = suggestion.get("canonical")
                    if other and other != args.canonical:
                        coordinator.reject_suggestion(item, canonical=other, actor=actor)
        elif args.action == "alias":
            if not args.canonical:
                print("--canonical required for alias")
                return 2
            coordinator.alias(item, canonical=args.canonical, actor=actor,
                              reason=args.reason or "")
        elif args.action == "new":
            kinds = {k.value: k for k in EntityKind}
            if args.kind not in kinds:
                print(f"--kind must be one of: {', '.join(sorted(kinds))}")
                return 2
            coordinator.new_entity(item, kind=kinds[args.kind], actor=actor,
                                   vault_path=args.vault_path,
                                   description=args.reason or "")
        elif args.action == "not-entity":
            coordinator.not_entity(item, actor=actor)
        elif args.action == "reject":
            if not args.canonical:
                print("--canonical required for reject")
                return 2
            coordinator.reject_suggestion(item, canonical=args.canonical, actor=actor)
        elif args.action == "defer":
            coordinator.save_for_review(item, actor=actor)
        elif args.action == "dont-know":
            coordinator.dont_know_yet(item, reason=args.reason or "other", actor=actor)
        elif args.action == "let-ai-pick":
            recommendation = coordinator.let_ai_pick(item)
            print(f"AI suggestion (NOT applied): {recommendation.action.value}"
                  + (f' -> "{recommendation.choice}"' if recommendation.choice else "")
                  + f"  confidence {recommendation.confidence:.2f}")
            print(f"  why: {recommendation.reason}")
            print("  apply it yourself with --action correct/alias if you agree")
            return 0
        print(f"{args.action}: {item.subject}"
              + (f" -> {args.canonical}" if args.canonical else ""))
    finally:
        memory.close()
    return 0


def cmd_entities(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        entities = memory.entities(args.campaign)
        needle = (args.query or "").casefold()
        for entity in entities:
            if needle and needle not in entity.name.casefold():
                continue
            path = f"  -> {entity.vault_path}" if entity.vault_path else ""
            print(f"[{entity.kind.value:12s}] {entity.name}  ({entity.status.value}){path}")
        aliases = memory.aliases(args.campaign)
        if aliases and not needle:
            print("\naliases:")
            for alias in aliases:
                print(f'  "{alias.observed}" -> "{alias.canonical}"')
    finally:
        memory.close()
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        facts = (
            memory.search_facts(args.campaign, args.query)
            if args.query
            else memory.facts_for_session(args.campaign, args.session or "")
        )
    finally:
        memory.close()
    for fact in facts:
        print(f"{fact.fact_id}  [{fact.status.value}]  {fact.statement}")
        print(f"    evidence: \"{fact.provenance.quote}\" "
              f"({fact.provenance.source_path}:{fact.provenance.line_start})")
    print(f"{len(facts)} fact(s)")
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        removed = memory.forget_fact(
            args.campaign, args.fact_id, actor=f"human:{args.user}", reason=args.reason
        )
    finally:
        memory.close()
    print("removed" if removed else "not found")
    return 0 if removed else 1


def cmd_export(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        data = memory.export_verified_dataset(args.campaign)
    finally:
        memory.close()
    json.dump(data, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


def cmd_spellcheck(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .spellcheck import SpellChecker
    from .transcript import parse_transcript

    path = Path(args.transcript)
    name = path.name.casefold()
    if "unmapped" in name or "mapped" not in name:
        print("REFUSED: spelling corrections apply only to Mapped transcripts; "
              "Transcript Unmapped is never modified.")
        return 3
    memory = _memory(args)
    try:
        checker = SpellChecker(memory)
        parsed = parse_transcript(path.read_text(encoding="utf-8-sig"),
                                  source_path=str(path))
        corrections = checker.find_corrections(
            parsed, args.campaign, min_confidence=args.min_confidence
        )
        for c in corrections:
            print(f"line {c.line_number:5d}: {c.original} -> {c.corrected} "
                  f"(confidence {c.confidence:.2f})")
        if not corrections:
            print("no ordinary-word corrections found")
            return 0
        if args.apply:
            applied = checker.apply_corrections(
                path, corrections, campaign_id=args.campaign,
                session_id=args.session or path.stem,
            )
            print(f"applied {applied} correction(s) to {path.name}; "
                  "audited as auto_spell_correction")
        else:
            print(f"{len(corrections)} correction(s) found (dry run — "
                  "pass --apply to write them)")
    finally:
        memory.close()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from .learner import SuggestionLearner

    memory = _memory(args)
    try:
        learner = SuggestionLearner(memory)
        weights = learner.train(args.campaign)
    finally:
        memory.close()
    if weights is None:
        print("not enough human feedback yet (need >= 8 decisions with both "
              "accepted and rejected examples); using default weights")
        return 0
    print("learner re-trained; weights:")
    for feature, value in weights.items():
        print(f"  {feature:22s} {value:+.3f}")
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    memory = _memory(args)
    try:
        entries = memory.audit_entries(args.campaign, limit=args.limit)
    finally:
        memory.close()
    for entry in entries:
        print(f"{entry['created_at']:.0f}  {entry['action']:24s} {entry['subject']} "
              f"({entry['actor']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcripts_ai")
    parser.add_argument("--db", default="engine_memory.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="process a Mapped transcript end to end")
    p.add_argument("--campaign", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--date", default=None,
                   help="session date; defaults to --session")
    p.add_argument("--summary-out")
    p.add_argument("--mapping", help="player mapping JSON (see mapping.example.json)")
    p.add_argument("--native", action="store_true",
                   help="force the self-contained engine even if AI_ROLE_* is set")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("ingest", help="alias of process (memory building pass)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--date", default=None)
    p.add_argument("--summary-out")
    p.add_argument("--mapping")
    p.add_argument("--native", action="store_true")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("spellcheck",
                       help="auto-correct ordinary-word misspellings in a "
                            "Mapped transcript (entity names untouched)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--transcript", required=True)
    p.add_argument("--session", default=None)
    p.add_argument("--apply", action="store_true",
                   help="write corrections (default: dry run)")
    from .spellcheck import DEFAULT_MIN_CONFIDENCE
    p.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    p.set_defaults(func=cmd_spellcheck)

    p = sub.add_parser("train", help="re-fit the suggestion learner from feedback")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("reviews", help="list pending review items")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_reviews)

    p = sub.add_parser("resolve", help="apply a human decision to a review item")
    p.add_argument("--campaign", required=True)
    p.add_argument("--item", required=True, help="item id or list number")
    p.add_argument("--action", required=True,
                   choices=["correct", "alias", "new", "not-entity", "reject",
                            "defer", "dont-know", "let-ai-pick"])
    p.add_argument("--canonical", help="target name for correct/alias/reject")
    p.add_argument("--kind", help="entity kind for new (pc, npc, location, ...)")
    p.add_argument("--vault-path", help="vault page path for new")
    p.add_argument("--reason", help="free-text reason/description")
    p.add_argument("--user", default="owner")
    p.add_argument("--this-occurrence-only", action="store_true")
    p.add_argument("--reject-others", action="store_true",
                   help="with correct: also mark other suggestions rejected")
    p.set_defaults(func=cmd_resolve)

    p = sub.add_parser("entities", help="list known entities and aliases")
    p.add_argument("--campaign", required=True)
    p.add_argument("--query", help="filter by substring")
    p.set_defaults(func=cmd_entities)

    p = sub.add_parser("facts", help="search or list remembered facts")
    p.add_argument("--campaign", required=True)
    p.add_argument("--query")
    p.add_argument("--session")
    p.set_defaults(func=cmd_facts)

    p = sub.add_parser("forget", help="remove a learned fact (audited)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--fact-id", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--user", default="owner")
    p.set_defaults(func=cmd_forget)

    p = sub.add_parser("export", help="export the human-verified dataset")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("audit", help="show the audit log")
    p.add_argument("--campaign", required=True)
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("vault-sync",
                       help="seed memory with the vault's canonical entities "
                            "and alias lists (one-way, vault is never written)")
    p.add_argument("--vault", required=True,
                   help="campaign folder of the vault (the one containing your "
                        "PC/NPC sections)")
    p.add_argument("--campaign", required=True)
    p.add_argument("--sections", nargs="*",
                   help="vault subfolders to scan (default: PCs, NPCs & "
                        "Locations, Quests, Bestiary)")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be imported; write nothing")
    p.set_defaults(func=cmd_vault_sync)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
