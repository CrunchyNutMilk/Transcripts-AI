"""Command-line interface for the engine.

    python -m transcripts_ai process --campaign camp-a --session 2026-08-01 \
        --transcript "path/to/... - Transcripts Mapped.md" --game "Heckuva Side Quest"
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


def cmd_process(args: argparse.Namespace) -> int:
    import os

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
                session_date=args.date or args.session,
            )
        else:
            pipeline = SessionPipeline(memory, RoleRegistry())
            report = pipeline.process_session(
                campaign_id=args.campaign,
                session_id=args.session,
                transcript_path=args.transcript,
                game_name=args.game,
                session_date=args.date or args.session,
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
    for item in items:
        print(f"[{item.item_type}] {item.item_id}  {item.subject}")
        print(f"    reason: {item.reason}")
        for suggestion in item.suggestions:
            print(f"    suggestion: {suggestion}")
    print(f"{len(items)} pending review item(s)")
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
    p.add_argument("--date", default=str(date.today()))
    p.add_argument("--summary-out")
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
    p.add_argument("--native", action="store_true")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("train", help="re-fit the suggestion learner from feedback")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("reviews", help="list pending review items")
    p.add_argument("--campaign", required=True)
    p.set_defaults(func=cmd_reviews)

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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
