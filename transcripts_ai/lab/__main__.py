"""Training-lab CLI (Layer 1).

    python -m transcripts_ai.lab export-records --db <db> --campaign <c> --out records.jsonl
    python -m transcripts_ai.lab tally          --records records.jsonl
    python -m transcripts_ai.lab split          --records records.jsonl --frozen s1,s2
    python -m transcripts_ai.lab make-bank      --db <db> --campaign <c> --out bank.jsonl
    python -m transcripts_ai.lab score          --db <db> --campaign <c> --bank bank.jsonl \
        [--history scorecard_history.jsonl] [--report scorecard.md]

Layer 1 is measurement only: nothing here writes to campaign memory,
transcripts, or the vault.
"""
from __future__ import annotations

import argparse

from ..memory import CampaignMemory
from . import records as records_mod
from . import scorecard as scorecard_mod
from .answerers import memory_answerer


def cmd_export_records(args: argparse.Namespace) -> int:
    memory = CampaignMemory(args.db)
    try:
        rows = records_mod.export_from_feedback(memory, args.campaign)
    finally:
        memory.close()
    count = records_mod.write_records(args.out, rows)
    print(f"exported {count} training record(s) to {args.out}")
    return 0


def cmd_tally(args: argparse.Namespace) -> int:
    rows = records_mod.read_records(args.records)
    counts = records_mod.tally(rows)
    print(f"{len(rows)} record(s)")
    for name, counter in counts.items():
        print(f"\n{name}:")
        for key, value in counter.most_common():
            print(f"  {key:24s} {value}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    rows = records_mod.read_records(args.records)
    frozen = [s for s in (args.frozen or "").split(",") if s]
    split = records_mod.split_sessions(
        (r.session_id for r in rows),
        frozen=frozen,
        eval_fraction=args.eval_fraction,
    )
    train, evaluation = records_mod.split_records(rows, split)
    print(f"train sessions ({len(split.train_sessions)}): "
          + ", ".join(split.train_sessions))
    print(f"eval sessions  ({len(split.eval_sessions)}): "
          + ", ".join(split.eval_sessions)
          + (f"   [frozen: {', '.join(split.frozen_sessions)}]"
             if split.frozen_sessions else ""))
    print(f"records: {len(train)} train / {len(evaluation)} eval")
    if args.out_train:
        records_mod.write_records(args.out_train, train)
        print(f"wrote {args.out_train}")
    if args.out_eval:
        records_mod.write_records(args.out_eval, evaluation)
        print(f"wrote {args.out_eval}")
    return 0


def cmd_make_bank(args: argparse.Namespace) -> int:
    memory = CampaignMemory(args.db)
    try:
        questions = scorecard_mod.questions_from_feedback(memory, args.campaign)
    finally:
        memory.close()
    if not questions:
        print("no human decisions found to build questions from")
        return 1
    count = scorecard_mod.write_bank(args.out, questions)
    by_type = {}
    for question in questions:
        by_type[question.qtype] = by_type.get(question.qtype, 0) + 1
    print(f"wrote {count} question(s) to {args.out}  "
          + "  ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    bank = scorecard_mod.read_bank(args.bank)
    memory = CampaignMemory(args.db)
    try:
        answerer = memory_answerer(memory, args.campaign)
        entry = scorecard_mod.run_scorecard(
            bank, answerer, answerer_name=args.answerer_name
        )
    finally:
        memory.close()
    history = scorecard_mod.read_history(args.history) if args.history else []
    print(f"{entry.answerer}: {entry.correct}/{entry.total} ({entry.accuracy:.0%})")
    for qtype, stats in sorted(entry.by_type.items()):
        share = stats["correct"] / stats["total"] if stats["total"] else 0.0
        print(f"  {qtype:6s} {stats['correct']}/{stats['total']} ({share:.0%})")
    if args.history:
        scorecard_mod.append_history(args.history, entry)
        print(f"appended to {args.history}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(scorecard_mod.render_markdown(entry, history))
        print(f"report written to {args.report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="transcripts_ai.lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export-records",
                       help="feedback ledger -> versioned training records")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_records)

    p = sub.add_parser("tally", help="distribution audit of a record archive")
    p.add_argument("--records", required=True)
    p.set_defaults(func=cmd_tally)

    p = sub.add_parser("split", help="deterministic session-level train/eval split")
    p.add_argument("--records", required=True)
    p.add_argument("--frozen", help="comma-separated session ids, always eval")
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--out-train")
    p.add_argument("--out-eval")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("make-bank",
                       help="human-decision archive -> scored question bank")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_make_bank)

    p = sub.add_parser("score", help="run a question bank against an answerer")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--answerer-name", default="memory-baseline")
    p.add_argument("--history", help="JSONL file to append this run to")
    p.add_argument("--report", help="write a Markdown report here")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
