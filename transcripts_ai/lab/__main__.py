"""Training-lab CLI.

Layer 1 (measurement):

    python -m transcripts_ai.lab export-records --db <db> --campaign <c> --out records.jsonl
    python -m transcripts_ai.lab tally          --records records.jsonl
    python -m transcripts_ai.lab split          --records records.jsonl --frozen s1,s2
    python -m transcripts_ai.lab make-bank      --db <db> --campaign <c> --out bank.jsonl
    python -m transcripts_ai.lab score          --db <db> --campaign <c> --bank bank.jsonl \
        [--answerer memory|teacher:<name>] [--history history.jsonl] [--report scorecard.md]

Layer 2a (teacher panel; needs TEACHERS env or --teachers):

    python -m transcripts_ai.lab teachers       # who is configured, zero network
    python -m transcripts_ai.lab panel          --db <db> --campaign <c> \
        --out-queue queue.jsonl --out-records banked.jsonl [--limit 25] [--dry-run]

Gold transcripts (hand-corrected sessions as measurement + labels):

    python -m transcripts_ai.lab whisper-prompt --db <db> --campaign <c> [--out prompt.txt]
    python -m transcripts_ai.lab gold-score     --gold perfect.md --hyp machine.md \
        [--db <db> --campaign <c>] [--session <s>] [--out-records r.jsonl] [--report gold.md]

Nothing in the lab writes to campaign memory, transcripts, or the vault.
The panel reads pending review items and writes JSONL files only; review
items stay unresolved until a human resolves them.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..memory import CampaignMemory
from . import compile as compile_mod
from . import gold as gold_mod
from . import overnight as overnight_mod
from . import panel as panel_mod
from . import records as records_mod
from . import scorecard as scorecard_mod
from . import trainkit as trainkit_mod
from .answerers import memory_answerer, teacher_answerer
from .teachers import discover_teachers


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


def _pick_answerer(memory: CampaignMemory, campaign: str, spec: str, env=None):
    """'memory' or 'teacher:<name>' -> (answerer, display name)."""
    if spec == "memory":
        return memory_answerer(memory, campaign), "memory-baseline"
    kind, _, name = spec.partition(":")
    if kind != "teacher" or not name:
        raise SystemExit(f"unknown answerer {spec!r}; use memory or teacher:<name>")
    teachers, skipped = discover_teachers(dict(os.environ if env is None else env))
    for teacher in teachers:
        if teacher.name == name:
            return teacher_answerer(memory, campaign, teacher), \
                f"teacher:{teacher.name}:{teacher.model}"
    available = ", ".join(t.name for t in teachers) or "(none configured)"
    hints = "".join(f"\n  skipped {s.spec}: {s.reason}" for s in skipped)
    raise SystemExit(f"no teacher named {name!r}; available: {available}{hints}")


def cmd_score(args: argparse.Namespace) -> int:
    bank = scorecard_mod.read_bank(args.bank)
    memory = CampaignMemory(args.db)
    try:
        answerer, default_name = _pick_answerer(memory, args.campaign, args.answerer)
        entry = scorecard_mod.run_scorecard(
            bank, answerer, answerer_name=args.answerer_name or default_name
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


def cmd_teachers(args: argparse.Namespace) -> int:
    teachers, skipped = discover_teachers(dict(os.environ), spec=args.teachers)
    if not teachers and not skipped:
        print('no teachers configured. Set TEACHERS, e.g.\n'
              '  TEACHERS="openai:gpt-5-mini,anthropic:claude-sonnet-5,'
              'gemini:gemini-2.5-pro,local:llama3.1"')
        return 1
    for teacher in teachers:
        print(f"ready    {teacher.name:12s} {teacher.model}")
    for skip in skipped:
        print(f"skipped  {skip.spec:24s} {skip.reason}")
    return 0 if teachers else 1


def cmd_panel(args: argparse.Namespace) -> int:
    teachers, skipped = discover_teachers(dict(os.environ), spec=args.teachers)
    for skip in skipped:
        print(f"skipped teacher {skip.spec}: {skip.reason}")
    if not teachers:
        print("no teachers are ready; set TEACHERS (see the `teachers` command)")
        return 1

    memory = CampaignMemory(args.db)
    try:
        items = memory.pending_reviews(args.campaign, session_id=args.session)
        if not items:
            print("no pending review items — nothing to ask the panel about")
            return 0
        if len(items) > args.limit:
            print(f"limiting to first {args.limit} of {len(items)} pending items"
                  " (raise --limit to do more)")
            items = items[: args.limit]

        if args.dry_run:
            known = panel_mod.known_names_for(memory, args.campaign, items[0])
            print(f"DRY RUN — no API calls. {len(items)} item(s) would be sent to: "
                  + ", ".join(f"{t.name} ({t.model})" for t in teachers))
            print("\nFirst item's prompt:\n" + "-" * 60)
            print(panel_mod.PANEL_SYSTEM)
            print(panel_mod.build_panel_prompt(items[0], known))
            return 0

        results = panel_mod.run_panel(memory, args.campaign, items, teachers,
                                      min_votes=args.min_votes)
    finally:
        memory.close()

    banked = panel_mod.records_from_panel(results)
    queued = panel_mod.queue_from_panel(results)
    records_mod.write_records(args.out_records, banked)
    panel_mod.write_queue(args.out_queue, queued)

    report = panel_mod.summarize(results, teachers)
    print(f"{report.total} item(s): {report.bank_accept} banked accept, "
          f"{report.bank_reject} banked reject, {report.queued} queued for you")
    print(f"banked records -> {args.out_records}")
    print(f"morning queue  -> {args.out_queue}")
    for name, usage in report.usage_by_teacher.items():
        spent = ", ".join(f"{k}={v}" for k, v in sorted(usage.items()))
        print(f"usage {name}: {spent}")
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    rows = []
    for path in args.records:
        rows.extend(records_mod.read_records(path))
    if not rows:
        print("no training records found in the given files")
        return 1
    frozen = [s for s in (args.frozen or "").split(",") if s]
    report = compile_mod.compile_dataset(
        rows, out_dir=args.out_dir, frozen=frozen,
        eval_fraction=args.eval_fraction, human_only=args.human_only,
    )
    print(f"{report.total_records} record(s) -> "
          f"SFT {report.sft_train} train / {report.sft_eval} eval, "
          f"DPO {report.dpo_train} train / {report.dpo_eval} eval")
    for task, count in report.by_task.most_common():
        print(f"  {task:18s} {count}")
    if report.unmapped:
        skipped = ", ".join(f"{k}={v}" for k, v in report.unmapped.most_common())
        print(f"not compiled: {skipped}")
    print(f"files + compile_report.md -> {args.out_dir}")
    return 0


def cmd_train_kit(args: argparse.Namespace) -> int:
    plan = trainkit_mod.build_plan(args.dataset, base_model=args.base_model)
    files = trainkit_mod.write_train_kit(plan, args.out_dir,
                                         model_name=args.model_name)
    state = "READY" if plan.ready else "NOT READY"
    print(f"{state}: {plan.sft_train} SFT train / {plan.sft_eval} eval, "
          f"{plan.dpo_train} DPO pairs "
          f"(DPO stage {'on' if plan.run_dpo else 'off'}), "
          f"{plan.epochs} epochs planned")
    for warning in plan.warnings:
        print(f"WARNING: {warning}")
    print(f"wrote {', '.join(files)} -> {args.out_dir}")
    return 0 if plan.ready else 1


def cmd_promote(args: argparse.Namespace) -> int:
    history = scorecard_mod.read_history(args.history)
    verdict = trainkit_mod.promotion_verdict(
        history, candidate=args.candidate, baseline=args.baseline,
        min_lead=args.min_lead,
    )
    print(("PROMOTE" if verdict.promote else "HOLD") + ": "
          + "; ".join(verdict.reasons))
    return 0 if verdict.promote else 1


def cmd_overnight(args: argparse.Namespace) -> int:
    teachers, skipped = discover_teachers(dict(os.environ), spec=args.teachers)
    for skip in skipped:
        print(f"skipped teacher {skip.spec}: {skip.reason}")
    if not teachers:
        print("no teachers are ready; set TEACHERS (see the `teachers` command)")
        return 1
    print("overnight loop: " + ", ".join(f"{t.name} ({t.model})" for t in teachers))

    memory = CampaignMemory(args.db)
    try:
        answerer = None
        if args.bank:
            answerer = memory_answerer(memory, args.campaign)
        report = overnight_mod.run_overnight(
            memory, args.campaign, teachers,
            out_dir=args.out_dir,
            session_id=args.session,
            max_items=args.max_items,
            min_votes=args.min_votes,
            bank_path=args.bank,
            answerer=answerer,
        )
    finally:
        memory.close()

    print(f"\nasked {report.processed_tonight} item(s): "
          f"{report.bank_accept} banked accept, {report.bank_reject} banked "
          f"reject, {report.queued} queued for you")
    if report.remaining_pending:
        print(f"{report.remaining_pending} pending item(s) left for another "
              "night (or raise --max-items)")
    if report.scorecard_line:
        print(f"scorecard: {report.scorecard_line}")
    print(f"morning report -> {report.out_dir}/morning_report.md")
    return 0


def cmd_names_check(args: argparse.Namespace) -> int:
    from ..dnd5e import scan_text

    text = Path(args.file).read_text(encoding="utf-8")
    mentions, suspects = scan_text(text)
    print(f"{sum(mentions.values())} official-name mention(s), "
          f"{len(mentions)} distinct")
    for name, count in mentions.most_common(args.top):
        print(f"  {name:36s} x{count}")
    if suspects:
        print(f"\n{len(suspects)} suspected mangling(s):")
        for s in suspects[: args.top]:
            print(f'  line {s.line:5d}: "{s.heard}" -> {s.official} '
                  f"[{s.category}] ({s.score:.0%})")
    else:
        print("\nno suspected manglings")
    return 0


def cmd_whisper_prompt(args: argparse.Namespace) -> int:
    memory = CampaignMemory(args.db)
    try:
        prompt = gold_mod.whisper_prompt(memory, args.campaign,
                                         max_chars=args.max_chars)
    finally:
        memory.close()
    if args.out:
        Path(args.out).write_text(prompt + "\n", encoding="utf-8")
        print(f"wrote {args.out} ({len(prompt)} chars)")
    else:
        print(prompt)
    return 0


def cmd_gold_score(args: argparse.Namespace) -> int:
    gold_text = Path(args.gold).read_text(encoding="utf-8")
    hyp_text = Path(args.hyp).read_text(encoding="utf-8")

    names = None
    if args.db and args.campaign:
        memory = CampaignMemory(args.db)
        try:
            names = gold_mod.known_names(memory, args.campaign)
        finally:
            memory.close()
    elif args.db or args.campaign:
        print("--db and --campaign go together (they enable name accuracy)")
        return 2

    report = gold_mod.compare_transcripts(hyp_text, gold_text, names=names)
    print(f"gold words: {report.gold_words}   WER: {report.wer:.1%}   "
          f"(sub {report.substitutions} / drop {report.deletions} / "
          f"invent {report.insertions})")
    if names is not None:
        print(f"known-name accuracy: {report.name_accuracy:.1%} "
              f"({report.name_hits}/{report.name_total})")
        for miss in report.name_misses[:10]:
            print(f'  missed {miss.name!r}: heard "{miss.heard or "(dropped)"}" '
                  f"(gold line {miss.gold_line})")

    if args.out_records:
        if not args.campaign or not args.session:
            print("--out-records needs --campaign and --session for provenance")
            return 2
        rows = gold_mod.records_from_gold(report, campaign_id=args.campaign,
                                          session_id=args.session)
        count = records_mod.write_records(args.out_records, rows)
        print(f"mined {count} correction record(s) -> {args.out_records}")
    if args.report:
        Path(args.report).write_text(
            gold_mod.render_markdown(report, session_id=args.session or ""),
            encoding="utf-8")
        print(f"report written to {args.report}")
    return 0


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


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
    p.add_argument("--answerer", default="memory",
                   help="memory (default) or teacher:<name> from TEACHERS")
    p.add_argument("--answerer-name", default="",
                   help="override the name recorded in history/report")
    p.add_argument("--history", help="JSONL file to append this run to")
    p.add_argument("--report", help="write a Markdown report here")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("teachers",
                       help="list configured teachers (no API calls)")
    p.add_argument("--teachers", help="override the TEACHERS env spec")
    p.set_defaults(func=cmd_teachers)

    p = sub.add_parser("panel",
                       help="ask the teacher panel about pending review items")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--session", help="only items from this session")
    p.add_argument("--limit", type=_positive_int,
                   default=panel_mod.DEFAULT_ITEM_LIMIT,
                   help="max items per run (cost guard, minimum 1)")
    p.add_argument("--min-votes", type=_positive_int,
                   default=panel_mod.MIN_VOTES_TO_BANK,
                   help="teacher votes (excluding the engine) required "
                        "before anything banks")
    p.add_argument("--teachers", help="override the TEACHERS env spec")
    p.add_argument("--out-queue", required=True,
                   help="JSONL morning queue (items still needing a human)")
    p.add_argument("--out-records", required=True,
                   help="JSONL banked training records from unanimous verdicts")
    p.add_argument("--dry-run", action="store_true",
                   help="show teachers + first prompt; make no API calls")
    p.set_defaults(func=cmd_panel)

    p = sub.add_parser("compile",
                       help="training records -> SFT + DPO files for QLoRA")
    p.add_argument("--records", action="append", required=True,
                   help="TrainingRecord JSONL (repeat for several files)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frozen", help="comma-separated session ids, always eval")
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--human-only", action="store_true",
                   help="drop teacher-panel records; train on human decisions only")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("train-kit",
                       help="compiled dataset -> unsloth script, Modelfile, runbook")
    p.add_argument("--dataset", required=True,
                   help="compile's --out-dir (sft_*.jsonl / dpo_*.jsonl)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default="heckuva-engine")
    p.add_argument("--base-model", default=trainkit_mod.DEFAULT_BASE_MODEL)
    p.set_defaults(func=cmd_train_kit)

    p = sub.add_parser("promote",
                       help="gate: does the candidate's scorecard earn a role?")
    p.add_argument("--history", required=True)
    p.add_argument("--candidate", required=True,
                   help="answerer name as recorded in the history file")
    p.add_argument("--baseline", default="memory-baseline")
    p.add_argument("--min-lead", type=float, default=0.02,
                   help="accuracy lead over the baseline required to pass")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("overnight",
                       help="run the whole overnight loop: panel -> bank -> "
                            "queue -> scorecard -> morning report")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--out-dir", required=True,
                   help="night's working directory (re-use it to resume)")
    p.add_argument("--session", help="only items from this session")
    p.add_argument("--max-items", type=_positive_int,
                   default=overnight_mod.DEFAULT_MAX_ITEMS,
                   help="hard spend cap on panel items per night")
    p.add_argument("--min-votes", type=_positive_int,
                   default=panel_mod.MIN_VOTES_TO_BANK,
                   help="teacher votes (excluding the engine) needed to bank")
    p.add_argument("--teachers", help="override the TEACHERS env spec")
    p.add_argument("--bank", help="question bank for the nightly scorecard")
    p.set_defaults(func=cmd_overnight)

    p = sub.add_parser("names-check",
                       help="find official 5e names (and manglings) in a transcript")
    p.add_argument("--file", required=True)
    p.add_argument("--top", type=int, default=40)
    p.set_defaults(func=cmd_names_check)

    p = sub.add_parser("whisper-prompt",
                       help="build a Whisper initial_prompt from campaign names")
    p.add_argument("--db", required=True)
    p.add_argument("--campaign", required=True)
    p.add_argument("--max-chars", type=int, default=gold_mod.PROMPT_MAX_CHARS)
    p.add_argument("--out", help="write to a file instead of stdout")
    p.set_defaults(func=cmd_whisper_prompt)

    p = sub.add_parser("gold-score",
                       help="score a machine transcript against a corrected one")
    p.add_argument("--gold", required=True, help="the hand-corrected transcript")
    p.add_argument("--hyp", required=True, help="the machine transcript")
    p.add_argument("--db", help="campaign db (with --campaign: name accuracy)")
    p.add_argument("--campaign")
    p.add_argument("--session", help="session id for mined records/report")
    p.add_argument("--out-records",
                   help="JSONL of mined heard->truth corrections")
    p.add_argument("--report", help="write a Markdown report here")
    p.set_defaults(func=cmd_gold_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
