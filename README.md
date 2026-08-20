# Transcripts-AI

A campaign-aware D&D transcript intelligence engine: the self-contained
replacement for the Summurizer Discord bot's AI layer. Pure Python 3.11+,
**zero runtime dependencies**, fully offline — **no outside AI required**.
Tests are the source of truth (`python -m pytest tests -q`); CI runs the
suite on every push (Python 3.11 and 3.12).

This README is the project's memory: it carries the full story, decisions and
current state, so a fresh Claude session (or a returning human) can pick the
project up from the repo alone.

---

## The story so far

1. **The starting point.** The Summurizer bot processes D&D sessions:
   Discord voice → whisper.cpp transcription → speaker-mapped transcripts →
   summaries, fact-checking and an Obsidian campaign vault. Its AI layer was
   three OpenAI model tiers nicknamed **Luna** (high-volume fact extraction
   per 5-minute window), **Terra** (independent verification) and **Sol**
   (adjudicator for disagreements, review advisor, website Q&A).
   `docs/AUDIT.md` documents that system in detail — including the parts
   worth keeping (immutable Unmapped transcripts, hash-guarded edits,
   resumability, cost tracking, audit trails).

2. **The goal.** Replace Luna/Terra/Sol with a purpose-built engine that is
   **evidence-based, campaign-specific, explainable, reversible, testable,
   efficient, provider-independent and safe against invented facts** — and
   that runs with **no external AI at all**. Cloud/local LLM providers exist
   only as an optional accelerator behind `AI_ROLE_*` env vars (setting them
   switches the whole pipeline; the default "native" mode needs nothing).

3. **What was built** (`transcripts_ai/`, ~200 tests):
   - **Staged evidence pipeline** — parse → scene segmentation (combat
     boundaries, OOC runs) → candidate detection → name/alias resolution →
     fact extraction → deterministic evidence gate → contradiction detection
     → review queue → campaign memory. Never one big "find everything" step.
   - **Native extractor** — pattern-based facts (introductions, aliases,
     travel, loot transfers, deaths, spell casts, quest transitions, rests,
     damage, initiative order) as subject/relationship/object triples with
     time status (planned vs happened vs negated), speaker mode
     (DM narration vs NPC dialogue vs table talk) and exact-line quotes, so
     evidence is correct by construction.
   - **Native summariser** — extractive, built only from sourced facts;
     empty sections say "None mentioned" instead of guessing.
   - **Campaign memory** (SQLite) — entities, aliases, facts with provenance
     and epistemic status (only humans create `confirmed_canon`),
     contradiction links (never overwrites), feedback ledger, append-only
     audit log, `forget` for reversible learning. Campaigns never share data.
   - **Trainable learner** — transparent logistic scorer over spelling/
     phonetic/context features, re-fit from human accept/reject decisions
     per campaign (`train` command); powers the advisory "Let AI Pick".
   - **Two-lane spelling** — ordinary words ("becuase" → "because") are
     auto-corrected in Mapped transcripts via a bundled 97k-word frequency
     dictionary + curated misspellings table, with guards for possessives,
     playful coinages, repeated unknown terms and anything within one edit
     or a phonetic match of a campaign name. Logged as
     `auto_spell_correction`, never reviewed, never an entity. Possible
     **PCs/NPCs/locations** go to the human review queue with six actions:
     Correct / Alias / New / Save for Review / Let AI Pick / Don't Know Yet.
   - **Player mapping authority** — a `mapping.json` (see
     `mapping.example.json`) is the source of truth for who plays whom;
     with it, PCs are never guessed from speaker frequency, and `Unknown`
     can never become a PC.

4. **Trained on the real campaign.** Sixteen *Heckuva Side Quest* Mapped
   transcripts (Nov 2025 – Jul 2026, ~43,600 speaker turns) were ingested in
   a rehearsal. The engine found the real alias clusters on its own —
   Jenx↔Jinx, Althena↔Althea, Diego↔Diego the Bear, Boblin↔Boblin Thee
   7enth, and Gomra/Gamra/Gamera/Goomra/Gomrad → Ghomra via phonetics — plus
   genuine entities (Silas, Oxwater, Silver Spire, Argon, Vel'Nadar, Sparkle
   Sword, Pearl of Power). Real-data dry-runs also exposed and fixed real
   defects (username-vs-PC confusion, capitalised filler leaking through,
   stuck combat scenes, dictionary false positives like "elven → even").

5. **Hardened.** A preflight review and a full code audit produced ~25 fixes
   with regression tests: crash guards, resume integrity (disputed facts can
   never be laundered into "verified"), upsert data-loss protection,
   word-boundary name matching, exact-casing preservation (Nwen'sua stays
   Nwen'sua), Mapped-only enforcement everywhere, evaluation-only databases
   for unlabelled transcripts, `.gitignore` for campaign data, and CI.

## Current status

- **Branches**: `claude/transcript-ai-pipeline-gc4vpn` (PR #1, the engine)
  and `claude/training-lab` (everything below — the training lab, vault
  sync, Craig support, 5e knowledge). **PR #1 is a draft on purpose** — it
  must not be merged, marked ready, or broadened until the owner completes
  the on-PC run in `docs/RUNBOOK.md` (mapping → ingest → review →
  re-ingest against the real vault).
- **On the PC**: code at `C:\dev\Transcript_AI`, engine state (database,
  mapping, reports) kept outside the repo at `C:\dev\Transcript_AI_data`.
- **Next**: complete the runbook, review the queue (alias pairs first),
  `train`, re-ingest, then process the 2026-08-09 and 2026-08-16 sessions
  once the bot has produced their Mapped transcripts.

## The training lab (branch `claude/training-lab`, 2026-08-20)

The self-improvement machine around the engine, adversarially reviewed
layer by layer (every layer was attacked by a fleet of verifier agents;
dozens of confirmed findings fixed with regression tests). All commands
under `python -m transcripts_ai.lab ...`:

| Layer | Commands | What it does |
|---|---|---|
| 1 Measurement | `export-records` `tally` `split` `make-bank` `score` | Human decisions become versioned training records and scored question banks; session-level train/eval splits (frozen sessions never train); deterministic token scoring; the memory-only baseline every model must beat |
| 2a Teacher panel | `teachers` `panel` | GPT / Claude / Gemini / local Llama vote on pending review items; unanimous verdicts through a deterministic evidence gate bank as training data (teachers can never invent a name past the gate or bank alone); everything else queues pre-answered |
| 2b Overnight loop | `overnight` | One crash-safe command: panel → bank → queue → nightly scorecard → morning report. Checkpoints after every item; resume never re-pays; out-dir locked against double runs |
| 3 Compile | `compile` | Records → SFT chat data + DPO preference pairs (human decisions only) in axolotl/unsloth format, leak-checked splits |
| 4 Train kit + gate | `train-kit` `promote` | Generates the RTX-3080 QLoRA script, Ollama Modelfile and runbook sized to the dataset; `promote` lets a model take an engine role only after beating the baseline on the same bank |
| 5 Proof + growth | `time-travel` `quiz` | Memory-compounding measurement (on the real campaign: 0% cold → 55-77% late-session recognition) and post-game recap trivia whose every answer is a training label |
| Gold transcripts | `whisper-prompt` `gold-score` `names-check` | Seed Whisper with campaign names; score machine vs hand-corrected transcripts (WER + known-name accuracy) and mine every fix as a label; catch mangled official 5e terms |

Engine-side additions on the same branch:

- **`vault-sync`** (`python -m transcripts_ai vault-sync`): seed memory from
  ANY campaign's Obsidian folder — entities, kinds, and frontmatter `aliases`
  lists become human-approved aliases. One command imported 138 entities and
  225 aliases from the real vault; the last session's review queue dropped
  from 5 items to 2, both pre-answered.
- **Official 5e knowledge** (`transcripts_ai/data/dnd5e_names.txt`): ~1,100
  SRD spells/monsters + standard items/deities. Mentions auto-register as
  kind-correct entities (closed vocabulary — nothing to invent), lowercase
  loot like "you'll find a necklace of prayer beads" becomes an evidenced
  fact, and the spellchecker never "fixes" rulebook words.
- **Craig multitrack support** (`--craig`, `scripts/relabel_transcript.py`):
  Discord recordings arrive with one track per speaker, so transcripts come
  out speaker-labelled with no diarization; `mapping.json` turns usernames
  into character names.
- **Campaign-generic by construction**: knowledge lives only in per-campaign
  memory, mapping.json and vault folder; a test proves two campaigns in one
  database can never cross.

## Usage

```bash
# Verify
python -m pytest tests -q

# Build campaign memory from a vault (Mapped transcripts only)
python scripts/ingest_campaign.py --data-dir "<vault>/<campaign>/01 - Transcript Mapped" \
  --db <data>/engine_memory.sqlite --campaign "<campaign name>" \
  --mapping <data>/mapping.json --report <data>/report.md

# Review workflow
python -m transcripts_ai --db <db> reviews --campaign "<name>"
python -m transcripts_ai --db <db> resolve --campaign "<name>" --user <you> \
  --item <id> --action correct|alias|new|not-entity|reject|defer|dont-know|let-ai-pick ...

# Learn from decisions · inspect · undo
python -m transcripts_ai --db <db> train --campaign "<name>"
python -m transcripts_ai --db <db> entities|facts|audit --campaign "<name>"
python -m transcripts_ai --db <db> forget --campaign "<name>" --fact-id <id> --reason "..."

# Process a session end to end (native, no AI keys needed)
python -m transcripts_ai --db <db> process --campaign "<name>" --session 2026-08-09 \
  --game "<name>" --mapping <data>/mapping.json \
  --transcript "path/to/... Transcripts Mapped.md" --summary-out summary.md

# Ordinary-word spelling cleanup (dry run by default; --apply to write)
python -m transcripts_ai --db <db> spellcheck --campaign "<name>" \
  --transcript "path/to/... Transcripts Mapped.md"
```

## Safety rules (enforced in code, not by convention)

- **Transcript Unmapped is never modified** — every writing tool refuses it
  by filename; corrections apply to Transcript Mapped only.
- AI output can never become `confirmed_canon`; only human decisions can.
- Jokes/OOC cap at table-talk; plans and negated actions never become events;
  NPC dialogue records what the NPC *claims*, not world fact.
- Contradictions are linked and kept for human resolution, never overwritten.
- Every automatic action is audited; every learned row is removable.
- Unlabelled (speaker-unknown) transcripts can only enter disposable
  evaluation databases, never permanent campaign memory.
- Campaigns are fully isolated from each other.

## Repository map

| Path | What it is |
|---|---|
| `transcripts_ai/` | The engine (schemas, transcript, scenes, detector, resolver, phonetics, extraction, native_extractor, summarizers, memory, learner, spellcheck, review, pipeline, providers, cli) |
| `transcripts_ai/data/` | Bundled word-frequency list (wordfreq-derived, CC-BY-SA 4.0), D&D lexicon, curated misspellings |
| `scripts/ingest_campaign.py` | Vault-wide memory building pass |
| `scripts/transcribe_session.py` | Local audio → transcript (whisper.cpp server / faster-whisper); evaluation-only processing |
| `tests/` | The full suite — no network, no keys |
| `docs/RUNBOOK.md` | Step-by-step first run on the PC |
| `docs/ARCHITECTURE.md` | Engine design and the staged evidence system |
| `docs/AUDIT.md` | Audit of the original Luna/Terra/Sol system |
| `docs/PLAN.md` | Original migration plan into the bot |
| `mapping.example.json` | Template for the authoritative player mapping |

## Honest limitations

Native mode is precision-first pattern intelligence: it prefers missing a
fact over inventing one and routes uncertainty to human review, so recall on
messy speech is deliberately modest and grows with reviewed sessions. It will
not catch every nuance an LLM would. The spell-checker defaults to dry-run
for the same reason. The migration of the Discord bot itself onto this engine
(shadow mode, then role-by-role takeover per `docs/PLAN.md`) is future work,
gated on real-vault validation.
