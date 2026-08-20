# Training & Evaluation Plan — the Overnight Lab

The plan for making the engine's AI better every week while driving two
numbers toward zero. This document is the source of truth for the training
effort; the runbook (`docs/RUNBOOK.md`) covers day-to-day engine use.

## The two product metrics

Everything below exists to move these, and both go on the nightly scorecard:

| Metric | Today | Target |
|---|---|---|
| **$ per game** (cloud AI spend) | ~$7 | cents, then ~$0 |
| **Questions per game** (human review burden) | up to hundreds (front-loaded) | single digits, answerable over one coffee |

First diagnostic before optimising anything: run the bot's `/ai_cost` after a
game to see where the $7 actually goes (suspicion from the audit: a large
slice is cloud diarization, which the bot's guarded local whisper mode can
eliminate with no AI training at all).

## Hardware layout (two machines, one job each)

| Machine | GPU | Role |
|---|---|---|
| **Main PC** | RX 7900 XT (Vulkan) | Production: whisper.cpp transcription, the engine, serving the tuned model via llama.cpp-Vulkan/GGUF |
| **Second PC** | RTX 3080 (CUDA, 10 GB) | Training rig: QLoRA fine-tunes (7–8B models), faster-whisper CUDA re-runs of saved audio, cloud-teacher calls, the overnight loop |

Wiring needs no code: the engine's provider layer takes any base URL, so the
3080 box running Ollama serves the main PC over the LAN:

```powershell
$env:LOCAL_AI_BASE_URL = "http://<3080-pc-ip>:11434/v1"
$env:AI_ROLE_EXTRACTOR = "local:<tuned-model>"
```

Handoff is standard: train on the 3080 (CUDA) → export GGUF → serve on the
7900 XT (Vulkan). Keep Ollama's LAN binding home-network only; never
port-forward it.

## The data flywheel

The after-game checker + human review already produces labelled decisions
(~100 at time of writing, growing every session). That stream is the fuel;
protect it with four rules:

1. **Save the disagreement, not just the answer** — each record keeps the
   model's proposal, the human's decision, and the evidence (session, line,
   quote). Rejections are as valuable as accepts (hard negatives).
2. **Version the record schema now** — one JSON-lines format
   (`session, item_type, proposal, human_decision, evidence_quote,
   checker_version, timestamp`), backfill the existing archive into it.
3. **Split train/test by session, never by item** — facts repeat across a
   campaign; item-level splits leak and flatter every score.
4. **Tally by category** — 100 answers may be 80 spelling / 20 facts; know
   which capability actually has data.

### Milestones

| Labels | What unlocks |
|---|---|
| **100 (now)** | Engine learner retraining (`train` command); DPO preference pairs (accepted-vs-rejected suggestions — the most sample-efficient use of small data) |
| **~250** | Per-category learners; real curves on the scorecard |
| **~500** | First QLoRA fine-tune with an honest session-level split (this is also the bot's own learning-engine collection target) |
| **~1000** | Proper frozen test set *and* a well-fed training set |

## The overnight loop (runs on the 3080 PC)

Per round, per session:

1. Run the **native engine**, the **local Llama**, and the **cloud teachers**
   (GPT / Claude / Gemini) over the same transcript.
2. Where all agree **and** the deterministic evidence gate passes (the quote
   exists verbatim in the transcript) → bank a training example.
   Fabrications die at the gate regardless of teacher confidence.
3. Where they **disagree** → that item is the morning's question, pre-answered
   with each model's pick so human review is accept/reject at ~2 s/item
   instead of cold answering at ~30 s/item. Disagreement is the signal.
4. Retrain the engine learner on the ledger; periodically DPO/QLoRA the Llama
   on the banked corpus.
5. Run the **nightly scorecard** (below). If the curve isn't bending by round
   three, the scorecard says which component is stalling.
6. Guardrails: a **hard spend cap** for teacher calls (route teachers only at
   disagreements — cost scales with uncertainty, not transcript length), and
   **checkpointing** so a 3 a.m. crash loses nothing.

Non-negotiables:
- **Teacher verdicts never become canon.** Teacher agreement caps at
  `strongly_supported`; only human decisions create `confirmed_canon`.
  (The bot's own learning engine draws the same line for the same reason:
  teachers are wrong in correlated ways.)
- **A frozen evaluation set exists before the loop first runs.** Otherwise
  "getting better" is the loop grading its own homework.

### Implemented so far (`python -m transcripts_ai.lab`)

Layer 1 (measurement): `export-records`, `tally`, `split`, `make-bank`,
`score`. Layer 2a (the teacher panel):

```powershell
# who is on the panel (no API calls)
$env:TEACHERS = "openai:gpt-5-mini,anthropic:claude-sonnet-5,gemini:gemini-2.5-pro,local:llama3.1"
python -m transcripts_ai.lab teachers

# preview the exact prompt the panel would send (no API calls)
python -m transcripts_ai.lab panel --db campaign.sqlite --campaign heckuva `
    --out-queue queue.jsonl --out-records banked.jsonl --dry-run

# run the panel over pending review items (default cap: 25 items/run)
python -m transcripts_ai.lab panel --db campaign.sqlite --campaign heckuva `
    --out-queue queue.jsonl --out-records banked.jsonl

# score any teacher on the same bank as the memory baseline
python -m transcripts_ai.lab score --db campaign.sqlite --campaign heckuva `
    --bank bank.jsonl --answerer teacher:anthropic --history history.jsonl
```

Panel mechanics as built: each pending review item gets one vote from the
engine's resolver (free, deterministic) plus one from every configured
teacher. Unanimous accept on the same name **and** an evidence-gate pass
(the name must already exist in campaign memory; agreeing on an invented
name is still an invented name) → banked to `banked.jsonl` with
`actor="panel:…"` so it can never be mistaken for a human decision.
Unanimous reject → banked hard negative. Anything else — a split, an
uncertain vote, an abstention leaving fewer than two votes, a gate
failure — lands in `queue.jsonl` pre-answered with every opinion. The
panel never writes to campaign memory; review items stay yours to
resolve. Missing API keys just shrink the panel, and a teacher that
errors or returns garbage becomes an abstention, never a crash.

## Question-based testing

- **Post-session quiz** — generate ~10 questions from extracted facts, post
  to Discord as recap trivia. Players enjoy it; every answer is a free gold
  label; the DM vetoes bad premises in seconds. Converts the labelling
  bottleneck into a game.
- **Open-book vs closed-book** — same questions with and without retrieval
  from campaign memory. High open-book + low closed-book = the engine works
  and the model barely matters (worth knowing before training).
- **Trick questions** — unanswerable and false-premise probes ("When did
  Boblin die?"). Correct answer: *not in the record*. Directly measures the
  no-invention property.
- **Checker archive as question bank** — every saved "checker proposed X,
  human said Y" converts mechanically into a scored question. The existing
  archive seeds the scorecard immediately.

## Automated tests (no humans required)

- **Time-travel test** — build memory from sessions 1..N, process session
  N+1 cold: are returning NPCs recognised? Are *new* name variants resolved
  from phonetics + learned aliases? The resulting curve across all sessions
  is the direct measurement of "memory compounds".
- **Corruption ladder** — degrade a gold transcript with the campaign's real
  Whisper manglings at 5/10/20% and measure extraction decay. Says how much
  a transcription improvement is worth vs an extraction improvement.
- **Held-out alias gauntlet** — train with one known variant (e.g. Gomrad)
  excluded; test whether phonetics + learner still route it to Ghomra.
  Rotate the held-out variant. Tests generalisation, not memorisation.
- **Contradiction planting** — edit a copy of a transcript (item to a
  different PC; a dead NPC speaks); the contradiction detector must fire.

## Perfect transcripts (gold data)

> **Implemented:** `python -m transcripts_ai.lab whisper-prompt` (seed
> Whisper with campaign names before transcribing) and
> `python -m transcripts_ai.lab gold-score` (WER + known-name accuracy +
> every hand-fix mined as a training record). Step-by-step:
> docs/RUNBOOK.md §7b.

Five saved games of audio → gold transcripts, built the cheap way:

1. **Start from the reviewed Mapped transcript**, not from scratch — listen
   only where the layers (audio / Whisper raw / Mapped) disagree.
2. **Annotate one session fully before doing five** — the format will need
   fixing after contact with reality (facts spanning lines, jokes, DM
   hedges). Then scale.
3. Annotation carries two things: entity mentions with type, and a per-scene
   block of the facts a correct system should extract (what precision/recall
   are computed against).
4. **Split: 3 development / 2 frozen forever.** The frozen pair is only ever
   used to report final numbers; tune against it once and it measures
   nothing.

Score three layers separately with the same gold data — a single end-to-end
number can't distinguish a Whisper mangle from an extractor miss:

| Layer | Input → output | Metric |
|---|---|---|
| Transcription | audio → text | word error rate, **name error rate** |
| Extraction | gold text → facts | precision / recall vs annotation |
| End-to-end | audio → facts | the number that matters day-to-day |

## The cheapest big win: seed Whisper with campaign names

Whisper accepts an initial prompt; the bot already sends a generic one. Fill
it from the approved-entity list (Ghomra, Jinx, Boblin, Silverspire,
Oxwater, Vel'Nadar, …) so names stop being misheard at the source. Saved
audio makes it measurable: re-run one session as-is, count name errors
against the Mapped transcript; re-run with the seeded prompt; compare. If
the Gomra/Gamra/Gamera cluster collapses, every future review queue shrinks
*upstream* of the engine. Do this before any model training — zero risk,
one afternoon, attacks both product metrics at once.

## Cost elimination map

| Paid today | Job | $0 replacement |
|---|---|---|
| Luna (per 5-min window) | fact extraction | native patterns → local Llama |
| Terra (per fact) | verification | deterministic evidence gate + contradiction check |
| Sol (15-min batches) | adjudication | review queue, or cents of teacher on disputes only |
| Sol (spell-check plans / AI picks) | review advice | trained learner's let-ai-pick |
| Cloud diarization (if enabled) | speakered transcript | bot's guarded local whisper mode |

Structural point: the bot pays per token read; the replacement pays per
*disagreement*. Uncertainty shrinks as memory grows, so spend trends to zero
by design.

## Shrinking the review queue

The queue is front-loaded debt: campaigns stop introducing new entities, and
an approved alias never asks again. Compression mechanisms, in build order:

1. **Family grouping** — one decision resolves a whole variant cluster
   (biggest single win; partially built).
2. **Overnight pre-answering** — morning review is accept/reject, not cold
   answering.
3. **Question budget** — surface the top ~25 items by information value
   (uncertainty × frequency); auto-defer the tail.
4. **Batch-confirm** — one command applying every single-suggestion item at
   ≥0.95, like the bot's "Approve Known Names" button.

## Order of operations

1. `/ai_cost` after next game → find where the $7 goes.
2. Whisper name-seeding experiment on one saved session.
3. **Shadow game**: bot runs paid one final time; engine processes the same
   Mapped transcript free; diff the fact lists → the cut-over decision.
4. Clear review queues (family/batch style) → labels for everything below.
5. Retrain learner; build DPO pairs from the archive.
6. First gold transcript → annotation format proven → remaining four.
7. Stand up the overnight loop on the 3080 with the scorecard and spend cap.
8. QLoRA at ~500 labels; serve the GGUF on the main PC via the provider layer.
