# Transcripts-AI

A campaign-aware D&D transcript intelligence engine: the modular,
provider-independent replacement for the Summurizer bot's Luna/Terra/Sol AI
layer. Pure Python 3.11+, **zero runtime dependencies**, 146 tests.

## Documentation

- [`docs/AUDIT.md`](docs/AUDIT.md) — audit of the current bot: what
  Luna/Terra/Sol actually do, what must be preserved, what the replacement
  corrects.
- [`docs/PLAN.md`](docs/PLAN.md) — staged migration plan into the bot.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — engine design: epistemic
  model, provider roles, retrieval flow, staged evidence system.

## What it does

For each session it runs the staged evidence pipeline — never one big
"find everything" prompt:

1. Load session context (campaign, player→PC mapping with per-session
   overrides, DM identity). Mappings are campaign-scoped and never leak.
2. Parse the Mapped transcript (the Unmapped original is never touched).
3. Segment into scenes (roleplay / travel / combat / loot / planning / rest /
   rules talk / out-of-character) with combat boundaries.
4. Detect candidate entities (rule-based, suppression-biased — capitalisation
   alone is never enough).
5. Resolve names against campaign memory: exact → alias → correction history →
   phonetic/typographic similarity → context, with plain-English explanations.
   Rejected corrections are never re-proposed.
6. Extract facts as structured claims (subject/relationship/object, time
   status, speaker mode, exact quote) via the extractor role.
7. Deterministically validate every quote against the sources — invented
   evidence is rejected before it can be stored.
8. Independently verify each fact (verifier role); disagreement goes to review.
9. Detect contradictions against memory — never overwrite, always link.
10. Summarise from the verified ledger with confirmed/uncertain separation and
    a no-hallucination gate.
11. Queue everything uncertain for human review (Correct / Alias / New /
    Save for Review / Let AI Pick / Don't Know Yet).
12. Learn from decisions: aliases, corrections, feedback, canon promotions —
    all campaign-scoped, provenance-tracked, and reversible (`forget`).

### Safety rules (enforced in code, not by convention)

- AI output can never become `confirmed_canon`; only human decisions can.
- Jokes/OOC lines are capped at `table_talk` whatever the model claims.
- NPC dialogue (DM voicing a character) confirms only what the NPC *claims*.
- Plans, hypotheticals and negated actions never become events.
- Contradictions keep both facts and demand human resolution.
- Every memory write is audited; every learned row can be inspected/removed.
- Interrupted runs resume per-chunk; edited transcripts reprocess only changes.

## Usage

```bash
# Configure roles (any mix of cloud and local):
export AI_ROLE_EXTRACTOR="openai:gpt-5.2-mini"
export AI_ROLE_VERIFIER="openai:gpt-5.2-mini"
export AI_ROLE_SUMMARIZER="openai:gpt-5.2"
export AI_ROLE_REVIEWER="openai:gpt-5.2"
export OPENAI_API_KEY="sk-..."

# Or run fully local (Ollama / llama.cpp server / LM Studio):
export LOCAL_AI_BASE_URL="http://127.0.0.1:11434/v1"
export AI_ROLE_EXTRACTOR="local:qwen3-32b"

# Process a session (memory lives in engine_memory.sqlite):
python -m transcripts_ai process \
  --campaign "silverspire" --session "2026-08-01" \
  --transcript "path/to/2026-08-01 - Game - Transcripts Mapped.md" \
  --game "Heckuva Side Quest" --summary-out summary.md

python -m transcripts_ai reviews --campaign silverspire   # pending review queue
python -m transcripts_ai facts --campaign silverspire --query "moon sickle"
python -m transcripts_ai forget --campaign silverspire --fact-id abc --reason "wrong"
python -m transcripts_ai export --campaign silverspire    # human-verified dataset
python -m transcripts_ai audit --campaign silverspire     # full audit log
```

Confidence bands (configurable): ≥0.95 **and** strong evidence → auto-link to
an existing entity; 0.80–0.94 → suggest; 0.60–0.79 → save for review; below →
drop. New entities are always proposed for review, never auto-created.

## Tests

```bash
python -m pytest tests/ -q   # 146 tests, no network, no API keys
```

## Switching providers

Roles, not models, are configured. `AI_ROLE_<ROLE>="provider:model"` where
provider is `openai`, `local` (any OpenAI-compatible server) or `fake`
(tests). Validation + bounded retry + review-queue routing make weaker local
models safe: they can be wrong, but they cannot write unvalidated data.
