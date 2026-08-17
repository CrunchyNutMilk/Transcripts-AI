# Implementation Plan — Replacing the Sol AI layer with a modular, provider-agnostic pipeline

Companion to `docs/AUDIT.md` (read that first). Verdict up front:

> **Yes, this is possible — and most of it already exists.** The current bot is
> not a monolith with one "Sol" file; it is already a modular pipeline with an
> immutable-transcript contract, a review queue, campaign memory, audit logs,
> and a shadow learning engine. The correct project is therefore NOT a rewrite.
> It is: (1) finish the provider abstraction so every AI call goes through one
> replaceable layer, (2) upgrade the summariser (the weakest module) to the new
> spec, (3) add the genuinely missing pieces (phonetic matching, incremental
> vault index, review follow-up polish), and (4) only then retire the
> Sol/Luna/Terra-specific wiring. Rewriting the working parts would throw away
> hundreds of tests and battle-tested safety code for zero user benefit.

---

## Stage 0 — Ground rules (safety invariants, non-negotiable)

Preserved from the current system; every later stage must keep these true:

1. `Transcript Unmapped` is never written after creation (`"xb"` create,
   hash-verified everywhere). Corrections apply to `Transcript Mapped` only.
2. No vault file is overwritten destructively: atomic writes, marker-delimited
   idempotent update blocks, template-based page creation only for
   human-confirmed (`confirmed_new`) entities.
3. No fact enters a vault page unless it passed the accepted-facts gate
   (human confirm/correct, or verified + reconciled in the final transcript).
4. Confidence below threshold → review queue, never silent auto-correction.
5. Similar names are never auto-merged; alias requires human approval (or the
   exact-match batch path with its strict 100.0-score single-suggestion rule).
6. Every automatic change is auditable (decision tables, before/after JSON,
   marker blocks, learning evidence).
7. All processing stages are resumable (job queue recovery, checkpoints,
   write-ahead edit marker, downstream leases).
8. Existing Discord commands and outputs stay compatible.
9. Campaigns never share memory, indexes, caches, or context.

## Stage 1 — Repo + baseline (half a day)

1. Bring the current bot source into version control (this repo or a private
   repo — decide; the code contains campaign paths but no secrets).
2. Run the existing test suite (`uv run pytest`) on the dev machine and record
   the baseline pass/fail state.
3. Rotate the Discord token + OpenAI key if the public zip's `.env` was real;
   remove the zip from the public Summurizer repo.

**Gate:** tests green (or failures documented) before any change.

## Stage 2 — Unified AI-provider layer (the actual "replace Sol" work)

Create a new package `ai_providers/` with one small interface per capability,
mirroring the pattern `TranscriptionEngine` already proves:

```
ai_providers/
├── base.py          # ChatProvider protocol: complete(request) -> ProviderResponse
│                    #   request: messages, json_schema | pydantic model, effort,
│                    #   max_tokens, cache_key; response: parsed, usage, request_id
├── roles.py         # Role registry: "extractor" | "verifier" | "adjudicator" |
│                    #   "summarizer" | "reviewer" | "answerer" → provider+model
├── openai_provider.py   # wraps today's OpenAI Responses/Chat calls
├── local_provider.py    # OpenAI-compatible local endpoint (Ollama / llama.cpp
│                        #   server / LM Studio all speak this protocol)
├── gemini_provider.py   # port of the existing GeminiLiveFactVerifier transport
└── validation.py    # shared strict-schema validation + bounded retry policy
```

Key decisions:

- **Roles, not models, are the unit of configuration.** Env config becomes:
  `AI_ROLE_EXTRACTOR="openai:gpt-5.6-luna"`,
  `AI_ROLE_VERIFIER="openai:gpt-5.6-terra"`,
  `AI_ROLE_ADJUDICATOR="openai:gpt-5.6-sol"`,
  `AI_ROLE_SUMMARIZER="openai:gpt-5.6-luna"` (upgrade from gpt-4.1-mini),
  `AI_ROLE_REVIEWER`, `AI_ROLE_ANSWERER`. Swapping to a local model later is
  one line: `AI_ROLE_EXTRACTOR="local:qwen3-32b"` — no code change. Existing
  `LIVE_FACT_CHECK_*_MODEL` vars stay supported as fallbacks for compatibility.
- **Keep the existing service protocols** (`LiveFactExtractor`, `LiveFactVerifier`,
  `LiveFactAdjudicator`) — the new providers plug in behind them via the
  constructor injection that already exists. The only service-code changes:
  the adjudicator branch stops hard-coding `OpenAILiveFactAdjudicator`, and
  `transcripts_util.py` gains a `client_factory`/provider parameter like
  `session_ai_review_service` already has.
- **Validation policy** (from `validation.py`, replacing today's no-retry rule
  with a bounded one): parse against schema → on failure, one retry with the
  validation error appended → on second failure, route to review queue exactly
  as today (`needs_human_review` / `ai_completion_status="failed"`). Never
  crash, never write unvalidated output. Local models get the same contract.
- **Usage tracking:** register price entries per (provider, model) in
  `ai_usage_service` (fix the Gemini `google`/`gemini` key mismatch while
  here); local models get zero-cost entries so `/ai_cost` still reports tokens.
- Fix the adjudicator reasoning-effort default inconsistency (`low` vs `medium`).

**Tests:** contract tests run against a `FakeChatProvider`; golden tests assert
byte-identical requests for the OpenAI provider vs today's code (so behaviour
is provably unchanged before any model swap); the existing
`test_sol_routing_policy.py` suite must pass untouched — routing policy is
independent of provider.

**Gate:** full suite green with OpenAI providers active; a smoke session
processed end-to-end produces identical outputs to baseline.

## Stage 3 — Summariser v3 (biggest user-visible upgrade)

Rebuild `helpers/transcripts_util.py` as `services/session_summary_service.py`:

1. **Inputs:** current Mapped transcript (chunked as today), the verified fact
   ledger, the previous 1–2 Mapped transcripts' *summaries* (not full text —
   token budget), and relevant campaign notes located via the entity index for
   entities that appear in this session.
2. **Output schema (structured, validated):** game + session date, party
   members, NPCs/groups, locations, key events (chronological), important
   dialogue/revealed info, decisions, plans, risks & unresolved questions,
   quest transitions (started/updated/completed/abandoned/failed), loot
   transitions (found/received/spent/lost/identified), combat encounters
   (trigger, initiative order *when present in transcript or Roll20 events*,
   notable enemy abilities, outcome/consequences), and **vault-update
   suggestions** (facts that should update existing pages, cross-referenced to
   the fact ledger by Fact ID).
3. **Confirmed vs uncertain:** every section renders confirmed items (backed by
   verified ledger facts or exact quotes) separately from
   uncertain/inferred items; the prompt forbids inventing facts, and a
   deterministic post-pass drops any summary claim naming an entity that
   appears nowhere in transcript/ledger/notes.
4. Markdown rendering keeps the current template's headings as a superset so
   existing vault notes and the website keep working; the fast-path
   (verified-ledger) behaviour is preserved.
5. Keep exact-input caching (cache key includes prompt version + provider+model).

**Tests:** golden-transcript fixtures from both campaigns (redacted samples);
schema-validation tests; a no-hallucination test that plants a decoy name in
the prompt context and asserts it cannot appear as a confirmed fact.

**Gate:** side-by-side comparison of v3 vs current summaries on ≥3 real
sessions per campaign, reviewed by you; false facts = blockers.

## Stage 4 — Fact finder completion

The Luna extraction schema already covers most requested types. Extend, don't
replace:

1. Map the requested taxonomy onto the ledger: PCs/NPCs/Locations/Factions/
   Quests/Loot/Combat/Relationships already exist as `ledger_category`; add
   sub-typing for loot (`weapon`/`armour`/`potion`/`spell-item`) and add
   `deity`/`lore`/`bestiary` categories aligned with the 12 review
   `ENTITY_CATEGORIES` (they aliase already for mentions — unify the mapping
   table in one module).
2. Every fact already stores statement, entities, source window, exact quotes,
   confidence, classification (new/supported/possible_conflict/uncertain), and
   human-review fields. Add: `speaker` on evidence lines (parseable from the
   Mapped format today; store it explicitly), and `vault_target_path` on the
   fact once resolved by the note-update planner (currently resolved late and
   not persisted back).
3. Aliases, character-name changes, deaths/disappearances/status changes:
   covered by `change_type` + relationship facts; add explicit
   `status_change` fact type validation and a review card presentation for it.
4. Keep the DM-statement preference: extraction prompt already prioritises
   in-world statements; add speaker-role weighting (DM lines outrank player
   speculation) to the confidence scorer — the mapped transcript makes DM lines
   identifiable (`DM:` label).

**Gate:** run against 2 historical sessions; compare extracted facts to the
existing fact reports; regressions/false facts fixed before proceeding.

## Stage 5 — Spelling & entity correction upgrades

1. **Phonetic layer:** add Double Metaphone keys (pure-Python, no new heavy
   deps) to `_name_similarity` as an additional bounded signal
   (e.g. phonetic-equal ⇒ score floor 0.88, never auto-apply). "Aragon" vs
   "Daragon" then surfaces as a suggestion with reason
   `phonetic_match + damerau_distance_1`, still requiring context/human
   confirmation per the length-tier thresholds.
2. **Explanations:** suggestion `reason` codes already exist
   (`spelling_correction`, `confirmed_alias`, `damerau_distance_1`,
   `consonant_skeleton`, …); render them as plain-English explanations on both
   Discord cards and the website (small formatting layer, no logic change).
3. **Filler-word safety:** the suppression lexicon + `ENTITY_NAME_*` env lists
   already handle "Ah"/"I've"; add the phonetic layer *behind* the existing
   `detect_name_use` gate so phonetics can never resurrect a suppressed token.
4. **Learning:** approved corrections already persist to
   `campaign_entity_glossary` and alias frontmatter, and feed later sessions'
   matching. No change needed beyond tests proving the loop.
5. Low-confidence corrections remain review-only (existing thresholds).

**Gate:** fixture tests for phonetic pairs from your real campaigns; zero new
auto-applied corrections below threshold.

## Stage 6 — Review workflow polish

1. Add "Don't Know Yet" (with the existing 7 reason codes) to the *name* review
   card and website form, distinct from `defer` (Save for Later) — it lands in
   the campaign Unknown/Review area without touching evidence, as the fact
   checker version already does.
2. After Correct/Alias/New, ask only the relevant follow-ups (entity type →
   vault folder → link target → this-occurrence-vs-all). Most of these forms
   exist (`CategorySelectView`, `CorrectSpellingModal`, `OccurrenceDecisionModal`,
   `CampaignMatchSearchModal` search bar); the change is sequencing them into a
   single guided flow instead of a secondary menu.
3. Keep the search-modal pattern everywhere a vault entry is selected (already
   the case; Discord's 25-option select limit makes this mandatory anyway).

**Gate:** manual walkthrough of each action path on a test session, plus the
existing session-review test suite extended for the new flow.

## Stage 7 — Campaign memory & incremental vault index

1. Add a persistent `campaign_entity_index` SQLite table (per guild+campaign):
   path, mtime, size, content hash, parsed entity JSON. `build_campaign_entity_index`
   consults it and re-parses only new/changed/deleted files, with a
   `--rebuild` escape hatch. (Transcript chunk indexing already works this way.)
2. Wire the (currently unused) FTS5 chunk search into the summariser/context
   loader for "relevant previous transcript" retrieval, replacing
   whole-transcript context where budget-constrained.
3. Memory provenance already exists (glossary rows carry reviewer, session,
   source); add a small admin command to list/remove a bad learned correction.

**Gate:** index correctness test (edit/add/delete files → index converges to a
full-rebuild result); performance check on your real vault.

## Stage 8 — Integration, comparison, cleanup

1. Run the full pipeline on real recordings from **both** campaigns; compare
   Mapped transcripts, fact reports, summaries, and vault updates against
   the existing outputs. Fix false positives / missed facts / unsafe
   corrections. Iterate until you approve.
2. Only now remove obsolete code: the old `transcripts_util` summariser,
   direct `OpenAI()` constructions, dead code paths (`_run_sol_wider_pass` if
   still unreachable, unused legacy config). "Sol/Luna/Terra" remain as *role
   names* in config defaults but nothing depends on OpenAI specifically.
3. Update README, `.env example` (new `AI_ROLE_*` block with comments), and the
   learning-engine docs (its Luna/Terra/Sol actor taxonomy gains a
   provider-agnostic note).

---

## Switching providers/models later

- Cloud → other cloud: change the `AI_ROLE_*` value; add a price entry via
  `AI_MODEL_PRICING_JSON` if not built in.
- Cloud → local: run any OpenAI-compatible server (Ollama `ollama serve`,
  llama.cpp `llama-server`, LM Studio), set
  `LOCAL_AI_BASE_URL="http://127.0.0.1:11434/v1"` and
  `AI_ROLE_<role>="local:<model>"`. Structured-output validation + bounded
  retry + review-queue fallback make weaker local models safe: they can be
  wrong, but they cannot write unvalidated data.
- Recommended first local experiments: summariser and extractor roles
  (high volume, verified downstream). Keep the adjudicator and reviewer roles
  on the strongest model until shadow comparisons say otherwise — the learning
  engine's comparison machinery is exactly the tool for judging a swap.

## Effort estimate (working on the PC, one stage at a time)

| Stage | Size |
|---|---|
| 1 Repo + baseline | 0.5 day |
| 2 Provider layer | 2–3 days |
| 3 Summariser v3 | 2–3 days |
| 4 Fact finder | 1–2 days |
| 5 Spelling/phonetics | 1 day |
| 6 Review polish | 1–2 days |
| 7 Memory/index | 1 day |
| 8 Integration + cleanup | 2–3 days (mostly your review time) |

## Known limitations to expect (honest accuracy notes)

- Phonetic + fuzzy matching will still miss corrections that need world
  knowledge; the review queue is the backstop, by design.
- Initiative order is only extractable when it is actually spoken or present in
  Roll20 events; the summariser must say "not stated" rather than guess.
- Local models will produce more schema-validation retries and more
  review-queue items than GPT-5.6-class models; that is the safe failure mode.
- Whisper transcription errors upstream bound everything downstream; the
  Mapped-transcript correction loop is the mitigation, not the extractor.

## Open questions for you

1. Should the current bot source move into this repo (Transcripts-AI) so
   implementation happens here, or into a new private repo? (The website code
   would follow the same decision.)
2. Which local model/hardware do you plan to target first (the RX 7900 XT
   noted in `.env example` suggests Vulkan-friendly llama.cpp builds)?
3. Are the `gpt-5.6-luna/terra/sol` defaults the models you want to keep for
   cloud mode during the transition?
