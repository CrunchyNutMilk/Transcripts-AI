# Current System Audit — "Sol" and the Summurizer AI layer

Audit date: 2026-08-17. Source: `SummurizerBotCode20260814131040.zip` (the current
bot) and `SummurizerWebsiteCode20260814131040.zip`. The public GitHub
`CrunchyNutMilk/Summurizer` repo is a much older (Oct 2025) prototype and was
not used for this audit beyond confirming it is stale.

---

## 1. What "Sol" actually is

Sol is **not a single AI component**. It is the top tier of a three-model
OpenAI pipeline, all authenticated with one `OPENAI_API_KEY`:

| Name | Model string | Role | Cost (in/cached/out per M tokens) |
|---|---|---|---|
| Luna | `gpt-5.6-luna` | High-volume fact **extractor** — runs on every completed 5-minute transcript window | $1.00 / $0.10 / $6.00 |
| Terra | `gpt-5.6-terra` | Independent **verifier** of every Luna fact | $2.50 / $0.25 / $15.00 |
| Sol | `gpt-5.6-sol` | **Adjudicator** — called only on disagreement/uncertainty/invalid evidence, batched every 3 windows (15 min), max 40 facts/request | $5.00 / $0.50 / $30.00 |

Sol has exactly **three production call sites**:

1. **Live fact adjudication** — `services/live_fact_check_service.py`
   (`OpenAILiveFactAdjudicator`, ~line 1662). Receives compact evidence packets
   (exact quotes + IDs, never full transcripts in production mode). Routing
   policy (`_requires_sol_adjudication`, ~line 4956): escalate only when Terra's
   verdict is not `supported`, evidence is missing, or Luna classified the fact
   as `possible_conflict`/`uncertain`. Importance alone never calls Sol.
   `request_wider_context` never buys a second call — it routes to human review.
2. **Session Name Review advisory** — `services/session_ai_review_service.py`
   (model default `gpt-5.6-sol`). Two modes: `plan()` (batch "Let AI Finish")
   and `recommend_candidate()` ("Ask AI for a suggestion" — display-only, human
   must re-submit the action).
3. **Campaign website Q&A** — `services/campaign_ai_search_service.py`
   (`text-embedding-3-large` retrieval + `gpt-5.6-sol` answers with citations).

Other AI in the system (not named Sol, but part of the AI layer):

- **Session summaries** — `helpers/transcripts_util.py`, hard-coded
  `gpt-4.1-mini`, two-stage map-reduce (12,000-char chunks → synthesis), with a
  fast path that skips chunk summarisation entirely when the live fact-check
  ledger is fully verified.
- **Cloud transcription/diarization** — `gpt-4o-transcribe-diarize`
  (`services/diarization_service.py`).
- **Local transcription** — whisper.cpp server (`ggml-large-v3-turbo`, Vulkan,
  Silero VAD) behind a real `TranscriptionEngine` ABC with quality gates and
  guaranteed cloud fallback.
- **Optional Gemini verifier** — `GeminiLiveFactVerifier` exists as a complete
  second-provider implementation of the verifier protocol (dormant unless
  `LIVE_FACT_CHECK_VERIFIER_PROVIDER=gemini`).

## 2. Existing strengths worth preserving (do NOT rewrite these)

- **Immutable Unmapped transcripts.** Seven independent defence layers: `"xb"`
  exclusive create + byte-verify readback, editor with no write path to
  Unmapped, filename-kind validation, immutable hash checked four times per
  edit (`hmac.compare_digest`), atomic same-dir temp + `os.replace`,
  restart-time re-hash, path containment/symlink rejection.
- **Write-ahead edit marker.** Mapped edits persist the expected result hash
  *before* applying, so an interrupted edit is detected and recovered on
  restart instead of corrupting review state.
- **Resumability.** Six mechanisms: progressive job queue with stale-claim
  recovery, per-chunk cloud transcription checkpoints keyed by audio hash +
  request fingerprint, the write-ahead marker above, downstream release leases,
  persistent Discord review views re-attached after restart, and an
  unfinished-session gate on `/start_recording`.
- **Structured output validation, fail-closed.** Sol uses strict JSON schema +
  ~15 application-side re-validation rules (unknown fact IDs, evidence-ID
  allow-lists, atomic-claim coverage). Any failure → `needs_human_review`,
  never a crash, never silent trust. Billing is recorded before parsing so
  malformed paid responses are still counted.
- **Campaign isolation.** Vault paths per guild with no legacy fallback,
  campaign folders must be direct children of the validated vault root,
  transcript memory keyed `UNIQUE(guild_id, mapped_transcript_path_key)`,
  learning scope requires (guild, campaign, session, channel), Roll20 scope
  IDs are salted hashes of the binding.
- **Audit trails everywhere.** `session_entity_decisions` stores full
  before/after JSON per decision; interaction-ID dedupe tables; campaign pages
  get marker-delimited, idempotent `SUMMURIZER_SESSION_*` update blocks with
  per-fact evidence quotes and Fact IDs; append-only learning-engine tables.
- **Cost tracking.** `ai_usage_service` stores a price snapshot per request
  (history never repriced), idempotent on `(provider, request_id)`, no prompt
  text ever stored, per-feature/stage reporting via `/ai_cost`, Sol
  resolution-rate reporting.
- **Exact-input AI caching.** SHA-256 exact-key caches for Sol decisions,
  summaries, AI picks, and Q&A answers. Fuzzy matching is deliberately never a
  cache key.
- **The review workflow.** Discord dashboard + per-candidate cards + website
  review with related-variant grouping, occurrence-level decisions, optimistic
  locking (`state_version`), undo, deferred queue, mandatory rescan passes
  after edits (max 3), and completion blocked while facts/pictures are pending.
- **Learning engine (shadow mode).** Append-only evidence store, human-only
  gold labels, Luna/Terra/Sol comparison tracking, campaign-scoped exports,
  privacy gates on Roll20 evidence, backfill CLI that opens the operational DB
  read-only.

## 3. Component map — requested vs existing

| Requested component | Existing implementation | State |
|---|---|---|
| Transcript cleaner & chunker | `transcript_chunk_service` (4k/6k/1.5k char bounds, chunker v2, exact-reconstruction validation, FTS5 index) | ✅ exists |
| Campaign-context loader | `transcript_context_service` (loads current + previous Mapped from session records — never directory scans; degrade-not-fail) | ✅ exists |
| Entity detector & classifier | `campaign_entity_service` (rule engine v10, no LLM: direct address, introduction nouns, verb families, suppression lists, offline lexicon 2026.08.11.2) + Luna entity mentions per window | ✅ exists |
| Alias & spelling resolver | `match_candidate` (exact + fuzzy w/ length-tiered thresholds 1.0/0.90/0.80/0.76, containment, single-substitution) + `campaign_entity_glossary` learned relations + alias frontmatter | ✅ exists — **no phonetic matching** (gap) |
| Fact extractor | Luna via `live_fact_check_service` (13 ledger categories, 12 change types, importance, confidence, exact-quote evidence) | ✅ exists |
| Summary generator | `helpers/transcripts_util.py` map-reduce + fixed template | ⚠️ exists but weakest module (see gaps) |
| Contradiction detector | Terra verification + `possible_conflict` classification + Sol adjudication + atomic evidence gate | ✅ exists |
| Confidence scorer | extraction confidence bands (high ≥.82 / med ≥.60), suggestion scores, fact confidence, AI pick confidence floor 0.7 | ✅ exists |
| Review queue | session review store (schema v5) + Discord + website UIs | ✅ exists |
| Campaign memory/index | `transcript_memories` + `transcript_chunks` + FTS5 + glossary + campaign entity index | ✅ exists — **entity index is rebuilt per scan, not incremental** (gap) |
| Vault update planner | `campaign_note_update_service` (accepted-facts gate, exact-match entity resolution, marker-block updates) | ✅ exists |
| Audit log | decision audit tables + vault marker blocks + learning evidence | ✅ exists |
| Replaceable AI provider | Protocols (`LiveFactExtractor/Verifier/Adjudicator`) + DI + Gemini verifier proof | ⚠️ partial — see gaps |

## 4. Genuine gaps found

1. **Provider abstraction is incomplete.**
   - The summary path (`transcripts_util.py`) constructs `OpenAI()` inline
     against a module constant — no injection point at all.
   - The adjudicator branch hard-codes `OpenAILiveFactAdjudicator`; only the
     verifier branch supports `provider=` selection.
   - No local-model (Ollama/llama.cpp/LM Studio) chat provider exists anywhere,
     even though local whisper.cpp proves the pattern works.
2. **No phonetic matching.** Spelling resolution is character-similarity only
   (SequenceMatcher + Damerau-Levenshtein + consonant skeleton on the website
   side). "Aragon"→"Daragon" class errors from speech transcription would
   benefit from Double Metaphone-style keys as an additional (never decisive)
   signal.
3. **Campaign entity index is rebuilt by walking the vault on each use.**
   There is no mtime/hash-based incremental index; the request's "only process
   files that are new or have changed" is not met for the entity index (the
   transcript chunk index *is* idempotent/incremental).
4. **Summary generator shortfalls vs the target spec:**
   - Does not receive previous Mapped transcripts or campaign notes as context
     (the fact checker does; the summarizer does not).
   - Template lacks: initiative order, per-quest status transitions
     (started/updated/completed/abandoned/failed as explicit fields), important
     enemy abilities, "facts that should update existing vault pages",
     explicit confirmed-vs-inferred separation (partially covered by
     "What's Known vs. Unknown" and the verified ledger fast path).
   - `gpt-4.1-mini` is the oldest model in the stack.
5. **Review follow-up flows.** The six requested actions all exist
   (Correct→`use_existing`/`correct_spelling`, Alias→`add_alias`,
   New→`keep_new`, Save for Review→`defer`, Let AI Pick→`ai_pick`,
   Don't Know Yet→`defer` + fact-checker `dont_know` reasons), and the website
   already uses a search modal (`CampaignMatchSearchModal`) instead of large
   dropdowns. Gaps are minor: "Don't Know Yet" with reasons exists only on the
   fact-checker form, not the name review; "apply to this occurrence vs all"
   exists (`Review Each Mention`) but is not offered as a follow-up question
   after every correction.
6. **Latent bugs (small, confirmed by reading):**
   - `live_fact_check_service.py` ~825–827: adjudicator reasoning-effort
     fallback mixes `"medium"`/`"low"` defaults.
   - Gemini verifier registers `provider="google"` / default model
     `gemini-3.7-flash`, but the price table keys `("gemini", "gemini-3.6-flash")`
     — every Gemini call would be recorded as unpriced.
   - FTS5 chunk search (`search_transcript_chunks`) has no production consumer
     — built but unused.

## 5. Repository/housekeeping findings

- `CrunchyNutMilk/Transcripts-AI` is empty (README only). The real code lives
  outside GitHub. Recommendation: make this repo (or a private one) the home of
  the current bot source so work like this plan can run against real history.
- `CrunchyNutMilk/Summurizer` (public) contains `IT IS PREFECT DONT LOSE.zip`,
  which includes a `.env` file. If that file holds your real Discord token or
  OpenAI key, **rotate both keys and remove the zip from the public repo**
  (removal alone is not enough — the git history keeps it).
