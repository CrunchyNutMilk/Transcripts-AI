# Architecture — the D&D Transcript Intelligence Engine (`transcripts_ai`)

This is the design for the system that will take over the responsibilities of
Luna (extraction), Terra (verification) and Sol (adjudication/advisory) in the
Summurizer pipeline. It builds on the audit in `docs/AUDIT.md` and the staged
plan in `docs/PLAN.md`.

## What Luna/Terra/Sol do today (behaviours to preserve)

| Behaviour | Owner today | Must-keep contract |
|---|---|---|
| Extract facts + entity mentions per 5-min window | Luna | exact-quote evidence, bounded categories, JSON only |
| Independently verify each fact | Terra | verdict supported/contradicted/uncertain with evidence IDs |
| Adjudicate only disagreements, batched | Sol | escalation on evidence problems only, never importance; unresolved → human |
| Advisory pick during name review | Sol | display-only; confidence floor; restricted to offered entity IDs |
| Campaign Q&A with citations | Sol | answers only from provided sources |
| Fail closed on invalid output | all | route to review, never crash, never write unvalidated data |
| Cost/usage accounting per request | all | price snapshot, idempotent, no prompt text stored |

Overlaps the replacement corrects:
- Three separately prompted models re-derive session context each time; the
  engine builds one **context package** per task from persistent memory.
- Entity knowledge lives in three places (vault index, glossary, live entity
  mentions); the engine unifies them in one campaign memory with provenance.
- The summariser is disconnected from the fact ledger except for the fast
  path; here the summary is *derived from* verified facts plus transcript.

## Package layout

```
transcripts_ai/
├── schemas.py        # dataclasses + validation: Provenance, EpistemicStatus,
│                     #   Fact, EntityRecord, AliasRecord, ReviewItem,
│                     #   ContextPackage, SessionSummary, VaultUpdatePlan
├── transcript.py     # Mapped-transcript parser, cleaner, chunker
├── phonetics.py      # phonetic keys + Damerau-Levenshtein
├── resolver.py       # name/alias/spelling resolution with explanations
├── detector.py       # rule-based entity detection + suppression
├── dnd_patterns.py   # initiative/rolls/rests/deaths/combat + epistemic cues
├── memory.py         # campaign-scoped SQLite store (facts, entities, aliases,
│                     #   contradictions, feedback, file index, FTS, audit log)
├── retrieval.py      # context-package builder (budgeted, source-recorded)
├── providers.py      # ChatProvider protocol, role registry, OpenAI-compatible
│                     #   HTTP provider, FakeProvider, validation + retry
├── extraction.py     # fact extractor + deterministic evidence validation
├── verification.py   # verifier + contradiction detection vs memory
├── summarizer.py     # structured summary + no-hallucination post-pass + MD
├── review.py         # review queue, six actions, feedback learning
├── vault_planner.py  # plan-only vault updates (marker blocks)
└── cli.py            # process / inspect / forget / export commands
```

Everything except `providers.py` network calls is deterministic and fully
unit-tested. Tests use `FakeProvider` — no network, no API keys.

## The epistemic model (core of "safe against invented facts")

Every stored statement carries an `EpistemicStatus`:

`confirmed_canon` > `strongly_supported` > `dm_hint` > `character_belief` >
`player_assumption` > `unconfirmed_theory` > `table_talk` > `conflicting` >
`retconned`

Rules:
1. Only **human decisions** or **verified facts reconciled against the final
   reviewed transcript** may reach `confirmed_canon`.
2. AI outputs enter at most `strongly_supported`, and only when the evidence
   quote validates deterministically against the source text.
3. Jokes/OOC/table-talk cues cap status at `table_talk` regardless of model
   confidence.
4. Contradictions never overwrite: both facts are kept, linked in
   `contradictions`, status of the older fact drops to `conflicting` until a
   human resolves or a retcon is confirmed.
5. Every fact keeps `Provenance`: campaign, session, source file, line span,
   speaker, exact quote, extractor identity + version, and the source-set hash
   of the context package that produced it. Memory is therefore inspectable,
   correctable and deletable by provenance.

## Provider roles (provider-independence)

Roles, not models, are configured: `extractor`, `verifier`, `adjudicator`,
`summarizer`, `reviewer`, `answerer`. Env: `AI_ROLE_EXTRACTOR="openai:gpt-…"`,
`"local:qwen3-32b"`, etc. One HTTP provider speaks the OpenAI-compatible
chat-completions dialect used by OpenAI, Ollama, llama.cpp server and LM
Studio, so cloud→local is a config change. Validation contract for every role:
strict JSON schema → one bounded retry with the validation error → review
queue. Invalid output can cost a retry; it can never write data.

## Retrieval flow (per task)

1. Resolve campaign (explicit ID — never inferred across campaigns).
2. Parse current Transcript Mapped; identify speakers, candidate entities,
   D&D patterns.
3. Query memory: matching entities + aliases, related facts (FTS + entity
   links), open contradictions, prior-session summaries, approved corrections.
4. Assemble a budgeted `ContextPackage` (char-bounded, priority-ordered) and
   record every source (path/id + hash) in the package manifest.
5. Run the role. 6. Validate output against the manifest — a claim citing a
   source not in the manifest is rejected.

## Learning loop (explainable, reversible)

After each session review, the feedback ledger records: accepted/rejected
corrections, new aliases, confirmed entities, misclassifications, missed and
false-positive facts, summary edits, chosen vault links, deferred items —
each with reviewer, timestamp and the evidence shown. Future runs consult the
ledger (e.g. a rejected correction is not re-proposed for the same
name+context; an approved alias resolves immediately). `cli.py forget`
removes any learned row by ID with an audit entry; nothing is ever silently
rewritten. The ledger doubles as the **verified dataset export** for optional
future fine-tuning (human-decided rows only).

## The staged evidence system (implemented)

The engine never asks a model to "find every name and fact". Stages, each a
separate tested module:

| Stage | Module | Notes |
|---|---|---|
| 1. Session context | `session_context.py` | player→PC mapping authoritative; per-session overrides; campaign-scoped; DM identity |
| 2. Parse Mapped transcript | `transcript.py` | Unmapped never read for editing, source never written |
| 3. Scene segmentation | `scenes.py` | roleplay/travel/arrival/NPC talk/combat/loot/planning/rest/rules/OOC; stateful combat bounds; sustained-run OOC rule |
| 4. Candidate detection | `detector.py` | suppression-biased rules; capitalisation alone never suffices |
| 5. Name/alias resolution | `resolver.py` + `phonetics.py` | exact → alias → feedback history → edit-distance/phonetic/containment → explanation; length-tiered thresholds |
| 6. Fact extraction | `extraction.py` (extractor role) | subject/relationship/object triples, time status, speaker mode, exact quote |
| 7. Evidence gate | `schemas.ContextPackage.contains_quote` | deterministic; invented quotes rejected pre-storage |
| 8. Independent verification | `extraction.py` (verifier role) | disagreement → review queue |
| 9. Deterministic ceilings | `dnd_patterns.py` | joke/OOC→table_talk; NPC dialogue→character claim; planned/negated→never events |
| 10. Contradiction detection | `extraction.py` + `memory.py` | link, never overwrite; human resolves; retcon support |
| 11. Summary | `summarizer.py` | derived from verified ledger; confirmed/uncertain split; no-hallucination gate; initiative order only when spoken |
| 12. Review queue | `review.py` + `memory.py` | Correct/Alias/New/Save for Review/Let AI Pick (advisory-only)/Don't Know Yet |
| 13. Learning | `memory.py` feedback ledger | approved + rejected decisions remembered; reversible via `forget`; human-only canon |
| 14. Confidence bands | `resolver.decide_band_action` | ≥0.95+strong evidence auto-link (existing entities only), 0.80 suggest, 0.60 review, else drop |

Speaker-authority hierarchy (enforced in `dnd_patterns.assess_speaker_mode` +
extraction ceilings): DM narration > mechanical result > NPC dialogue (claim
only) > PC dialogue > player statement > table talk. "Arden says he works for
the king" is stored as Arden's claim, never as world fact.

## Migration path (mirrors docs/PLAN.md stages)

1. Engine developed and tested standalone in this repo (this codebase).
2. Shadow mode inside the bot: engine processes the same windows; outputs are
   compared, never applied — reusing the learning-engine comparison pattern.
3. Role-by-role takeover behind config flags (summariser first, adjudicator
   last), each gated on shadow accuracy vs the incumbent.
4. Luna/Terra/Sol wiring removed only after every role passes its gate.
