# RUNBOOK — first session on your PC

Everything below is copy-paste, in order. Windows PowerShell assumed; use the
same Python 3.11+ you run the bot with (`py` or `python`). The engine has zero
dependencies — no venv changes needed unless you want `faster-whisper`.

---

## 1. Get the code and prove it works (2 min)

```powershell
git clone https://github.com/CrunchyNutMilk/Transcripts-AI.git
cd Transcripts-AI
git checkout claude/transcript-ai-pipeline-gc4vpn
pip install pytest
python -m pytest tests/ -q        # expect: all tests pass (CI runs the same suite)
```

## 2. Write your player mapping (5 min, do this BEFORE ingesting)

Copy `mapping.example.json` to somewhere **outside the repo** (it's
git-ignored anyway, but keep campaign data out of the clone) and fill in the
real Discord IDs, speaker labels and character names, plus who the DM labels
are. The mapping file is the authoritative source of PCs — with it, the
engine never guesses PCs from speaker frequency.

```powershell
copy mapping.example.json C:\Users\neill\Documents\engine\mapping.json
notepad C:\Users\neill\Documents\engine\mapping.json
```

## 3. Build campaign memory from your real vault (2 min)

Keep the database **outside the repo** too. The ingester walks subfolders,
reads only `* Transcripts Mapped*.md`, and never touches Unmapped files.

```powershell
python scripts/ingest_campaign.py `
  --data-dir "C:\Users\neill\Documents\<your vault>\<campaign folder>\01 - Transcript Mapped" `
  --db C:\Users\neill\Documents\engine\engine_memory.sqlite `
  --campaign "Heckuva Side Quest" `
  --mapping C:\Users\neill\Documents\engine\mapping.json `
  --report C:\Users\neill\Documents\engine\heckuva_report.md
```

Open `heckuva_report.md` — same shape as the one from the cloud session.

## 4. Clear the review queue (10–15 min, the highest-value step)

All `python -m transcripts_ai` commands below take the database first:
`python -m transcripts_ai --db C:\Users\neill\Documents\engine\engine_memory.sqlite <command> ...`
(shortened to `--db ...` in the examples).

```powershell
python -m transcripts_ai --db C:\Users\neill\Documents\engine\engine_memory.sqlite `
  reviews --campaign "Heckuva Side Quest"
```

Not sure about an item? `--action let-ai-pick` gives the engine's own
advisory recommendation (learner-scored, explained, never applied).

Act on items with `resolve`. **The decisions are yours** — the commands below
are prepared for the items the cloud pass found, assuming the obvious answer
is the right one. Use `--item <id>` (ids are stable; list numbers shift).

Speaker-label variants (approve as aliases if correct):

```powershell
python -m transcripts_ai resolve --campaign "Heckuva Side Quest" --user neill `
  --item <id-for-Jenx> --action alias --canonical "Jinx" --reason "speaker label drift"
# repeat for: Althena -> Althea, Diego the Bear -> Diego (or the reverse,
# whichever name is canon), Boblin Thee 7enth -> Boblin
```

Phonetic variants of Ghomra/Gomph the DM's transcription mangles:

```powershell
# Gomra, Gamra, Gamera, Goomra, Gomrad -> your canonical spelling
python -m transcripts_ai resolve --campaign "Heckuva Side Quest" --user neill `
  --item <id> --action correct --canonical "Ghomra" --reject-others
```

Real entities (create pages later; kinds: pc npc location faction quest item
weapon armour potion spell creature deity lore organisation other):

```powershell
python -m transcripts_ai resolve --campaign "Heckuva Side Quest" --user neill `
  --item <id-for-Silas> --action new --kind npc
# Silver Spire / Oxwater -> --kind location ; Argon -> npc ;
# Vel'Nadar -> npc or lore ; Sparkle Sword / Pearl of Power -> item ;
# Eldritch Blast / Detect Magic / Branding Smite -> spell
```

Junk that slipped through ("AIDS", "Dad", "Jew", "Tell", ...):

```powershell
python -m transcripts_ai resolve --campaign "Heckuva Side Quest" --user neill `
  --item <id> --action not-entity
```

Unsure? `--action defer` keeps it pending; `--action dont-know --reason
dm_has_not_said` records why. **Important:** a `correct`/`alias`/`not-entity`
decision is remembered — wrong ones can be undone via
`python -m transcripts_ai audit` + `forget`, so don't agonise.

## 4b. Optional: auto-fix ordinary-word misspellings in a Mapped transcript

Ordinary vocabulary ("becuase" → "because") is corrected automatically —
never sent to review, never made an entity. Names are protected: capitalised
words, campaign entities/aliases (including phonetic and one-edit
neighbours), repeated unknown terms, and playful coinages are left alone.
Always dry-run first; corrections are audited as `auto_spell_correction`.
Transcript Unmapped is refused outright.

```powershell
python -m transcripts_ai --db ...engine_memory.sqlite spellcheck `
  --campaign "Heckuva Side Quest" --transcript "path\to\... Transcripts Mapped.md"
# review the printed list, then add --apply to write
```

## 5. Train the learner on your decisions (10 s)

```powershell
python -m transcripts_ai train --campaign "Heckuva Side Quest"
```

Needs ≥8 decisions including some rejections; run it again after every review
session. `entities --campaign ...` shows what the engine now knows.

## 6. Re-ingest so approved names deepen the index (1 min)

```powershell
python scripts/ingest_campaign.py --data-dir "...\01 - Transcript Mapped" `
  --db C:\Users\neill\Documents\engine\engine_memory.sqlite --campaign "Heckuva Side Quest" --report heckuva_report2.md
```

Resolved items stay resolved; the report should now show fewer unknowns.

## 7. Transcribe or evaluate the two new sessions (2026-08-09, 2026-08-16)

Start your whisper.cpp server (the bot's usual one), then:

**The production path (recommended):** run both sessions through the bot as
usual so speakers get mapped, then feed the Mapped transcripts into permanent
memory with your mapping file:

```powershell
python -m transcripts_ai --db C:\Users\neill\Documents\engine\engine_memory.sqlite `
  process --campaign "Heckuva Side Quest" --session 2026-08-09 `
  --game "Heckuva Side Quest" --date 2026-08-09 `
  --mapping C:\Users\neill\Documents\engine\mapping.json `
  --transcript "path\to\2026-08-09 ... Transcripts Mapped.md" `
  --summary-out C:\Users\neill\Documents\engine\20260809_summary.md
```

**The quick evaluation path (optional):** transcribe the raw audio without
the bot. Speakers come out as `Unknown:`, so this writes to a **disposable
evaluation database** (`<out>_eval.sqlite`) automatically — it can never
touch your permanent campaign memory, and `process` refuses unmapped files.

```powershell
python scripts/transcribe_session.py `
  --backend whisper-cpp --server http://127.0.0.1:8178 `
  --out C:\Users\neill\Documents\engine\20260809_unmapped.md --process `
  --campaign "Heckuva Side Quest" --session 2026-08-09 `
  --game "Heckuva Side Quest" `
  "path\to\20260809*Part*.mp3"
```

Notes:
- The script expands `*` wildcards itself, so the quoted pattern works in
  PowerShell as written.
- If the server rejects MP3 (some builds want 16 kHz WAV), convert first:
  `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav` — the script accepts WAV too.
  Or use `--backend faster-whisper` after `pip install faster-whisper`.

## 7b. Game-day audio: make a gold transcript that pays for itself

For a session where you have the audio and time to correct the transcript
once (the "perfect transcript" plan in docs/TRAINING.md). Everything below
assumes the memory DB from step 3 exists.

```powershell
$env:DB = "C:\Users\neill\Documents\engine\engine_memory.sqlite"
$env:WORK = "C:\dev\Transcript_AI_data\gold\2026-08-16"
mkdir $env:WORK -Force

# 1. Seed Whisper with your campaign's names BEFORE transcribing —
#    the single cheapest transcription improvement available.
python -m transcripts_ai.lab whisper-prompt --db $env:DB `
  --campaign "Heckuva Side Quest" --out $env:WORK\prompt.txt

# 2. Transcribe the audio with the seeded prompt.
python scripts/transcribe_session.py `
  --backend whisper-cpp --server http://127.0.0.1:8178 `
  --initial-prompt-file $env:WORK\prompt.txt `
  --out $env:WORK\machine.md `
  "path\to\that game*Part*.mp3"

# 3. Make the gold copy and correct it by hand (this is the human work:
#    fix names, fix words; don't worry about punctuation).
copy $env:WORK\machine.md $env:WORK\gold.md
#    ... edit gold.md in your editor ...

# 4. Score the machine against your corrections and mine every fix:
python -m transcripts_ai.lab gold-score `
  --gold $env:WORK\gold.md --hyp $env:WORK\machine.md `
  --db $env:DB --campaign "Heckuva Side Quest" --session 2026-08-16 `
  --out-records $env:WORK\corrections.jsonl `
  --report $env:WORK\gold_report.md
```

What you get: the session's **WER** and **known-name accuracy** (the
baseline every later improvement is measured against), a report of exactly
which names Whisper mangled and what it wrote instead, and
`corrections.jsonl` — every fix you made, as human-labelled training
records. Keep `gold.md` forever; it is a frozen-eval candidate.

Two tips that make step 3 fast:
- Correct names first (search-replace the mangled forms from the report of
  a previous run); ordinary-word fixes are a bonus, not the point.
- You do not need to fix everything — an 80%-corrected gold transcript
  still yields correct WER trends and hundreds of mined corrections.

## 8. Inspect what came out

```powershell
python -m transcripts_ai facts --campaign "Heckuva Side Quest" --session 2026-08-09
python -m transcripts_ai reviews --campaign "Heckuva Side Quest"
python -m transcripts_ai audit --campaign "Heckuva Side Quest"
```

The summary was written next to the transcript (`*_summary.md`). Compare it
against what you remember of the session — missed facts and wrong facts are
exactly the feedback the engine learns from.

## Safety notes (unchanged from the design)

- Transcript Unmapped is never read for editing, never written.
- The engine writes only to its own `engine_memory.sqlite` and the output
  files you name — it does not modify vault pages (vault updates are emitted
  as suggestions for you to apply).
- Every decision and memory write is in the audit log; `forget` removes any
  learned row.
