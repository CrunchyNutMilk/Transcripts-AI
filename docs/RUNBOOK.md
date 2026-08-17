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
python -m pytest tests/ -q        # expect: 162+ passed
```

## 2. Build campaign memory from your real vault (2 min)

Point at the campaign's transcripts folder — it walks subfolders, reads only
`* Transcripts Mapped*.md`, and never touches Unmapped files.

```powershell
python scripts/ingest_campaign.py `
  --data-dir "C:\Users\neill\Documents\<your vault>\<campaign folder>\01 - Transcript Mapped" `
  --db engine_memory.sqlite `
  --campaign "Heckuva Side Quest" `
  --report heckuva_report.md
```

Open `heckuva_report.md` — same shape as the one from the cloud session.

## 3. Clear the review queue (10–15 min, the highest-value step)

```powershell
python -m transcripts_ai reviews --campaign "Heckuva Side Quest"
```

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

## 4. Train the learner on your decisions (10 s)

```powershell
python -m transcripts_ai train --campaign "Heckuva Side Quest"
```

Needs ≥8 decisions including some rejections; run it again after every review
session. `entities --campaign ...` shows what the engine now knows.

## 5. Re-ingest so approved names deepen the index (1 min)

```powershell
python scripts/ingest_campaign.py --data-dir "...\01 - Transcript Mapped" `
  --db engine_memory.sqlite --campaign "Heckuva Side Quest" --report heckuva_report2.md
```

Resolved items stay resolved; the report should now show fewer unknowns.

## 6. Transcribe + process the two new sessions (2026-08-09, 2026-08-16)

Start your whisper.cpp server (the bot's usual one), then:

```powershell
python scripts/transcribe_session.py `
  --backend whisper-cpp --server http://127.0.0.1:8178 `
  --out "20260809_unmapped.md" --process `
  --campaign "Heckuva Side Quest" --session 2026-08-09 `
  --game "Heckuva Side Quest" --db engine_memory.sqlite `
  "path\to\20260809*Part*.mp3"
```

Notes:
- If the server rejects MP3 (some builds want 16 kHz WAV), convert first:
  `ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav` — the script accepts WAV too.
  Or use `--backend faster-whisper` after `pip install faster-whisper`.
- This quick path has **no speaker names** (lines are `Unknown:`). For full
  quality, run the session through the bot as usual and point
  `python -m transcripts_ai process` at the resulting Transcripts Mapped file
  instead — that is the intended production path.

## 7. Inspect what came out

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
