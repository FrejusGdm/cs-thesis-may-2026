# P1 Cascade Pipeline

**Adja speech → ASR → MT(→French) → [response] → MT(→Adja) → TTS → Adja speech**

Measures error accumulation at each stage to identify where degradation is worst.

## Quick start

```bash
# Dry run — no GPU, no API key, uses stubs. Runs in <30s on Mac:
bash dry_run.sh

# Round-trip evaluation (no LLM, just error measurement):
python pipeline.py \
  --mode roundtrip \
  --input path/to/adja_audio.wav \
  --ref-transcript "reference adja transcript"

# Q&A mode (Gemini generates French answer):
GEMINI_API_KEY=your_key python pipeline.py \
  --mode qa \
  --input path/to/question.wav

# Batch over a directory:
python pipeline.py --mode roundtrip --input-dir samples/ --output-dir runs/batch01/
```

## Setup

1. **Dependencies**: `pip install -r requirements.txt`
   - For TTS: also `pip install unsloth`
   - For Gemini modes: also `pip install google-generativeai`

2. **NLLB checkpoint** (MT): Set `mt.checkpoint` in `configs/p1_config.yaml` and change `mt.mode: nllb`.
   Until your NLLB checkpoint is ready, leave `mode: stub` (identity transform, no model).

3. **ASR checkpoints**: Already pre-filled (`E4v4` + `D4_C4v2_lm_optuna`).
   Requires `HF_TOKEN` env var for private repos.

4. **TTS checkpoint**: Set `tts.checkpoint` in config to your Spark TTS fine-tune.

## Pipeline modes

| Mode | Description |
|------|-------------|
| `roundtrip` | Adja → ASR → MT→FR → MT→Adja → TTS → RTT-ASR. No LLM in loop. |
| `qa` | Adja → ASR → MT→FR → Gemini(FR) → MT→Adja → TTS. Full spoken Q&A. |

## Stage config (p1_config.yaml)

```yaml
mt:
  mode: stub      # stub | nllb | gemini
  checkpoint: "..." # set to your NLLB checkpoint when ready

response:
  mode: identity  # identity | gemini

asr:
  primary: xlsr   # which ASR result feeds downstream (whisper | xlsr)
```

## Error accumulation report

Each sample prints a table like:

```
────────────────────────────────────────────────────────────────────────
  Sample: my_audio
────────────────────────────────────────────────────────────────────────
Stage                    Metric           Output preview
────────────────────────────────────────────────────────────────────────
ASR (xlsr)               CER=22.7         mo yi adja bo...
MT Adja→FR               chrF=41.0        bonjour je suis...
MT FR→Adja               chrF=38.0        mo yi adja...
TTS                      —                runs/latest/my_audio_output.wav
TTS RTT-WER              WER=35.0         mo yi adja ...
────────────────────────────────────────────────────────────────────────
```

## Outputs

Each run produces in `--output-dir` (default `runs/latest/`):
- `results.json` — all per-sample metrics
- `<sample>_output.wav` — synthesized Adja speech
- `<sample>_cache.json` — cached stage outputs (re-run skips completed stages)

## Tracking (per CLAUDE.md)

After the first real run, update:
1. `experiments/registry.md` — add P1 row
2. `results/run-ledger.md` — chronological entry
3. `results/pipeline-comparison.md` — create this new file for cascade results
