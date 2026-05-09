# Qwen3 HF Jobs for Adja

Prepared for: Josue Godeme
Requested by: Josue Godeme

This directory documents the first Hugging Face job submissions for:

- `Qwen3-ASR-0.6B` on Adja
- `Qwen3-TTS-12Hz-0.6B-Base` on Adja
- `Qwen3-TTS-12Hz-1.7B-Base` on Adja

## Purpose

Keep one place that links:

- the local implementation revision
- the remote Hugging Face job IDs
- the exact job commands and hardware choices
- the output repos used for artifacts
- the reasoning behind any runtime fallback or failure

## Outcome Snapshot

As of Sunday, April 19, 2026 (`America/New_York`):

- `QASR-s42` full pilot `69e38fe4ac288e522d8efd32` completed and saved pilot checkpoint `checkpoint-320`
- held-out ASR eval `69e4db9cac288e522d8f001f` completed with:
  - raw WER `100.0%`
  - raw CER `53.14%`
  - normalized WER `100.65%`
  - normalized CER `52.26%`
- `T5-qwen3-0p6b` remains a real `0.6B` architecture failure (`2048` vs `1024` embedding-width mismatch)
- first `1.7B` A100 smoke/full runs proved the width mismatch was gone, but both died later on `soundfile.LibsndfileError: Format not recognised`
- patched `1.7B` reruns on `l40sx1` now complete end-to-end:
  - smoke rerun `69e4da24ac288e522d8f001a`
  - full rerun `69e4da24ac288e522d8f0018`
- no human listening verdict is recorded yet for the Qwen TTS samples

## Sources

- Official Qwen upstream repositories:
  - `QwenLM/Qwen3-ASR`
  - `QwenLM/Qwen3-TTS`
- Local vendored and wrapper files:
  - `experiments/finetuning-qwen3/vendor/qwen3_tts_official/*`
  - `scripts/hf_jobs/qwen3_asr_adja.py`
  - `scripts/hf_jobs/qwen3_tts_adja.py`
  - `experiments/finetuning-qwen3/data_pipeline/adja_hf_materialize.py`
- HF job submission references:
  - `hf jobs run --help`
  - `hf jobs uv run --help`
  - `learnings-from-the-past/hf-jobs-gotchas.md`

## Related documentation

- Documentation map:
  - [docs/qwen3-documentation-map.md](<LOCAL_PATH>
- Technical retrospective:
  - [docs/qwen3-hf-jobs-retrospective.md](<LOCAL_PATH>
- Blog draft:
  - [docs/qwen3-hf-jobs-blog-draft.md](<LOCAL_PATH>
- Operational runbook:
  - [experiments/finetuning-qwen3/docs/adja_runbook.md](<LOCAL_PATH>

## Correct launch pattern

Use the repo-native HF Jobs submission flow:

- submit `scripts/hf_jobs/qwen3_asr_adja.py` or `scripts/hf_jobs/qwen3_tts_adja.py` directly with `hf jobs uv run`
- run from the local repo root so the workspace context is uploaded
- do **not** bootstrap the job by cloning `adja-nmt` inside the container

Convenience helper:

```bash
python scripts/hf_jobs/submit_qwen3_jobs.py --modality both --smoke --execute
python scripts/hf_jobs/submit_qwen3_jobs.py --modality asr --pilot --execute --detach
python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --smoke --tts-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-output-repo-id JosueG/qwen3-adja-tts-1p7b --tts-flavor l40sx1 --execute --detach
python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --pilot --tts-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-output-repo-id JosueG/qwen3-adja-tts-1p7b --tts-flavor l40sx1 --execute --detach
```

Direct ASR smoke:

```bash
hf jobs uv run \
  --flavor a10g-large \
  --timeout 4h \
  --python 3.11 \
  --secrets HF_TOKEN \
  -e HF_DATASET_ID=JosueG/adja-tts-orpheus \
  -e MODEL_ID=Qwen/Qwen3-ASR-0.6B \
  -e OUTPUT_REPO_ID=JosueG/qwen3-adja-asr-0p6b \
  -e RESULTS_REPO_ID=JosueG/qwen3-adja-results \
  -e SMOKE_RUN=1 \
  -e WORKSPACE_DIR=/tmp/qwen3_adja_asr \
  -e SEED=42 \
  scripts/hf_jobs/qwen3_asr_adja.py
```

Direct TTS smoke:

```bash
hf jobs uv run \
  --flavor a10g-large \
  --timeout 4h \
  --python 3.11 \
  --secrets HF_TOKEN \
  -e HF_DATASET_ID=JosueG/adja-tts-orpheus \
  -e MODEL_ID=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  -e OUTPUT_REPO_ID=JosueG/qwen3-adja-tts-0p6b \
  -e RESULTS_REPO_ID=JosueG/qwen3-adja-results \
  -e SMOKE_RUN=1 \
  -e WORKSPACE_DIR=/tmp/qwen3_adja_tts \
  -e SEED=42 \
  scripts/hf_jobs/qwen3_tts_adja.py
```

## Submission Log

### 2026-04-18 — Attempt 1: container-side clone bootstrap [FAILED]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Git commit: `0402b6a`
- HF namespace: `JosueG`
- Results repo: `JosueG/qwen3-adja-results`
- Docker image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`

ASR smoke:

- Experiment ID: `QASR-s42`
- Job ID: `69e30a7bac288e522d8efb5c`
- URL: `https://huggingface.co/jobs/JosueG/69e30a7bac288e522d8efb5c`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-asr-0p6b`
- Status at submission: `SCHEDULING`
- Final status: `ERROR`
- Failure stage: container bootstrap, before Python startup
- Failure signature: `fatal: could not read Username for 'https://github.com': No such device or address`
- Retrieve logs: `hf jobs logs 69e30a7bac288e522d8efb5c`

TTS smoke:

- Experiment ID: `T5-qwen3-0p6b`
- Job ID: `69e30a8eac288e522d8efb5e`
- URL: `https://huggingface.co/jobs/JosueG/69e30a8eac288e522d8efb5e`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-tts-0p6b`
- Status at submission: `SCHEDULING`
- Final status: `ERROR`
- Failure stage: container bootstrap, before Python startup
- Failure signature: `fatal: could not read Username for 'https://github.com': No such device or address`
- Retrieve logs: `hf jobs logs 69e30a8eac288e522d8efb5e`

Notes:

- The first submission pair used a container-side `git clone` bootstrap. That was the wrong pattern for this repo.
- The corrected pattern is `hf jobs uv run` against the local launcher scripts in `scripts/hf_jobs/`.
- Both jobs pass `HF_TOKEN` as a secret and use `JosueG/adja-tts-orpheus` as the dataset source.
- The first pass is smoke-only with `SMOKE_RUN=1`. Pilot runs should only be launched after these complete or fail with actionable evidence.
- Observed result: neither job reached dataset access, dependency install inside the repo, model load, or training. The failure happened at the `git clone` step inside the HF container.

### 2026-04-18 — Attempts 2-5: launcher-shape corrections [FAILED]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Goal: align Qwen with the repo's working HF pattern used by Parakeet and the other self-contained job scripts

Attempt 2:

- ASR job: `69e382dfac288e522d8efd10`
- TTS job: `69e382e1cd8c002f31dfe942`
- Result: switched to repo-native `hf jobs uv run`, but both launchers still imported `qwen3_job_utils`; HF only uploaded the script file, so both failed with `ModuleNotFoundError`

Attempt 3:

- ASR job: `69e384d0cd8c002f31dfe952`
- TTS job: `69e384d1cd8c002f31dfe954`
- Result: launchers were made standalone, but still tried `python -m pip install ...` at runtime; the UV image failed with `No module named pip`

Attempt 4:

- ASR job: `69e38514cd8c002f31dfe956`
- TTS job: `69e38515ac288e522d8efd1c`
- Result:
  - ASR hit package resolution failure because the HF index exposed `qwen-asr==0.0.6`, not `>=0.1.0`
  - TTS installed packages into the wrong interpreter context; runtime then failed with `ModuleNotFoundError: No module named 'librosa'`

Attempt 5:

- ASR job: `69e3856aac288e522d8efd1e`
- TTS job: `69e3856bcd8c002f31dfe964`
- Result: `uv pip --python /usr/bin/python3.11 ...` still violated the container's externally-managed interpreter rules and both jobs failed before dataset access

### 2026-04-18 — Attempt 6: repo-native `uv` metadata launchers [FAILED IN MODEL/TRAINER CODE]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Git commit: `538d415`
- HF namespace: `JosueG`
- Results repo: `JosueG/qwen3-adja-results`
- Docker image: `ghcr.io/astral-sh/uv:python3.12-bookworm`

ASR smoke:

- Experiment ID: `QASR-s42`
- Job ID: `69e3865bac288e522d8efd24`
- URL: `https://huggingface.co/jobs/JosueG/69e3865bac288e522d8efd24`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-asr-0p6b`
- Final status: `ERROR`
- Failure stage: after dependency resolution, dataset download, split/materialization, and model load
- Failure signature: `TypeError("TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'")`
- Retrieve logs: `hf jobs logs 69e3865bac288e522d8efd24`

TTS smoke:

- Experiment ID: `T5-qwen3-0p6b`
- Job ID: `69e3865dcd8c002f31dfe96e`
- URL: `https://huggingface.co/jobs/JosueG/69e3865dcd8c002f31dfe96e`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-tts-0p6b`
- Final status: `ERROR`
- Failure stage: after dependency resolution, dataset download, split/materialization, and model load
- Failure signature: `RuntimeError('The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 2')`
- Retrieve logs: `hf jobs logs 69e3865dcd8c002f31dfe96e`

Notes:

- This attempt finally matches the repo's working HF pattern:
  - self-contained `scripts/hf_jobs/*.py`
  - inline `uv` dependency metadata
  - local submission via `hf jobs uv run`
  - no container-side `git clone`
- `qwen-asr` is pinned to `0.0.6` because that is the version currently resolvable in the HF runtime used here.
- The vendored upstream Qwen files under `experiments/finetuning-qwen3/vendor/` were preserved; only Adja-specific launcher code was changed.

### 2026-04-18 — Attempt 7: ASR compatibility fix + TTS traceback instrumentation [FAILED AFTER REAL EXECUTION]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Git commit: `538d415` plus uncommitted local launcher fixes
- Goal:
  - ASR: adapt to the `TrainingArguments` signature exposed in the HF runtime
  - TTS: print full traceback and model-dimension diagnostics so the 2048/1024 mismatch can be fixed precisely

ASR smoke:

- Experiment ID: `QASR-s42`
- Job ID: `69e388ccac288e522d8efd26`
- URL: `https://huggingface.co/jobs/JosueG/69e388ccac288e522d8efd26`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-asr-0p6b`
- Final status: `ERROR`
- Effective result: smoke training itself succeeded, but the job exited during interpreter teardown
- Failure signature: `Exception ignored in sys.unraisablehook`
- Retrieve logs: `hf jobs logs 69e388ccac288e522d8efd26`

TTS smoke:

- Experiment ID: `T5-qwen3-0p6b`
- Job ID: `69e388cdcd8c002f31dfe98a`
- URL: `https://huggingface.co/jobs/JosueG/69e388cdcd8c002f31dfe98a`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-tts-0p6b`
- Final status: `ERROR`
- Failure signature: `RuntimeError('The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 2')`
- Retrieve logs: `hf jobs logs 69e388cdcd8c002f31dfe98a`

Notes:

- ASR smoke produced a real checkpoint and sample decode, then died at shutdown because the launcher closed the tee log file while `stdout` / `stderr` still pointed at it.
- TTS 0.6B reached the real training path and failed on a representation-size mismatch in eager mode:
  - `text_embedding_dim=2048`
  - `codec_embedding_dim=1024`
- The vendored upstream TTS fine-tuning script defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, which is the main source-backed reason to test 1.7B next instead of repeating 0.6B unchanged.

### 2026-04-18 — Attempt 8: ASR full run + TTS 1.7B smoke [ASR COMPLETED, A100 TTS FAILED LATE]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Git commit: uncommitted local launcher fixes plus submit-helper overrides
- Goal:
  - ASR: launch the full pilot now that smoke training, checkpointing, and decoding have already worked
  - TTS: switch from `0.6B` to `1.7B` for the next smoke because the official upstream script is `1.7B`-first and the `0.6B` path hit a real embedding-width mismatch

ASR full pilot:

- Experiment ID: `QASR-s42`
- Job ID: `69e38fe4ac288e522d8efd32`
- URL: `https://huggingface.co/jobs/JosueG/69e38fe4ac288e522d8efd32`
- Flavor: `a10g-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-asr-0p6b`
- Final status: `COMPLETED`
- Retrieve logs: `hf jobs logs 69e38fe4ac288e522d8efd32`

TTS 1.7B smoke:

- Experiment ID: `T5-qwen3-1p7b`
- Job ID: `69e38fe3ac288e522d8efd30`
- URL: `https://huggingface.co/jobs/JosueG/69e38fe3ac288e522d8efd30`
- Flavor: `a100-large`
- Timeout: `4h`
- Output repo: `JosueG/qwen3-adja-tts-1p7b`
- Final status: `ERROR`
- Retrieve logs: `hf jobs logs 69e38fe3ac288e522d8efd30`

Notes:

- The launcher teardown bug was fixed in both `scripts/hf_jobs/qwen3_asr_adja.py` and `scripts/hf_jobs/qwen3_tts_adja.py` by restoring `stdout` / `stderr` before closing the log handle.
- The submit helper now accepts explicit model, flavor, and timeout overrides so future Qwen runs can be traced back to an exact submission command instead of ad hoc manual edits.
- The `1.7B` smoke log confirms that the old `0.6B` width mismatch is gone on `1.7B`:
  - `text_embedding_dim=2048`
  - `codec_embedding_dim=2048`
  - first observed train line: `attn=eager epoch=0 step=0 loss=10.4554`
- ASR full pilot summary:
  - smoke checkpoint: `checkpoint-2`
  - pilot checkpoint: `checkpoint-320`
  - smoke and pilot sample decode artifacts were both written
- The A100 `1.7B` smoke failed after real training when writing `sample.wav`:
  - failure signature: `soundfile.LibsndfileError: Format not recognised`

### 2026-04-18 — Attempt 9: TTS 1.7B full pilot launched in parallel with smoke [A100 FAILED LATE]

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- Goal:
  - trade a bit of duplicate compute for faster validation
  - if the `1.7B` smoke path keeps working, avoid waiting for smoke completion before starting the full train

TTS 1.7B full pilot:

- Experiment ID: `T5-qwen3-1p7b-full`
- Job ID: `69e3942dac288e522d8efd40`
- URL: `https://huggingface.co/jobs/JosueG/69e3942dac288e522d8efd40`
- Flavor: `a100-large`
- Timeout: `8h`
- Output repo: `JosueG/qwen3-adja-tts-1p7b`
- Final status: `ERROR`
- Retrieve logs: `hf jobs logs 69e3942dac288e522d8efd40`

Notes:

- This full run was launched before the `1.7B` smoke finished because the user explicitly preferred faster testing over conservative compute usage.
- The `1.7B` smoke and `1.7B` full pilot currently share:
  - output repo: `JosueG/qwen3-adja-tts-1p7b`
  - results repo prefix: `JosueG/qwen3-adja-results/qwen3_tts_adja/...`
- Consequence: whichever job finishes last can overwrite repo-level TTS artifacts. That is acceptable for this iteration but should be separated in a future cleanup if artifact provenance matters more than speed.
- Failure signature:
  - `soundfile.LibsndfileError: Format not recognised`

### 2026-04-19 — Attempt 10: held-out ASR scoring + TTS generation fix + L40 reruns [COMPLETED]

- Requested by: Josue Godeme
- Local date of patch: Sunday, April 19, 2026 (`America/New_York`)
- Local branch at patch time: `codex/omni-7b-finetune`
- Goal:
  - compute actual held-out `WER` / `CER` for the completed Qwen ASR pilot rather than relying on training loss only
  - fix the TTS sample-generation failure `soundfile.LibsndfileError: Format not recognised`
  - move new TTS reruns from `a100-large` to `l40sx1` because A100 queue latency was worse than the work itself

Code changes made:

- Added `scripts/hf_jobs/qwen3_asr_eval.py`
- Patched `scripts/hf_jobs/qwen3_tts_adja.py` so sample generation unwraps the model output into a 1-D float waveform and writes with explicit `format="WAV"`
- Changed the default TTS flavor in `scripts/hf_jobs/submit_qwen3_jobs.py` to `l40sx1`

ASR evaluation submissions:

- Eval attempt 1: `69e4d95dac288e522d8f0016`
  - Status: `ERROR`
  - Failure signature: `TypeError: Unsupported audio input type: <class 'dict'>`
  - Why: the scorer passed the HF dataset `Audio` dict directly into `Qwen3ASRModel.transcribe()`
- Eval attempt 2: `69e4da70cd8c002f31dff60d`
  - Status: `ERROR`
  - Failure signature: `ImportError: To support encoding audio data, please install 'torchcodec'.`
  - Why: `datasets.cast_column(Audio(...))` pulled in an avoidable `torchcodec` dependency inside the HF runtime
- Eval attempt 3: `69e4daddac288e522d8f001f`
  - Status: `ERROR`
  - Failure signature: `TypeError: Unsupported audio input type: <class 'numpy.ndarray'>`
  - Why: Qwen ASR inference here still did not accept raw arrays; the working path in this repo is audio file paths
- Eval attempt 4: `69e4db9cac288e522d8f001f`
  - Final status: `COMPLETED`
  - Fix: materialize the held-out test split to job-local 16 kHz WAV files, then call `transcribe(audio=audio_path)` the same way the repo's Qwen inference scripts do
  - Final held-out metrics:
    - raw WER `100.0%`
    - raw CER `53.14%`
    - normalized WER `100.65%`
    - normalized CER `52.26%`

TTS reruns:

- Canceled A100 smoke: `69e4d95dcd8c002f31dff5ff`
- Canceled A100 full: `69e4d95dcd8c002f31dff601`
- Completed L40 smoke: `69e4da24ac288e522d8f001a`
  - Flavor: `l40sx1`
  - Final status: `COMPLETED`
- Completed L40 full: `69e4da24ac288e522d8f0018`
  - Flavor: `l40sx1`
  - Final status: `COMPLETED`

Notes:

- This is still the same Qwen TTS `1.7B` experiment line; the only runtime shift is hardware flavor plus the sample-write fix.
- The ASR evaluator failures were all interface mismatches, not model-quality failures.
- The current ASR evaluator revision follows the repo-proven inference contract:
  - resample to 16 kHz
  - write WAVs locally
  - pass file paths into `Qwen3ASRModel.transcribe()`
- The waveform-write patch resolved the previous late-stage generation/export failure:
  - smoke rerun wrote `checkpoint-epoch-0`, `sample.wav`, and `sample.json`
  - full rerun wrote pilot `checkpoint-epoch-0` and pilot sample metadata
  - eager-mode full training loss was observed from `9.0878` at step 0 down to `4.7412` by step 630
