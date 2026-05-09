# Adja Qwen3 Hugging Face Runbook

Prepared for: Josue Godeme
Requested by: Josue Godeme
Maintainer note: this document tracks the first Hugging Face submission path for Qwen3-based Adja ASR and TTS so later review can tie a remote job back to a specific local code revision and upstream source.

This is the first-success path for running `Qwen3-ASR` and `Qwen3-TTS` against the private Adja dataset on Hugging Face jobs.

## Scope

- Keep `experiments/finetuning-qwen3/` in place
- Use `JosueG/adja-tts-orpheus` as the source of truth
- Reproduce split seed `42` inside the job
- Use `Qwen/Qwen3-ASR-0.6B` first for ASR
- TTS started on `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, but the first real failure exposed a `2048` vs `1024` embedding-width mismatch; the next active smoke run moves to `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, which is also the default in the vendored upstream fine-tuning script

## Provenance

- Local implementation owner: Josue Godeme project workspace
- Local implementation files:
  - `experiments/finetuning-qwen3/data_pipeline/adja_hf_materialize.py`
  - `scripts/hf_jobs/qwen3_asr_adja.py`
  - `scripts/hf_jobs/qwen3_tts_adja.py`
  - `experiments/finetuning-qwen3/training/hf/train_asr.py`
  - `experiments/finetuning-qwen3/training/hf/train_tts.py`
- Vendored upstream files:
  - `experiments/finetuning-qwen3/vendor/qwen3_tts_official/prepare_data.py`
  - `experiments/finetuning-qwen3/vendor/qwen3_tts_official/dataset.py`
  - `experiments/finetuning-qwen3/vendor/qwen3_tts_official/sft_12hz.py`

## Sources

- Qwen3-TTS upstream finetuning files from `QwenLM/Qwen3-TTS`:
  - `finetuning/prepare_data.py`
  - `finetuning/dataset.py`
  - `finetuning/sft_12hz.py`
- Qwen3-ASR upstream finetuning reference from `QwenLM/Qwen3-ASR`
- Hugging Face CLI local help:
  - `hf jobs run --help`
  - `hf jobs uv run --help`
- Repo references:
  - `experiments/finetuning-qwen3/docs/01_repository_forensics.md`
  - `experiments/finetuning-qwen3/docs/02_infrastructure_decision_matrix.md`
  - `learnings-from-the-past/hf-jobs-gotchas.md`

## Entrypoints

- ASR: `scripts/hf_jobs/qwen3_asr_adja.py`
- TTS: `scripts/hf_jobs/qwen3_tts_adja.py`
- Materialization layer: `experiments/finetuning-qwen3/data_pipeline/adja_hf_materialize.py`
- Local submit helper: `scripts/hf_jobs/submit_qwen3_jobs.py`

## Correct HF submission pattern

Use the repo's normal HF Jobs flow:

- run from the repo root
- submit the local launcher script with `hf jobs uv run`
- let HF upload the local workspace context
- do **not** clone `adja-nmt` from inside the container

ASR smoke:

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

TTS smoke:

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

Convenience helper:

```bash
python scripts/hf_jobs/submit_qwen3_jobs.py --modality both --smoke --execute
python scripts/hf_jobs/submit_qwen3_jobs.py --modality asr --pilot --execute --detach
python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --smoke --tts-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-output-repo-id JosueG/qwen3-adja-tts-1p7b --tts-flavor l40sx1 --execute --detach
python scripts/hf_jobs/submit_qwen3_jobs.py --modality tts --pilot --tts-model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-output-repo-id JosueG/qwen3-adja-tts-1p7b --tts-flavor l40sx1 --execute --detach
```

## Required Environment

```bash
export HF_TOKEN=...
export HF_DATASET_ID=JosueG/adja-tts-orpheus
export MODEL_ID=Qwen/Qwen3-ASR-0.6B            # or Qwen/Qwen3-TTS-12Hz-0.6B-Base
export OUTPUT_REPO_ID=your-user/qwen3-...-model
export RESULTS_REPO_ID=your-user/qwen3-...-results
export SMOKE_RUN=1                              # set 0 to run smoke then pilot
export WORKSPACE_DIR=/tmp/qwen3_adja
export SEED=42
```

## What The Jobs Do

1. Resolve runtime dependencies through inline `uv` script metadata
2. Verify private dataset access
3. Materialize the dataset into job-local WAVs and JSONL files
4. Validate schema and sample rates
5. Run a smoke train
6. Run one inference/generation sample
7. Optionally run a bounded pilot on the full split
8. Upload model artifacts and run metadata

## Data Contracts

ASR JSONL:

```json
{"audio": "/tmp/qwen3_adja/asr/wavs/train/train_00000.wav", "text": "language None<asr_text>..."}
```

TTS JSONL before codec prep:

```json
{"audio": "/tmp/qwen3_adja/tts/wavs/train/train_00000.wav", "text": "...", "ref_audio": "/tmp/qwen3_adja/tts/reference.wav"}
```

## Notes

- The TTS wrapper uses vendored upstream Qwen3-TTS fine-tuning files under `experiments/finetuning-qwen3/vendor/qwen3_tts_official/`.
- The Adja first pass stays single-speaker for TTS by selecting one deterministic reference clip from the train split.
- SageMaker is only a fallback if the HF runtime cannot sustain the TTS dependency or training stack.
- Every submission should also update:
  - `experiments/registry.md`
  - `results/run-ledger.md`
  - a results note under `results/qwen3_hf_jobs/`

## Submission history

- Attempt 1: container-side clone bootstrap
  - ASR job: `69e30a7bac288e522d8efb5c`
  - TTS job: `69e30a8eac288e522d8efb5e`
  - Result: failed before Python startup with `fatal: could not read Username for 'https://github.com'`
- Attempt 2: repo-native `hf jobs uv run`, but launcher still imported a local helper that was not uploaded by HF
  - ASR job: `69e382dfac288e522d8efd10`
  - TTS job: `69e382e1cd8c002f31dfe942`
  - Result: failed with `ModuleNotFoundError: No module named 'qwen3_job_utils'`
- Attempt 3: standalone launchers, but they still expected `pip` inside the UV image
  - ASR job: `69e384d0cd8c002f31dfe952`
  - TTS job: `69e384d1cd8c002f31dfe954`
  - Result: failed with `/usr/bin/python3.11: No module named pip`
- Attempt 4: `uv pip --system`
  - ASR job: `69e38514cd8c002f31dfe956`
  - TTS job: `69e38515ac288e522d8efd1c`
  - Result:
    - ASR: `qwen-asr>=0.1.0` was not available; HF resolved only up to `qwen-asr==0.0.6`
    - TTS: package install completed, but the runtime still missed `librosa`, which showed the install went into the wrong interpreter context
- Attempt 5: `uv pip --python sys.executable`
  - ASR job: `69e3856aac288e522d8efd1e`
  - TTS job: `69e3856bcd8c002f31dfe964`
  - Result: failed because `/usr` is externally managed; direct package bootstrap remained the wrong pattern
- Attempt 6: inline `uv` metadata, matching the repo's working HF script pattern
  - ASR job: `69e3865bac288e522d8efd24`
  - TTS job: `69e3865dcd8c002f31dfe96e`
  - Result:
    - ASR reached dataset download, split/materialization, and model load, then failed with `TypeError("TrainingArguments.__init__() got an unexpected keyword argument 'evaluation_strategy'")`
    - TTS reached dataset download, split/materialization, and model load, then failed after the flash-attention fallback with `RuntimeError('The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 2')`
  - Inspect:
    - `hf jobs inspect 69e3865bac288e522d8efd24`
    - `hf jobs inspect 69e3865dcd8c002f31dfe96e`
  - Logs:
    - `hf jobs logs 69e3865bac288e522d8efd24`
    - `hf jobs logs 69e3865dcd8c002f31dfe96e`
- Attempt 7: ASR `TrainingArguments` compatibility fix + fuller TTS diagnostics
  - ASR job: `69e388ccac288e522d8efd26`
  - TTS job: `69e388cdcd8c002f31dfe98a`
  - Result:
    - ASR smoke training completed, saved a checkpoint, and produced a sample decode, but the HF job still exited `ERROR` because the launcher closed the tee log handle before interpreter shutdown finished
    - TTS 0.6B still failed on the real training path with `RuntimeError('The size of tensor a (2048) must match the size of tensor b (1024) at non-singleton dimension 2')`
  - Inspect:
    - `hf jobs inspect 69e388ccac288e522d8efd26`
    - `hf jobs inspect 69e388cdcd8c002f31dfe98a`
  - Logs:
    - `hf jobs logs 69e388ccac288e522d8efd26`
    - `hf jobs logs 69e388cdcd8c002f31dfe98a`
- Attempt 8: launcher teardown fix + ASR full run + TTS 1.7B smoke
  - ASR job: `69e38fe4ac288e522d8efd32`
  - TTS job: `69e38fe3ac288e522d8efd30`
  - Final statuses:
    - ASR `69e38fe4ac288e522d8efd32`: `COMPLETED`
    - TTS A100 smoke `69e38fe3ac288e522d8efd30`: `ERROR`
  - Why this split:
    - ASR already proved the smoke path; the next rational step is the full pilot
    - TTS did not justify a full 0.6B pilot because the failure was architectural, not incidental
    - The vendored upstream `sft_12hz.py` defaults to `Qwen/Qwen3-TTS-12Hz-1.7B-Base`, so `1.7B` is the next source-backed retry rather than another blind `0.6B` repeat
  - Inspect:
    - `hf jobs inspect 69e38fe4ac288e522d8efd32`
    - `hf jobs inspect 69e38fe3ac288e522d8efd30`
  - Logs:
    - `hf jobs logs 69e38fe4ac288e522d8efd32`
    - `hf jobs logs 69e38fe3ac288e522d8efd30`
- Attempt 9: TTS 1.7B full pilot launched in parallel with smoke
  - TTS full job: `69e3942dac288e522d8efd40`
  - Final status: `ERROR`
  - Why:
    - the `1.7B` smoke had already reached real training with matching embedding widths (`2048` / `2048`)
    - the user explicitly optimized for faster testing over conservative compute usage
  - Caveat:
    - the `1.7B` smoke and `1.7B` full pilot currently write to the same output repo and the same results-repo prefix, so last writer wins at the repo level
  - Failure signature:
    - `soundfile.LibsndfileError: Format not recognised`
  - Inspect:
    - `hf jobs inspect 69e3942dac288e522d8efd40`
  - Logs:
    - `hf jobs logs 69e3942dac288e522d8efd40`
- Launch revision:
  - Branch: `codex/qwen3-hf-jobs`
  - Commit: `538d415`
- Attempt 10: held-out ASR scoring + TTS generation patch on Sunday, April 19, 2026 (`America/New_York`)
  - Local patch branch: `codex/omni-7b-finetune`
  - Why:
    - ASR full fine-tuning completed, but there was still no real held-out `WER` / `CER`
    - TTS `1.7B` training got past the old width mismatch, but generation still failed when writing `sample.wav`
    - `a100-large` spent too much time in scheduling, so new TTS reruns were moved to `l40sx1`
  - Code:
    - added `scripts/hf_jobs/qwen3_asr_eval.py`
    - patched `scripts/hf_jobs/qwen3_tts_adja.py` to unwrap the generated waveform and write explicit WAV output
    - changed the default TTS flavor in `scripts/hf_jobs/submit_qwen3_jobs.py` to `l40sx1`
  - ASR eval submissions:
    - `69e4d95dac288e522d8f0016` failed because `transcribe()` rejected the HF `Audio` dict
    - `69e4da70cd8c002f31dff60d` failed because `cast_column(Audio(...))` required `torchcodec`
    - `69e4daddac288e522d8f001f` failed because `transcribe()` also rejected raw `numpy.ndarray`
    - `69e4db9cac288e522d8f001f` completed and uses job-local WAV paths, which matches the repo's working Qwen inference path
    - final held-out metrics:
      - raw WER `100.0%`
      - raw CER `53.14%`
      - normalized WER `100.65%`
      - normalized CER `52.26%`
  - TTS reruns:
    - canceled A100 smoke: `69e4d95dcd8c002f31dff5ff`
    - canceled A100 full: `69e4d95dcd8c002f31dff601`
    - completed L40 smoke: `69e4da24ac288e522d8f001a`
    - completed L40 full: `69e4da24ac288e522d8f0018`
  - Final status after the April 19 patch:
    - TTS L40 smoke rerun wrote `checkpoint-epoch-0`, `sample.wav`, and `sample.json`
    - TTS L40 full rerun wrote pilot `checkpoint-epoch-0` and pilot sample metadata
    - eager-mode loss was observed from `9.0878` at step 0 to `4.7412` by step 630
