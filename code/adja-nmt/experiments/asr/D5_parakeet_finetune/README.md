# D5: Parakeet TDT Fine-Tuning for Adja ASR

## Overview

Fine-tune NVIDIA's **Parakeet TDT 0.6B v3** ASR model on the Adja speech corpus.
Parakeet TDT is a Token-and-Duration Transducer (an RNN-T variant) trained by
NVIDIA on >120k hours of English audio. This experiment adapts it to Adja
(`aj_Latn`).

The original code in `hpc/` was provided by a collaborator and was written for
**Dartmouth Discovery HPC** (SLURM + Apptainer/Singularity, NeMo container).
This README documents both the original HPC path and the new **HuggingFace
Jobs** path that was added for parity with the rest of the repo's experiments.

## Model

- **Name**: `nvidia/parakeet-tdt-0.6b-v3`
- **Hub**: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- **Architecture**: FastConformer encoder + Token-and-Duration Transducer
  decoder/joint (a streaming-friendly RNN-T variant that emits both a token
  and a duration per step).
- **Parameters**: ~600M
- **Training data (pretraining)**: ~120k hours of English speech (LibriSpeech,
  Common Voice, GigaSpeech, etc.).
- **Tokenizer**: SentencePiece BPE, English-only.

### Papers

- Parakeet TDT (model card cites): TDT — *Token-and-Duration Transducer for
  Efficient Speech Recognition*, Xu et al. 2023,
  <https://arxiv.org/abs/2304.06795>
- FastConformer encoder: Rekesh et al. 2023,
  <https://arxiv.org/abs/2305.05084>
- RNN-Transducer (parent architecture): Graves 2012,
  <https://arxiv.org/abs/1211.3711>
- NVIDIA NeMo toolkit:
  <https://github.com/NVIDIA/NeMo>

### Why this model for Adja (and known caveats)

Pros:
- Parakeet is currently top of the OpenASR leaderboard for English; the
  encoder is a strong general-purpose speech feature extractor.
- The TDT head is faster than CTC + LM at inference and copes well with
  long utterances.

Caveats — **important for low-resource Adja**:
- The pretraining is **English-only**. Unlike Whisper or MMS, Parakeet has
  zero exposure to Adja phonotactics during pretraining, so we should expect
  it to underperform multilingual baselines (C2 Whisper, C3 MMS, C4 XLS-R)
  in the small-data regime.
- The SentencePiece tokenizer was trained on English text. Adja-specific
  characters (`ɛ`, `ɔ`, `ŋ`, `ɖ`, tone marks) will likely fall back to
  byte-fallback tokens or be split into many subwords, which inflates target
  sequence length and hurts data efficiency.
- A proper Adja fine-tune would re-train the tokenizer on Adja text and
  resize embeddings — this script does **not** do that yet. Expect the
  baseline numbers to be a high-WER reference point, not the best possible
  Parakeet+Adja pipeline.

These caveats are documented up-front so that the first run is interpreted
correctly. See `learnings-from-the-past/` for analogous tokenizer issues
encountered with other models.

## Data

- HuggingFace dataset: `JosueG/adja-tts-orpheus` (private, ~1.6k utterances).
- Loaded as HF Audio features (in-memory arrays + `sampling_rate`).
- Splits are reproduced from the project standard (80/10/10, seed=42) — see
  `scripts/hf_jobs/parakeet_finetune.py` and the project's
  `experiments/asr/shared/data_prep.py`.
- For NeMo, the HF samples are **materialized to 16 kHz mono WAV** files in
  the job's local scratch (`/tmp`) and a NeMo JSON-lines manifest is
  generated with `audio_filepath`, `text`, `duration`. The manifests live
  alongside the WAVs so the full data layout is reproducible from one run.

## Two execution paths

### Path A — HuggingFace Jobs (preferred default)

Self-contained launcher: `scripts/hf_jobs/parakeet_finetune.py`.

We run this as **two stages** on HF Jobs. The UV default image
(`uv:python3.12-bookworm`) has the CUDA runtime but no NVCC, and NeMo's
normal fused TDT loss JIT-compiles via numba and needs `libnvvm.so` from
NVCC. Fighting that inside Bookworm is a losing battle (see
`results/parakeet_hf_jobs/README.md` for the architectural conclusion after
~6 failed attempts), so we split:

- **A.1 smoke** on `hf jobs uv run` with NeMo's pure-PyTorch TDT fallback
  (`tdt_pytorch` — slow, debug-only per NeMo docs) just to validate the
  end-to-end script cheaply on HF's exact UV runtime.
- **A.2 pilot** on `hf jobs run` with the public
  `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` image (has NVCC + cuDNN),
  using the normal `tdt` loss, and *the same* script bytes that the smoke
  validated (via the immutable `UV_SCRIPT_URL` from `hf jobs inspect`).

`HF_TOKEN` is required for both stages — the dataset
(`JosueG/adja-tts-orpheus`) is private, so even `DRY_RUN=1` needs it.

#### Path A.1: UV smoke (cheap validation, fallback loss)

```bash
export HF_TOKEN=hf_...

hf jobs uv run \
    --flavor a10g-small \
    --timeout 2h \
    --secrets HF_TOKEN \
    -p 3.11 \
    -e DRY_RUN=1 \
    -e EXP_ID=D5_smoke \
    -e RNNT_LOSS_NAME=tdt_pytorch \
    scripts/hf_jobs/parakeet_finetune.py
```

Expected log signals on success:

- `[versions] torch=... nemo=... lightning=... numba=...`
- `[loss-swap] original=... -> selected=tdt_pytorch`
- `[loss-swap] effective_kwargs={'durations': [...], 'sigma': ...}`
- `trainer.fit` completes the 1-epoch run on 4 samples
- Eval runs, then `*** DRY_RUN complete — skipping Hub upload ***`

This path is for validation only — `tdt_pytorch` is explicitly documented
as slow and debug-only. Do not use for real training.

#### Path A.2: Docker pilot (real run, real loss)

After a clean smoke, extract the uploaded script URL and submit the pilot
against a public CUDA devel image:

```bash
SMOKE_JOB=<smoke_job_id>
SCRIPT_URL=$(hf jobs inspect "$SMOKE_JOB" \
  | python -c 'import json,sys; print(json.load(sys.stdin)[0]["environment"]["UV_SCRIPT_URL"])')

hf jobs run \
    --flavor a100-large \
    --timeout 24h \
    --secrets HF_TOKEN \
    -e EXP_ID=D5 \
    -e MAX_EPOCHS=20 \
    -e BATCH_SIZE=8 \
    -e EVAL_BATCH_SIZE=16 \
    -e RNNT_LOSS_NAME=tdt \
    -e SCRIPT_URL="$SCRIPT_URL" \
    pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel \
    bash -lc 'python -m pip install --no-cache-dir "setuptools<80" "nemo_toolkit[asr]==2.0.0" datasets soundfile librosa "numpy<2" pandas huggingface-hub jiwer "lightning>=2.2,<2.4" && python -c "import os, urllib.request; from pathlib import Path; req = urllib.request.Request(os.environ[\"SCRIPT_URL\"], headers={\"Authorization\": \"Bearer \" + os.environ[\"HF_TOKEN\"]}); Path(\"/tmp/parakeet_finetune.py\").write_bytes(urllib.request.urlopen(req).read())" && python /tmp/parakeet_finetune.py'
```

Why not `nvcr.io/nvidia/nemo:25.02` directly? HF Jobs' CLI doesn't expose
NGC registry auth, so the NeMo container can't be pulled from here. The
public `pytorch:*-devel` image gives us NVCC + libnvvm, and we just `pip
install nemo_toolkit[asr]==2.0.0` on top.

Fallback if `UV_SCRIPT_URL` is not retrievable from `hf jobs inspect`
(depends on `huggingface_hub` CLI version): upload the edited
`parakeet_finetune.py` to a personal Hub repo (e.g.
`JosueG/hf-cli-jobs-uv-run-scripts`) and use that raw Hub URL as
`SCRIPT_URL`.

Key environment variables:

| Var                | Default                                      | Notes                                                |
|--------------------|----------------------------------------------|------------------------------------------------------|
| `HF_TOKEN`         | —                                            | **Required** (dataset is private, incl. for DRY_RUN) |
| `EXP_ID`           | `D5`                                         | Tag used for the results-repo subfolder              |
| `MODEL_NAME`       | `nvidia/parakeet-tdt-0.6b-v3`                |                                                      |
| `RNNT_LOSS_NAME`   | `tdt_pytorch` under UV, else `tdt`           | `tdt_pytorch` = slow PyTorch fallback; `tdt` = fused |
| `MAX_EPOCHS`       | `20`                                         |                                                      |
| `BATCH_SIZE`       | `8`                                          | Training batch size                                  |
| `LR`               | `1e-4`                                       |                                                      |
| `WARMUP_STEPS`     | `500`                                        |                                                      |
| `EVAL_BATCH_SIZE`  | `16`                                         |                                                      |
| `DRY_RUN`          | `0`                                          | Set to `1` for a 4-sample / 1-epoch smoke            |

Outputs go to `JosueG/adja-asr-results/D5_parakeet/` on the Hub:
`metrics.json`, `evaluation_summary.txt`, `test_results.csv`, and the best
`.nemo` checkpoint when small enough to upload.

### Path B — Dartmouth HPC (original collaborator setup)

The original code lives in [`hpc/`](hpc/). It expects:

- A pre-built Apptainer/Singularity image (`parakeet_finetune.sif`) at
  `<HPC_WORKDIR>`. Build it with the
  job script `hpc/make-sif.script` from the definition file
  `hpc/parakeet_finetune.def` (based on `nvcr.io/nvidia/nemo:25.02`).
- CSV manifests with columns `audio_filepath`, `text`, optional `duration`.
- Audio WAV files reachable on a bind-mounted directory.

Submit a finetune:

```bash
sbatch experiments/asr/D5_parakeet_finetune/hpc/run_finetune_mqr02.slurm
```

Submit a stand-alone evaluation against an existing `.ckpt`:

```bash
sbatch experiments/asr/D5_parakeet_finetune/hpc/run_eval.slurm
```

Edit the `USER CONFIGURATION` block at the top of each `.slurm` to point at
your CSVs, audio directory, and output directory. See `hpc/README.md` for
the full HPC walkthrough.

## What the script does (both paths)

1. Build NeMo manifests (`train`, `dev`, `test`) from CSVs (HPC) or from
   the HF dataset (HF Jobs).
2. Load `nvidia/parakeet-tdt-0.6b-v3` via NeMo's `ASRModel.from_pretrained`.
3. Override `train_ds` / `validation_ds` config with our manifests, batch
   size, and `[min, max] = [0.1, 20.0]s` duration filter.
4. Configure AdamW + CosineAnnealing with `warmup_steps`, train with
   bf16-mixed precision, `val_wer` checkpointing via NeMo `exp_manager`.
5. Reload the best checkpoint, run `transcribe()` on the test set, and
   compute per-sentence WER/CER (jiwer) plus corpus-level WER/CER. Save
   `test_results.csv` and `evaluation_summary.txt`.

## Expected outputs

- `metrics.json` — best epoch, WER/CER, runtime, training history
- `evaluation_summary.txt` — corpus and per-sentence WER/CER table
- `test_results.csv` — per-utterance reference / hypothesis / WER / CER
- `best_model/` — `.nemo` checkpoint (if size permits Hub upload)

## TODOs / next iteration

- Extend / retrain the SentencePiece tokenizer on Adja text before
  fine-tuning, then `model.resize_token_embeddings(...)` (analogous to the
  `aj_Latn` fix documented in `learnings-from-the-past/`).
- Try freezing the encoder for the first N epochs (uncomment the
  `asr_model.encoder.freeze()` block in `finetune_parakeet.py`).
- Unicode NFC normalisation on `text` before manifest emission, matching the
  rest of the ASR experiments.
