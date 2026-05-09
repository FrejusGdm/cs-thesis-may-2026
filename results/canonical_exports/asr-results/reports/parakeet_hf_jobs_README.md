# Parakeet TDT 0.6B v3 — HF Jobs submission log

Experiment folder: `experiments/asr/D5_parakeet_finetune/` (HPC + HF Jobs paths)
Launcher script:   `scripts/hf_jobs/parakeet_finetune.py`
Original HPC code: `experiments/asr/D5_parakeet_finetune/hpc/` (collaborator-provided)

## Submission log

### 2026-04-18 — Smoke (DRY_RUN, deps + dataset access only)

- Requested by: Josue Godeme
- Git branch: `codex/qwen3-hf-jobs`
- HF namespace: `JosueG`
- Flavor: `a10g-small` / Python `3.11` / Timeout `1h`
- Dataset: `JosueG/adja-tts-orpheus`
- Env: `DRY_RUN=1 EXP_ID=D5_smoke`
- Purpose: validate the heavy NeMo install
  (`nemo_toolkit[asr]==2.0.0` + torch + CUDA), dataset access, and a 4-sample
  materialize → manifest → 1-epoch train + eval round-trip. No Hub upload.

Attempt 1 — **failed** (good signal, easy fix):

- Job ID: `69e37e2ecd8c002f31dfe8fc` ·
  <https://huggingface.co/jobs/JosueG/69e37e2ecd8c002f31dfe8fc>
- Got past dep install (161 packages), dataset load (1597 utts), WAV
  materialization, then crashed on `import lightning.pytorch as pl` with
  `ModuleNotFoundError: No module named 'pkg_resources'` because
  `lightning>=2.2` does `import pkg_resources` at module top, and uv's
  `python:3.11-bookworm` image does NOT preinstall `setuptools`.
- Fix: added `setuptools>=68` to the UV header in
  `scripts/hf_jobs/parakeet_finetune.py`.

Attempt 2 — **resubmitted**:

- Job ID: `69e380d4cd8c002f31dfe912` ·
  <https://huggingface.co/jobs/JosueG/69e380d4cd8c002f31dfe912>
- Retrieve logs: `hf jobs logs 69e380d4cd8c002f31dfe912`

### Architectural conclusion (2026-04-18)

After Attempt 2 cleared the `pkg_resources` error, subsequent jobs
(`69e38446` / `69e38563` / `69e38656` / `69e3874a` / `69e389ab` /
`69e38cc5` / `69e38e94` / `69e39061`) all died inside numba's NVVM
codepath while trying to JIT-compile NeMo's fused TDT loss — attempted
fixes included:

- installing `nvidia-cuda-nvcc-cu12` to a `--target` dir
- creating the unversioned `libnvvm.so` symlink that pip wheels strip
- setting `NUMBA_CUDA_HOME` / `NUMBA_CUDA_NVVM` / `LD_LIBRARY_PATH`
- preloading libnvvm via `ctypes.CDLL(..., RTLD_GLOBAL)`
- monkey-patching `numba.core.config.CUDA_HOME` and
  `numba.cuda.cudadrv.libs.get_cudalib`

Conclusion: the default `uv:python3.12-bookworm` image is not a viable
runtime for NeMo's fused TDT loss. We stop patching Bookworm and instead
use a two-stage flow:

1. **UV smoke** (cheap validation) — `hf jobs uv run`, fallback loss
   `tdt_pytorch` (pure PyTorch, slow, debug-only per NeMo docs). This
   avoids the numba NVVM path entirely and runs end-to-end on the UV
   image as a sanity check.
2. **Docker pilot** (real run) — `hf jobs run` on the public
   `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` image (has NVCC +
   libnvvm + cuDNN), with the normal fused `tdt` loss, reusing the
   exact script URL the smoke validated (via `hf jobs inspect`).

We don't attempt `nvcr.io/nvidia/nemo:25.02` directly: HF Jobs' CLI
doesn't expose NGC registry auth.

The launcher `scripts/hf_jobs/parakeet_finetune.py` has been updated
accordingly: the numba/NVCC monkey-patch block is removed; a new
`RNNT_LOSS_NAME` env var selects between `tdt_pytorch` (default under
UV, detected via `UV_SCRIPT_URL`) and `tdt` (default elsewhere); and the
loss is swapped post-`from_pretrained` via `RNNTLoss(...)` +
`joint.set_loss(...)`.

### Smoke — UV fallback-loss validation

```bash
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

Expected log signals:
- `[versions] torch=... nemo=... lightning=... numba=...`
- `[loss-swap] original=... -> selected=tdt_pytorch`
- `trainer.fit` reaches and completes the 1-epoch run
- eval runs
- `*** DRY_RUN complete — skipping Hub upload ***`

### Pilot — Docker, real fused loss

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

Fallback if the CLI's `hf jobs inspect` doesn't return `UV_SCRIPT_URL`:
upload the edited launcher to e.g. `JosueG/hf-cli-jobs-uv-run-scripts`
and use its raw Hub URL as `SCRIPT_URL`.

Outputs land in `JosueG/adja-asr-results/D5_parakeet/`:
`metrics.json`, `evaluation_summary.txt`, `test_results.csv`, and best
`.nemo` checkpoint when small enough.

### Future runs (append below)

| Date | Stage | Job ID | Flavor | Loss | Outcome | WER / CER | Notes |
|------|-------|--------|--------|------|---------|-----------|-------|
| 2026-04-18 | smoke | `69e39c0fcd8c002f31dfea2d` | a10g-small | `tdt_pytorch` | pipeline OK | val_wer=1.00 (4 samples) | UV smoke validated path end-to-end; exit 0 (HF flagged as ERROR/Unknown). |
| 2026-04-18 | pilot v1 | `69e3d5b4ac288e522d8efdd1` | l40sx1 | `tdt` | failed | — | `FileNotFoundError: 'git'` — `pytorch:2.5.1-cuda12.1-cudnn9-devel` has no git; NeMo `exp_manager` calls `git rev-parse HEAD`. |
| 2026-04-18 | pilot v2 | `69e406c9ac288e522d8efe49` | l40sx1 | `tdt` | **COMPLETED** | val_wer=**0.6535** (ep 17, dev) ; test Mean/Median WER=100%, Corpus WER=NaN | Added `apt-get install -y git` to bootstrap. 20 epochs / 30.5 min. Fused `tdt` loss works on CUDA 12.1 devel image. Eval hyps are `⁇`-riddled — expected Parakeet-EN tokenizer caveat on Adja (`ɛ ɔ ŋ ɖ`). Next: retrain SentencePiece + `resize_token_embeddings`. Artifacts at `JosueG/adja-asr-results/D5_l40_parakeet/`; 2.51 GB `.nemo` uploaded. |
| 2026-04-18 | smoke v2 | `69e42d8dcd8c002f31dfefa3` | a10g-small | `tdt_pytorch` | canceled | — | DRY_RUN smoke for tokenizer retrain path. Stuck RUNNING at 42 min on UV image (pip-solve + model download); canceled and went direct to pilot since tokenizer code is narrow and pilot fails fast on bugs. |
| 2026-04-18 | pilot v3 | `69e437a7cd8c002f31dff009` | l40sx1 | `tdt` | **COMPLETED** | val_wer=**0.9544** (ep 12, dev); test Median WER=100%, Corpus WER=NaN | `RETRAIN_TOKENIZER=1 VOCAB_SIZE=1024`. 20 epochs / 20.0 min. Tokenizer swap mechanically works end-to-end (SPM trained on 12050 LM + 1277 train = 13327 sents, `change_vocabulary` reshaped `num_classes_with_blank` 8198 → 1030, fused `tdt` loss swapped to `num_classes=1024`). Decodes now emit Adja chars (no `⁇`) but are drastically under-length (HYP=`E` for 5-word references) — the freshly re-initialised joint + decoder embedding can't converge in 20 epochs at LR=1e-4. Dev WER **regressed** +30 pp vs pilot v2: v2 has a trained decoder but no Adja vocab; v3 has Adja vocab but a fresh decoder. See [docs/parakeet-tokenizer-retrain-adja.md §9](../../docs/parakeet-tokenizer-retrain-adja.md) for full analysis + recommended recipe changes (longer training, higher LR on decoder, staged freeze). Artifacts at `JosueG/adja-asr-results/D5_l40_tokv1_parakeet/`; logs archived at `logs/69e437a7_pilot_v3.txt`. |

### Fix applied 2026-04-18 after pilot v1

`scripts/hf_jobs/parakeet_finetune.py` itself was already clean; the issue was
in the `hf jobs run` launch command: the public
`pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` image ships without `git`, and
NeMo's `exp_manager` unconditionally shells out to `git rev-parse HEAD` to
tag the run. Prepend `apt-get update -qq && apt-get install -y -qq git &&` to
the bash bootstrap for future pilots.

### Known eval bug — Corpus WER = NaN

All test hypotheses contain the `⁇` unknown-token marker (English
SentencePiece can't produce Adja characters), which pushes jiwer's
corpus-level aggregation to a zero-division / NaN. Per-sentence Mean/Median
still report sanely (pegged at 100% for the same reason). Not a blocker for
the baseline interpretation; clean this up in the eval helper alongside the
tokenizer retrain.
