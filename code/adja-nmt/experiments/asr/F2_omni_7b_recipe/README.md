# F2: OmniASR 7B Recipe Fine-tuning on Adja

## Goal

Reuse the working Omni recipe path from `main` to test the largest available Omni v2 checkpoints on Adja:

- `omniASR_CTC_7B_v2`
- `omniASR_LLM_7B_v2`

This track exists to answer one direct question:

Can scaling Omni from 3B to 7B close the gap to our best ASR system in this repo?

## Baseline to beat

- repo-best normalized test CER: `22.67%` (`D4_C4v2_lm_optuna`)
- best Omni result so far before 7B:
  - LLM: `Omni_AJG_LLM3B_R1` test CER `44.57`
  - CTC: `Omni_AJG_CTC3B_R5_M2` test CER `51.75`

## What actually happened

### CTC 7B

- `R1` — `69e4dc5ecd8c002f31dff619`
  - failed at startup: CUDA warning, fairseq2 bound to `cpu`
- `R2` — `69e4dd98cd8c002f31dff621`
  - same startup failure as `R1`
- `R3` — `69e4de80cd8c002f31dff62b`
  - completed on `h200`
  - result path: `JosueG/adja-asr-results/Omni_AJG_CTC7B_R3_M2_ctc_1776606884/`
  - final metrics:
    - dev: WER `91.86`, CER `49.15`
    - test: WER `92.48`, CER `47.63`
  - eval meta: `expected_total=310`, `used_pairs=310`

### LLM 7B

- `R1` — `69e4dc45cd8c002f31dff617`
  - healthy GPU startup on `h200`
  - failed with CUDA OOM during optimizer step
  - settings: `MAX_AUDIO_SEC=8`, `MAX_NUM_ELEMENTS=100000`, `GRAD_ACCUM=32`
- `R2` — `69e4e4c0cd8c002f31dff665`
  - still failed with CUDA OOM on `h200`
  - stricter settings: `MAX_AUDIO_SEC=4`, `MAX_NUM_ELEMENTS=50000`, `GRAD_ACCUM=64`
- `R3` — `69e520beac288e522d8f009f`
  - failed before training on `h200x2`
  - launcher packaging issue: script path was missing in-container
- `R4` — `69e5213cac288e522d8f00a1`
  - failed on `h200x2`
  - returned to the more faithful original settings:
    - `MAX_AUDIO_SEC=8`
    - `MAX_NUM_ELEMENTS=100000`
    - `GRAD_ACCUM=32`
  - rationale: the OOM was in optimizer state, so adding GPUs is a cleaner fix than shrinking context further
  - note: `R4` explicitly uses the HF jobs script repo upload path after `R3` exposed a launcher packaging miss
  - result: still hit CUDA OOM in the Adam optimizer step, so `h200x2` FSDP was still not enough
- `R5` — `69e52820ac288e522d8f00b1`
  - failed on `h200x4`
  - keeps the same core settings as `R4`:
    - `MAX_AUDIO_SEC=8`
    - `MAX_NUM_ELEMENTS=100000`
    - `GRAD_ACCUM=32`
  - change vs `R4`: deeper sharding headroom instead of more task truncation
  - extra runtime hint: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - result: still hit CUDA OOM in the Adam optimizer step
  - root cause: the wrapper still launched a single process, so the job log showed `Running on 1 process(es)` even on `h200x4`
- `R6` — `69e52eccac288e522d8f00b9`
  - failed on `h200x4`
  - same 7B LLM training distribution as `R5`
  - change vs `R5`: patched launcher uses `torch.distributed.run` when multiple GPUs are visible, so FSDP finally launched across 4 processes
  - result: no OOM signature; logs show `WORLD_SIZE=4`, multi-process gang creation, model load, and FSDP wrap/broadcast completed before a rank-1 child exited with code `1`
  - current blocker: HF only surfaced `ChildFailedError`, not the underlying rank-1 traceback
- `R7` — `69e53548cd8c002f31dff936`
  - completed on `h200x4`
  - same 7B LLM training distribution as `R6`
  - change vs `R6`: launcher now adds `torchrun` rank-log capture (`--log-dir`, `--redirects 3`, `--tee 3`) and prints per-rank failure logs back into the HF job log if the distributed run fails again
  - result path: `JosueG/adja-asr-results/Omni_AJG_LLM7B_R7_h200x4_torchrunlogs_llm_1776629102/`
  - final metrics:
    - dev: WER `85.71`, CER `44.84`
    - test: WER `87.49`, CER `43.22`
  - eval meta: `expected_total=320`, `used_pairs=320`

## Final interpretation

- CTC 7B is real and reproducible now.
- CTC 7B improved over AJG CTC 3B (`51.75 -> 47.63` test CER), but it still does not beat Omni 3B LLM (`44.57` test CER).
- LLM 7B only became viable once the wrapper launched a real multi-process FSDP job. Bigger HF flavors alone were not enough.
- `R7` is the best Omni result in this repo so far:
  - better than AJG Omni 3B LLM (`44.57 -> 43.22` test CER)
  - better than the earlier 3B forensic no-drop LLM run (`44.11 -> 43.22` test CER)
- The improvement is real but modest. Omni 7B still remains far behind the repo-best non-Omni ASR path (`22.67` normalized CER with XLS-R + LM, and mid-20s CER acoustic baselines).
