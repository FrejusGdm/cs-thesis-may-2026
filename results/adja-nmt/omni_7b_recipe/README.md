# Omni 7B Recipe Runs

## Summary

This log tracks the dedicated OmniASR 7B recipe fine-tuning cycle launched from the latest `main` recipe path.

## CTC 7B

- `Omni_AJG_CTC7B_R1_M2`
  - Job: `69e4dc5ecd8c002f31dff619`
  - Status: canceled
  - Failure: CUDA init warning, fairseq2 bound to `cpu`
- `Omni_AJG_CTC7B_R2_M2`
  - Job: `69e4dd98cd8c002f31dff621`
  - Status: canceled
  - Failure: same CPU-init startup issue
- `Omni_AJG_CTC7B_R3_M2`
  - Job: `69e4de80cd8c002f31dff62b`
  - Status: completed
  - Flavor: `h200`
  - Result repo path: `JosueG/adja-asr-results/Omni_AJG_CTC7B_R3_M2_ctc_1776606884/`
  - Final metrics:
    - dev: WER `91.86`, CER `49.15`
    - test: WER `92.48`, CER `47.63`
  - Eval meta:
    - `expected_total=310`
    - `used_pairs=310`

## LLM 7B

- `Omni_AJG_LLM7B_R1`
  - Job: `69e4dc45cd8c002f31dff617`
  - Status: failed
  - Flavor: `h200`
  - Failure: CUDA OOM during optimizer step
- `Omni_AJG_LLM7B_R2_A4_E50k`
  - Job: `69e4e4c0cd8c002f31dff665`
  - Status: failed
  - Flavor: `h200`
  - Changed knobs:
    - `MAX_AUDIO_SEC=4`
    - `MAX_NUM_ELEMENTS=50000`
    - `GRAD_ACCUM=64`
  - Failure: still CUDA OOM during optimizer step
- `Omni_AJG_LLM7B_R3_x2`
  - Job: `69e520beac288e522d8f009f`
  - Status: failed
  - Flavor: `h200x2`
  - Failure: launcher packaging miss (`scripts/hf_jobs/omni_asr_finetune_recipe.py` missing in-container)
- `Omni_AJG_LLM7B_R4_x2_repo`
  - Job: `69e5213cac288e522d8f00a1`
  - Status: failed
  - Flavor: `h200x2`
  - Rationale: keep the original 8-second context and fix memory with multi-GPU sharding instead of more aggressive truncation
  - Note: explicitly submitted through `--repo JosueG/hf-cli-jobs-uv-run-scripts` so the recipe launcher is actually present in the job
  - Failure: still CUDA OOM during the Adam optimizer step; `h200x2` was not enough
- `Omni_AJG_LLM7B_R5_h200x4_repo`
  - Job: `69e52820ac288e522d8f00b1`
  - Status: failed
  - Flavor: `h200x4`
  - Rationale: preserve the original 7B LLM training distribution and increase FSDP sharding headroom instead of cutting context again
  - Runtime hint: adds `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - Failure: still CUDA OOM during the Adam optimizer step
  - Root cause evidence: the job log reported `Running on 1 process(es)`, so the wrapper never actually launched a multi-process distributed job
- `Omni_AJG_LLM7B_R6_h200x4_torchrun`
  - Job: `69e52eccac288e522d8f00b9`
  - Status: failed
  - Flavor: `h200x4`
  - Rationale: same 7B LLM recipe as `R5`, but with a patched launcher that uses `torch.distributed.run` when multiple GPUs are visible
  - Outcome: confirmed real 4-process launch (`WORLD_SIZE=4`) and FSDP wrap/broadcast, then failed with `ChildFailedError` from rank `1`
  - Failure class: distributed child failure, exact traceback hidden by default torchrun logging
- `Omni_AJG_LLM7B_R7_h200x4_torchrunlogs`
  - Job: `69e53548cd8c002f31dff936`
  - Status: completed
  - Flavor: `h200x4`
  - Rationale: same 7B LLM recipe as `R6`, but with rank-level `torchrun` log capture enabled so the next distributed failure, if any, should expose the actual child traceback
  - Result repo path: `JosueG/adja-asr-results/Omni_AJG_LLM7B_R7_h200x4_torchrunlogs_llm_1776629102/`
  - Final metrics:
    - dev: WER `85.71`, CER `44.84`
    - test: WER `87.49`, CER `43.22`
  - Eval meta:
    - `expected_total=320`
    - `used_pairs=320`

## Takeaway so far

- The stricter single-GPU memory settings did not produce a usable result because the LLM 7B path was bottlenecked by optimizer/model-state memory.
- For this recipe, the binding constraint appears to be optimizer/model-state memory rather than just sequence-length activations.
- That means harsher truncation is likely the wrong long-term fix for LLM 7B quality. The winning change was a real multi-process FSDP launch, not more task distortion.
- Final outcome:
  - CTC 7B finished at test CER `47.63`
  - LLM 7B finished at test CER `43.22`
  - 7B LLM is now the best Omni result in the repo, but it still does not approach the repo-best non-Omni systems.
