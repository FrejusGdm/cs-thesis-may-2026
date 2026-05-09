# TTS Attempt Log

Short operational log for Adja TTS cycles.

Use this for exact runtime facts and failure signatures. Keep paper-style interpretation in `ideas/` and aggregate comparisons in `results/tts-comparison.md`.

## Entry Template

### Cycle X — YYYY-MM-DD

- Model:
- Runtime:
- Entrypoint:
- Dataset split policy:
- Package pins:
- Result:
- Failure signature or success artifact:
- Stage reached:
- Next action:

## Cycle 0 — Recovery Setup

- Model: Sesame CSM / Spark TTS recovery planning
- Runtime: repo setup only
- Entrypoint: re-anchored on `references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb`, with repo automation in `scripts/hf_jobs/T1_sesame_csm_finetune.py`
- Dataset split policy: seed 42, 80/10/10, `Audio(sampling_rate=24000)` for CSM
- Package pins:
  - `unsloth==2025.5.4`
  - `unsloth_zoo==2025.5.6`
  - `transformers==4.52.3`
  - `trl==0.19.1`
- Result: repo recovery implemented, and the source of truth was corrected from the local adapted notebook back to the downloaded Unsloth notebook
- Failure signature or success artifact: identified that the broken branch was the generic text-only CSM flow plus the `use_gradient_checkpointing="unsloth"` path
- Stage reached: canonical notebook/script cleanup completed
- Next action: run one clean Colab T4 CSM dry-run, then either continue or pivot to the vanilla diagnostic and Spark fallback

## Cycle 1 — Install Cell Isolation

- Model: Sesame CSM
- Runtime: local notebook inspection + isolated temp environment
- Entrypoint: `references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb`, then `experiments/tts/T1_sesame_csm_finetune/T1_adja_csm_finetune.ipynb`
- Dataset split policy: not reached
- Package pins:
  - `unsloth==2025.5.4`
  - `unsloth_zoo==2025.5.6`
  - `transformers==4.52.3`
  - `trl==0.22.2`
- Result: confirmed that the install path itself is brittle and needed a visible, stepwise replacement
- Failure signature or success artifact: upstream/local notebook install cells could hide a package abort; this can leave `unsloth_zoo` present while `unsloth` is still missing from the kernel. A later Colab import error showed `unsloth_zoo` also expects `trl.trainer.utils.ConstantLengthDataset`, so the recovery path was repinned from `trl==0.22.2` to `trl==0.19.1`
- Stage reached: Colab-safe install gate added to the local Adja notebook
- Next action: run the updated `T1_adja_csm_finetune.ipynb` on Colab T4 and stop immediately at the new install check if `unsloth` is still missing

## Cycle 2 — Model Load Patch Failure

- Model: Sesame CSM
- Runtime: Google Colab
- Entrypoint: `experiments/tts/T1_sesame_csm_finetune/T1_adja_csm_finetune.ipynb`
- Dataset split policy: not reached
- Package pins:
  - `unsloth==2025.5.4`
  - `unsloth_zoo==2025.5.6`
  - `transformers==4.52.3`
  - `trl==0.19.1`
- Result: install/import passed, but model loading failed inside Unsloth's compiled-autograd patcher
- Failure signature or success artifact: `AttributeError: 'NoneType' object has no attribute 'group'` in `unsloth_zoo.patching_utils.patch_compiled_autograd`
- Stage reached: model-load hotfix prepared to mirror the newer upstream guard before `FastModel.from_pretrained(...)`
- Next action: run the compiled-autograd hotfix cell, then retry the model-load cell

## Cycle 3 — Collator Length Mismatch

- Model: Sesame CSM
- Runtime: Google Colab
- Entrypoint: `experiments/tts/T1_sesame_csm_finetune/T1_adja_csm_finetune.ipynb`
- Dataset split policy: full train split after Adja dataset load
- Package pins:
  - `unsloth==2025.5.4`
  - `unsloth_zoo==2025.5.6`
  - `transformers==4.52.3`
  - `trl==0.19.1`
- Result: model load and training start succeeded, but batching failed after the first logged steps
- Failure signature or success artifact: `ValueError: expected sequence of length 280800 at dim 2 (got 240001)` from the default Trainer data collator
- Stage reached: training loop entered, then failed on variable-length audio tensors
- Next action: filter raw audio to `<= 240001` samples before preprocessing, force `load_from_cache_file=False`, rerun dataset + preprocessing cells, then restart training

## Cycle 4 — L40S Vanilla Diagnostic (first success) — 2026-04-17

- Model: Sesame CSM (1B)
- Runtime: HF Jobs, L40S (48GB), $1.80/hr
- Entrypoint: `scripts/hf_jobs/T1_csm_vanilla.py`
- Dataset split policy: 5 train samples, 2 steps (diagnostic only, seed 42)
- Package pins: **no Unsloth.** `transformers==4.52.3`, `peft>=0.11,<0.16`, vanilla HuggingFace stack
- Result: **PIPELINE WORKS.** Training loop completes, plain + speaker-conditioned wavs generated and pushed to Hub
- Failure signature or success artifact: train loss 16.04 → 16.05 (no real learning expected — 5 samples × 2 steps). Artifacts at `JosueG/adja-tts-results/T1_diagnostic/`
- Stage reached: end-to-end success, proves the vanilla-HF path is viable
- Next action: scale up to full 1234-sample × 120-step training run on L40S

## Cycle 5 — Full Training (baseline) — 2026-04-17

- Model: Sesame CSM (1B)
- Runtime: HF Jobs, L40S
- Entrypoint: `scripts/hf_jobs/T1_sesame_csm_finetune.py --push-to-hub` (canonical vanilla-HF version)
- Dataset split policy: 80/10/10 split seeded 42. Dropped 43 clips longer than 10s (`MAX_AUDIO_SAMPLES=240001` collator constraint). 1234 train / 140 dev / 160 test.
- Package pins: `transformers==4.52.3`, `peft>=0.11,<0.16`, `accelerate`, `torch==2.6.0+cu124`
- Result: Training completed in **2.8 min**. Train loss 19.95 → 18.61 over 120 steps. Peak VRAM 15.6 GB on L40S (fits on T4).
- Failure signature or success artifact: generated audio was **pure noise / Mimi codec artifacts** — 120 steps on a 1B model is nowhere near converged. Artifacts at `JosueG/adja-tts-results/T1/`
- Stage reached: full pipeline validated; audio inaudible but technically a complete success
- Next action: longer run with evaluation + early stopping on dev loss

## Cycle 6 — Overnight Run with Early Stopping — 2026-04-18

- Model: Sesame CSM (1B)
- Runtime: HF Jobs, L40S
- Entrypoint: `scripts/hf_jobs/T1_sesame_csm_finetune.py --push-to-hub --results-prefix T1_long_20ep_earlystop_2026-04-18`
- Dataset split policy: 1234 train / 140 dev / 160 test (post length filter). Effective batch 16 (4 × grad-accum 4).
- Package pins: same as Cycle 5 + `label_names=["labels"]` fix in TrainingArguments to make `eval_loss` appear for PeftModel
- Result: Ran cleanly for 7 epochs then early-stopped. Best dev loss **6.4876 at epoch 3.85** (step 300). Training wall time 24.7 min. Train loss at stop ~17.4 (still decreasing — classic overfitting signature).
- Failure signature or success artifact:
  - First submission ID `69e309c6cd8c002f31dfe427` crashed at step 50's eval with `KeyError: 'eval_loss'`. Root cause: PeftModel hides base forward signature, Trainer couldn't infer label column, `label_names=[]` meant `has_labels=False`, eval loop skipped `compute_loss()`. Fix: explicit `label_names=["labels"]`.
  - Second submission ID `69e37748ac288e522d8efcdc` completed. Eval history:
    - ep 0.65: 6.776  ↓
    - ep 1.28: 6.635  ↓
    - ep 1.93: 6.563  ↓
    - ep 2.57: 6.545  ↓
    - ep 3.21: 6.501  ↓
    - ep 3.85: **6.488** (best)
    - ep 4.49: 6.508  ↑
    - ep 5.13: 6.556  ↑
    - ep 5.78: 6.583  ↑
    - ep 6.41: 6.716  ↑
    - ep 7.05: 6.785  ↑ (patience-5 early stop fired)
  - Generated wavs at `JosueG/adja-tts-results/T1_long_20ep_earlystop_2026-04-18/generated_audio/` — all 6 clips still perceived as noise / gibberish
- Stage reached: capacity ceiling hit. LoRA r=32 saturates around eval_loss 6.5 on this 1.7h dataset
- Next action: test if the ceiling is capacity or data by comparing LoRA r=128 and full fine-tune with the same early-stopping protocol

## Cycle 7 — T3 Spark TTS first submission + torchaudio binary mismatch — 2026-04-18

- Model: Spark TTS 0.5B (via Unsloth)
- Runtime: HF Jobs, L40S
- Entrypoint: `scripts/hf_jobs/T3_spark_tts_finetune.py --push-to-hub --results-prefix T3_spark_20ep_2026-04-18`
- Dataset split policy: 80/10/10 seed=42 inside the script
- Package pins at failure: Unsloth 2026.4.6 (latest), Transformers 4.56.2, **torch 2.10.0+cu128**
- Result: **FAILED at dataset loading**
- Failure signature: `OSError: /opt/conda/lib/python3.11/site-packages/torchaudio/lib/libtorchaudio.so: undefined symbol: _ZNK5torch8autograd4Node4nameEv`
- Root cause: `pip install unsloth` upgraded torch from the container's 2.6.0 to 2.10.0+cu128, but the pre-installed torchaudio (built for torch 2.6) stayed. Binary ABI mismatch when `torchaudio` loads.
- Fix applied: add `pip uninstall -y torchaudio; pip install -q torchaudio` after unsloth install so pip resolves a torchaudio matching torch 2.10.
- Stage reached: model loaded fine, dataset loading step crashed
- Next action: resubmit T3 with fix (→ Cycle 8)
- Job ID: `69e38873cd8c002f31dfe97e`

## Cycle 8 — T3 Spark TTS resubmission with torchaudio fix — 2026-04-18

- Model: Spark TTS 0.5B (via Unsloth)
- Runtime: HF Jobs, L40S (48GB)
- Entrypoint: `scripts/hf_jobs/T3_spark_tts_finetune.py --push-to-hub --results-prefix T3_spark_20ep_2026-04-18`
- Dataset split policy: 80/10/10 seed=42
- Package pins: same Unsloth stack + `pip install torchaudio` to match torch 2.10
- Result: **submitted, running** (as of 2026-04-18)
- Assumption: fix is sufficient — all subsequent imports (`torchaudio.transforms as T` inside Spark formatting) should succeed
- Job ID: `69e39023ac288e522d8efd34`
- Results destination: `JosueG/adja-tts-results/T3_spark_20ep_2026-04-18/`

## Cycle 9 — T6 MMS-TTS-Ewe first submission via ylacombe/finetune-hf-vits — 2026-04-18

- Model: `facebook/mms-tts-ewe` (VITS, CC-BY-NC 4.0)
- Runtime: HF Jobs, L40S
- Entrypoint: `scripts/hf_jobs/T6_mms_tts_ewe_finetune.py --push-to-hub`
- Training recipe: `https://github.com/ylacombe/finetune-hf-vits` (cloned inside the job)
- Dataset: `JosueG/adja-tts-orpheus` → preprocessed into `JosueG/adja-tts-mms-ready` (new private repo, NFC text + 16 kHz audio + 80/10/10 seed=42 splits)
- Config (inline JSON mirrors ylacombe's `finetune_mms.json` for Gujarati):
  - lr 2e-5, epochs 20, batch 16, fp16 true
  - GAN weights: disc 3, fmaps 1, gen 1, kl 1.5, duration 1, mel 35
- Flow: install → clone ylacombe → `cython monotonic_align` → `convert_original_discriminator_checkpoint.py --language_code ewe` → prep dataset → write JSON → `accelerate launch run_vits_finetuning.py <json>` → inference via `transformers.pipeline("text-to-speech", ...)` → push to Hub
- Key assumption: MMS-TTS-Ewe tokenizer handles raw Adja Unicode after NFC normalize. No G2P needed. Hypothesis: Ewe is Adja's closest Gbe-family relative with a public TTS, so starting from Ewe should give us real intelligibility that CSM/Orpheus/Spark couldn't reach on 1.7h.
- Job ID: `69e390f1cd8c002f31dfe9d4`
- Results destination: `JosueG/adja-tts-results/T6_mms_ewe_20ep_2026-04-18/`
- Stage reached: submitted, pending first 5-min verification

## Cycle 10 — T2 Orpheus 3B canonical HF Jobs scripts landed — 2026-04-18

- Model: `canopylabs/orpheus-3b-0.1-ft` (default) / `canopylabs/3b-fr-ft-research_release` / `canopylabs/3b-de-ft-research_release` (alternates)
- Runtime: HF Jobs, L40S (LoRA default) / A100 80GB (full fine-tune)
- Entrypoint: `scripts/hf_jobs/T2_orpheus_finetune.py`
- Diagnostic entrypoint: `scripts/hf_jobs/T2_orpheus_vanilla.py` (thin wrapper forwarding to canonical with `--dry-run`)
- Read-only reference: `experiments/tts/T2_orpheus_finetune/T2_adja_orpheus_finetune_reference.py` (Kinyarwanda Unsloth Colab notebook, hardcoded token stripped, Colab gotchas annotated)
- Dataset split policy: seed 42, 80/10/10 via `train_test_split` on `JosueG/adja-tts-orpheus` — identical to T1 and CLAUDE.md convention
- Package pins (from upstream Unsloth Orpheus notebook, hardened for HF Jobs):
  - `transformers==4.56.2`
  - `trl==0.22.2 --no-deps`
  - `snac`
  - `peft>=0.11.0,<0.16.0`, `accelerate`, `datasets>=3.4.1,<4.0.0`, `huggingface_hub>=0.34.0`
  - `torchaudio`, `bitsandbytes`
- Key design choices (research-backed):
  - Vanilla HF + PEFT (not Unsloth) — inherits T1's stability; `--use-unsloth` intentionally not exposed until vanilla is green. Reddit thread (Kazakh/Finnish) confirms Unsloth works at scale but T1 burned a week on Unsloth regressions for CSM.
  - LoRA r=64 default: 2x T1's r=32 for the 3.8x larger backbone. Upstream notebook's r=512 is overkill and would dominate the small Adja dataset.
  - Load base in fp32, Trainer handles bf16 — T1's fix #1 for codec-token `index_put_` dtype errors
  - `label_names=["labels"]`, gradient checkpointing OFF — T1's fixes #2 and #3
  - NFC normalise + filter empty/very-short audio — matches T1 and the upstream notebook's validation
  - SNAC 24kHz encode → 7-tokens-per-frame layout (+AUDIO_TOKENS_START=128266 with layer offsets) + dedupe consecutive frames → wrap with `[SOH] text [EOT] [EOH] [SOA] [SOS] codes [EOS] [EOA]` sequence (Orpheus convention)
- Result: diagnostic submitted to HF Jobs — Job `69e3a4feac288e522d8efd55` | L40S | 30 min timeout | `--dry-run --push-to-hub --results-prefix T2_diagnostic_2026-04-18`
- Failure signature: **exit code 128** at container init — `exec: "--flavor": executable file not found in $PATH`. Root cause: `hf jobs run <image> --flavor ...` placed flags AFTER the positional `image`, so the `command` positional (which is `nargs=REMAINDER`) swallowed `--flavor l40sx1 --secrets HF_TOKEN --timeout 30m -d -- bash -c "..."` wholesale and forwarded it to Docker as the entrypoint. `hf jobs run --help` confirms signature is `hf jobs run [options] image command`: options MUST come before the image. T1/T3/T6 READMEs have the same wrong ordering — flagged for cleanup but not fixed yet.
- Fix applied: flags before image. Resubmitted as Job `69e3c218cd8c002f31dfeb64` | L40S | 30m timeout | same `--dry-run` payload.
- **Second failure**: `huggingface_hub.errors.GatedRepoError: 403` on `canopylabs/orpheus-3b-0.1-ft`. All four Canopylabs Orpheus variants (`orpheus-3b-0.1-ft`, `3b-fr-ft-research_release`, `3b-de-ft-research_release`, `orpheus-3b-0.1-pretrained`) return `gated=auto` from the Hub API — they're one-click-agreement gated, not individually approved. Upside: the command-syntax fix worked — container started, pip installs succeeded, script ran all the way to `AutoTokenizer.from_pretrained`. SNAC + text tokenizer code path verified up to that point.
- Blocker for user: visit https://huggingface.co/canopylabs/orpheus-3b-0.1-ft (and the fr/de variants if we want to sweep bases) and click "Agree and access repository". Once done, the sweep can fire without code changes.
- Stage reached: **BLOCKED on user acknowledging Canopylabs model-card gate**. Do not submit sweep until user confirms.
- Next action (once unblocked): rerun diagnostic `69e3c218`-style to confirm dry-run passes end-to-end, then fire the 4-variant sweep (r=32, r=64, r=128 LoRA on L40S + full-FT on A100-large).

## Cycle 11 — T2 Orpheus capacity sweep fired — 2026-04-18

- Gate accepted by user at 2026-04-18. Rerun diagnostic `69e3c450cd8c002f31dfeb72` (L40S, 30m) completed end-to-end: snac/transformers/peft/trl installed, 3.8B base downloaded, 5-sample preprocess + 2 train steps + 1 eval + 2 WAV generations, adapter + tokenizer + WAVs uploaded to `JosueG/adja-tts-results/T2_diagnostic_2026-04-18_v2/`, final log line `T2 canonical training finished.` Dry-run gate PASSED — T2 pipeline confirmed working on HF Jobs.
- Capacity sweep (4 variants in parallel) submitted right after:
  - **r=32 LoRA**, 20 ep, lr 2e-4, L40S, 4h timeout → Job `69e3c68dcd8c002f31dfeb91` → `T2_orpheus_en_lora_r32_20ep_2026-04-18/`. Mirrors T1's LoRA rank for apples-to-apples on the 3.8B backbone vs CSM's 1B.
  - **r=64 LoRA** (default), 20 ep, lr 2e-4, L40S, 4h timeout → Job `69e3c68eac288e522d8efd97` → `T2_orpheus_en_lora_r64_20ep_2026-04-18/`. Canonical mid-rank per the T2 README.
  - **r=128 LoRA**, 20 ep, lr 2e-4, L40S, 4h timeout → Job `69e3c68fac288e522d8efd99` → `T2_orpheus_en_lora_r128_20ep_2026-04-18/`. Upper bound without paying A100 cost.
  - **Full fine-tune**, 10 ep, lr 5e-5, A100-large (80GB), 8h timeout → Job `69e3c692ac288e522d8efd9b` → `T2_orpheus_en_fullft_10ep_2026-04-18/`. Tests whether the ceiling is capacity or data. Full-FT of 3.8B needs A100 for bf16 + AdamW states (L40S OOM).
- All four submitted via `hf jobs run --flavor X --secrets HF_TOKEN --timeout Y -d pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel bash -c "..."` (flags BEFORE image — Cycle 10 fix applied consistently).
- Research hypothesis under test: if all four plateau like T1 did, then data (1.7h) is the bottleneck, not capacity — hard-stop rule triggers and we pivot to MMS-TTS-Ewe (T6) as the primary TTS direction.
- Expected completion: L40S LoRA runs ~45-90 min each; A100 full-FT ~3-5 hours.
- **Added French-base variant** (user request, 2026-04-18): `canopylabs/3b-fr-ft-research_release`, LoRA r=64, 20 ep, L40S, 4h → Job `69e3cba3cd8c002f31dfebdc` → `T2_orpheus_fr_lora_r64_20ep_2026-04-18/`. Direct comparison vs English r=64 (`69e3c68e`): same hyperparameters, same dataset, different pretraining prior.
- **FINAL OUTCOME (2026-04-18)**: All 5 variants completed. User listened to generated audio from every variant. No intelligible Adja words or phonology in any clip — all output perceived as unintelligible noise/non-speech sounds. Final results:
  - r=32: best dev loss 5.4976 (ep 5.00), 26.8 min
  - r=64: best dev loss 5.4823 (ep 3.75), 21.6 min
  - r=128: best dev loss 5.4596 (ep 2.82), 17.8 min
  - full-FT: best dev loss 5.4242 (ep 1.88), 23.3 min — diverged to 9.5+ by ep 3.13
  - French r=64: best dev loss 5.5058 (ep 3.75), 21.4 min — lost to English at matched rank
- Capacity ordering monotonic and clean (r=32 > r=64 > r=128 > full-FT). Better loss did NOT translate to audible Adja. French prior hypothesis falsified. Data bottleneck confirmed. English-prior LLM-TTS direction (CSM, Orpheus) hard-stopped. Next high-leverage experiment: T6 MMS-TTS-Ewe (Gbe-family prior).

## Cycle 12 — Multilingual TTS expansion tracks landed — 2026-04-18

- Models:
  - T7 `openbmb/VoxCPM2` (LoRA / full FT)
  - T8 IMS-Toucan (`Flux9665/ToucanTTS` fine-tuned on `ajg`)
  - T9 `coqui/XTTS-v2`
  - T10 `F5TTS_Base`
  - T11 `E2TTS_Base`
- Runtime target: Hugging Face Jobs, mostly A100 80GB for smoke + longer pilots
- Entrypoints:
  - `scripts/hf_jobs/T7_voxcpm_finetune.py`
  - `scripts/hf_jobs/T8_ims_toucan_finetune.py`
  - `scripts/hf_jobs/T9_xtts_v2_finetune.py`
  - `scripts/hf_jobs/T10_f5_e2_tts_finetune.py`
- Result: repo-native experiment folders, launchers, and runbooks landed for all requested multilingual TTS families.
- Key implementation choices:
  - **VoxCPM**: materialize Adja to upstream JSONL manifests and auto-read sample rate from the downloaded base model config instead of hardcoding.
  - **IMS-Toucan**: use direct Adja ISO code `ajg` because it exists in upstream `iso_lookup.json`; rely on the frontend's Transphone fallback if the language is not directly supported by eSpeak.
  - **XTTS-v2**: materialize Adja into Coqui's LJSpeech-style `metadata.csv + wavs/` format and fine-tune the GPT encoder via the public recipe path.
  - **F5/E2**: materialize Adja into `raw.arrow + duration.json + vocab.txt`, then explicitly expand pretrained text embeddings when Adja introduces unseen symbols (`ɛ`, `ɔ`, `ŋ`, `ɖ`, tone-marked forms).
- Success/failure signals from first smoke wave:
  - **T7 VoxCPM** — `69e3ea21cd8c002f31dfed1e`: SUCCESS. Real train loop ran 2 steps, validation ran, sample audio generation ran, LoRA checkpoint + metrics uploaded.
  - **T8 IMS-Toucan** — `69e3ea21cd8c002f31dfed1f`: moved past CLI/job syntax and dataset materialization, but failed importing upstream Toucan because `Utility/utils.py` requires `matplotlib.pyplot`. Patched launcher to add `matplotlib`; retry3 `69e3ed35cd8c002f31dfed3e` reached `RUNNING` before the final poll, then moved back to `ERROR` by the last inspect. Pending next log pull for the new exact signature.
  - **T9 XTTS-v2** — `69e3e521ac288e522d8efe03`: got through Coqui editable install and Adja dataset materialization, then failed importing XTTS because Coqui's `stream_generator.py` expected a top-level `BeamSearchScorer` export from `transformers`. Patched launcher to re-pin `transformers>=4.41,<5` after the editable install. Retry2 `69e3ea21ac288e522d8efe18` was still `SCHEDULING` at last poll.
  - **T10 F5-TTS** — `69e3e9dbcd8c002f31dfed12`: got through Adja dataset materialization + checkpoint seeding, then hit upstream EMA init failure: `Error: While trying to deepcopy model: cannot pickle '_thread._local' object`. Patched launcher to pass an explicit EMA model instance instead of relying on `deepcopy(model)`. Retry2 `69e3ed35cd8c002f31dfed40` was still `SCHEDULING` at last poll.
  - **T11 E2-TTS** — `69e3e9dccd8c002f31dfed14`: same as T10. Data path validated; trainer failed during EMA deepcopy with the same `_thread._local` pickling error. Patched launcher identically. Retry2 `69e3ed33cd8c002f31dfed3c` was still `SCHEDULING` at last poll.
- Stage reached: implementation complete, HF CLI installed locally, smoke submissions launched, with one confirmed end-to-end training success (T7) and the remaining tracks narrowed to specific runtime blockers rather than vague submission issues.
- Next action:
  1. pull the latest retry3 T8 logs to capture the current post-`matplotlib` blocker
  2. wait for T9/T10/T11 retry jobs to leave scheduling and re-poll logs
  3. if T10/T11 still fail, patch around the next upstream trainer issue rather than touching the Adja dataset path

## Cycle 13 — Multilingual TTS retry loop after live-log patching — 2026-04-19

- Git branch / commits:
  - `cursor/add-multilingual-tts-asr-tracks-652c` / `8786040`
  - follow-up live-log fixes / asset prefetch / Vox audio export: `957f625`
- Submitted retries:
  - `T9-xtts-v2` retry 3: `69e42b13ac288e522d8efe85` — still `SCHEDULING` at the last poll in this cycle
  - `T8-ims-toucan` retry 6: `69e42b13ac288e522d8efe86` — moved into real preprocessing and multilingual fallback execution
  - `T10-f5` retry 6: `69e42b13ac288e522d8efe87` — **completed**, with live checkpoint saves
  - `T11-e2` retry 6: `69e42b13ac288e522d8efe8b` — **completed**, with live checkpoint saves
- Live blocker reductions achieved this cycle:
  - **T8 IMS-Toucan**
    - old blocker removed: Transphone asset race / missing `grapheme.vocab`
    - proof: retry 6 logs show `Transphone assets ready under .../transphone/data/model/042801_base`
    - new blocker isolated: missing PhonePiece inventory file during `string_to_tensor(...)`
      - exact signature: `FileNotFoundError: .../site-packages/phonepiece/data/model/latest/eng/phone.txt`
    - fix landed locally and pushed in `957f625`: prefetch PhonePiece assets before worker processes start, matching the successful Transphone prefetch pattern
  - **T10 F5 / T11 E2**
    - old blocker removed: upstream EMA deepcopy / `_thread._local` pickling crash
    - proof of real startup: retry 6 logs reached
      - `Loading dataset ...`
      - `Saved last checkpoint at update 2`
      - `Saved last checkpoint at update 4` (E2)
    - status interpretation: these two launchers have now crossed the user's "training actually started" requirement for smoke validation
  - **T7 VoxCPM**
    - longer non-A100 run `69e4261ecd8c002f31dfef6b` finished cleanly and uploaded checkpoints/metrics
    - remaining issue is no longer training startup; it is artifact export only
    - root cause isolated from completed logs:
      - post-training export path `generate_audio_samples()` failed with shape mismatches such as
        - `The size of tensor a (8) must match the size of tensor b (32) ...`
        - `The size of tensor a (8) must match the size of tensor b (16) ...`
      - training itself still completed and pushed checkpoints
    - fix landed locally and pushed in `957f625`: switch post-training sample export to the upstream `scripts/test_voxcpm_lora_infer.py` path that already understands the saved LoRA checkpoint format
- Net result of this cycle:
  - **Confirmed startup successes now include T7, T10, and T11**
  - **T8** is down to one more deterministic asset-prefetch issue
  - **T9** is still waiting for a clean post-patch poll because it remained queued

## Cycle 14 — Vox fair-evaluation full fine-tune submission + listenable artifact sync — 2026-04-19

- Requested by: Josue Godeme
- Reason:
  - user listened to the current VoxCPM WAVs and judged them poor / still too anchored to non-Adja priors
  - user explicitly requested a fairer evaluation via a true full-finetune rather than another short LoRA smoke/pilot
- Confirmed listenable artifacts available now:
  - `T7_voxcpm_smoke_audio_retry2_2026-04-19-011425/generated_audio/sample_00_with_lora.wav`
  - `T7_voxcpm_smoke_audio_retry2_2026-04-19-011425/generated_audio/sample_01_with_lora.wav`
  - `T7_voxcpm_smoke_audio_retry2_2026-04-19-011425/generated_audio/sample_02_with_lora.wav`
  - plus `lora_disabled`, `lora_reenabled`, `lora_reloaded`, and `lora_reset` comparison WAVs for each sample
- Important clarification recorded for fairness:
  - those published T7 WAVs are **post-finetuning outputs**, but they come from the **audio-export smoke retry**, not from a long converged full-FT run
  - the longer previous T7 run (`69e4261ecd8c002f31dfef6b`) was a completed **LoRA pilot** (`--num-iters 300`) on `l40sx1`, not a full-finetune job
  - therefore the user's criticism that Vox had not yet received a fair capacity test is valid
- New submission:
  - `T7-voxcpm-fullft` — job `69e43943ac288e522d8efea4`
  - flavor: `a100-large`
  - launcher path: `scripts/hf_jobs/T7_voxcpm_finetune.py`
  - launch mode:
    - `--full-finetune`
    - `--learning-rate 1e-5`
    - `--num-iters 1000`
    - `--valid-interval 50`
    - `--save-interval 100`
    - `--push-to-hub`
    - results prefix: `T7_voxcpm2_fullft_<timestamp>`
- Current state at attempt-log update time:
  - full-finetune job is submitted and awaiting further polling
- Parallel status snapshot relevant to “what else can I listen to?”:
  - T7 listenable WAVs: available now on Hub
  - T10 F5 smoke: startup succeeded, but no generated-audio artifacts are uploaded yet
  - T11 E2 smoke / full: training paths validated, but no generated-audio artifacts are uploaded yet
  - T8 Toucan latest retry completed; latest blocker had moved into `speechbrain` repo-layout drift before the next patch
  - T9 XTTS latest retry still requires another compatibility pass, so no useful audio artifacts there yet

## Cycle 15 — T10/T11 re-smoke with audio generation fix — 2026-04-19

- Root cause fixed: `T10_f5_e2_tts_finetune.py` had no post-training inference phase — training completed and checkpoints saved but no WAVs were generated or pushed to Hub
- Fix: added `generate_audio_samples()` which loads the best checkpoint, runs `f5_tts.infer.utils_infer.infer_process` for 5 test samples using sample 0 as the reference voice, saves WAVs to `output_dir/generated_audio/`, and uploads them to Hub
- Also wired `generated_audio` into `collect_metrics()` (JSON) and `push_results()` (Hub folder upload)
- Submitted re-smokes (both `--dry-run --push-to-hub`):
  - `T10-f5` retry 7: `69e582adac288e522d8f012e`
  - `T11-e2` retry 7: `69e582b1cd8c002f31dffbc9`
- Expected artifact: `JosueG/adja-tts-results/T10_f5_smoke_2026-04-19/generated_audio/sample_01.wav` (and `sample_02` through `sample_05`)

## Cycle 16 — Gbe-family cascade: Stage 1 results + Stage 2 submissions — 2026-04-22

### Stage 1 (WaxalNLP Ewe TTS → checkpoint on Hub)

- Runtime: HF Jobs `l40sx1`, `uv run --python 3.11`, inline PEP-723 deps
- Dataset split policy: `google/WaxalNLP` config `ewe_tts` (1,215 / 152 / 152), NOT the larger `ewe_asr` (chose TTS curation over ASR volume — see `docs/gbe-cascade-tts-settings-2026-04-22.md` §2.2)
- Package pins (Stage 1 CSM): `torch==2.5.1 transformers==4.52.3 peft>=0.11.0,<0.16.0`
- Package pins (Stage 1 Orpheus): `torch==2.6.0 torchaudio==2.6.0 transformers==4.56.2 peft>=0.11.0,<0.16.0`
- Package pins (Stage 1 Spark): `torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 transformers==4.56.2` + Unsloth

Results (native-speaker listening test 2026-04-22):

| Job | Model | Stage 1 data | Verdict |
|-----|-------|--------------|---------|
| `69e830e1ac288e522d8f0782` | CSM 1B LoRA r=32 | WaxalNLP ewe_tts | **Intelligible Ewe** ✓ Gbe-family prior works |
| `69e846e8ac288e522d8f084f` | Orpheus 3B EN base, LoRA r=64 | WaxalNLP ewe_tts | eval_loss ~0.16 @ ep 14 (audio deferred to Stage 2) |
| `69e846eaac288e522d8f0851` | Orpheus 3B FR base, LoRA r=64 | WaxalNLP ewe_tts | eval_loss ~0.16 @ ep 14 (audio deferred to Stage 2) |
| `69e8e3edd2fd2eb837d769b1` | Orpheus 3B ZH base, LoRA r=64 | WaxalNLP ewe_tts | resubmit with grad fix — RUNNING |
| `69e8e3efd2fd2eb837d769b3` | Spark 0.5B LoRA r=128 | WaxalNLP ewe_tts | on `a100-large` after l40sx1 OOM — RUNNING |

### Stage 1 failure signatures cleared this cycle (details in `session-logs/2026-04-22-tts-failure-modes-debug.md`)

1. CVE-2025-32434 gate: torch 2.5.1 + transformers 4.56.2 blocks torch.load. Fix: bump torch to 2.6.0.
2. PEFT grad graph: `gradient_checkpointing=True` + PEFT breaks backward. Fix: `model.enable_input_require_grads()` after `get_peft_model()`.
3. CSM tokfix shape mismatch: `resize_token_embeddings` mutates `config.vocab_size` but CSM's `backbone_loss` uses that field for the Mimi audio codebook (2051), not text vocab. Fix: save/restore `config.vocab_size` around both resize paths.
4. Triton JIT gcc failures: `Python.h missing` + `libcuda.so -lcuda cannot link`. Fix: apt-install `python3-dev libpython3.11-dev` + symlink `/usr/lib64-nvidia/libcuda.so.1` → `libcuda.so` + extend `LIBRARY_PATH`.
5. Spark L40S OOM: fp32 load + LoRA r=128 on 48GB. Workaround this cycle: switch to `a100-large`. Proper fix TBD (bf16 load, lower rank, or grad accumulation).

### Direct-Adja controls (same 2026-04-22 listening test)

- `69e83cf1ac288e522d8f07e9` — Orpheus tokfix, Adja only + expanded Llama tokenizer: completed but 0 audio files generated (investigation pending).
- `69e857f3cd8c002f31e016a6` — CSM tokfix, Adja only + 35 added chars: speech-like noise with rare fragments. **Retires the "tokenizer fragmentation is the primary bottleneck" hypothesis.**
- `69e8318fac288e522d8f0787` / `69e831e4ac288e522d8f078b` — F5-TTS / E2-TTS full fine-tune, Adja only: pure noise. Direct-Adja data-volume ceiling confirmed across architectures.

### Stage 2 submissions (just fired)

All `l40sx1`, 8h, `--push-to-hub`, loading the Stage 1 merged checkpoint from `JosueG/adja-tts-checkpoints/`:

- `69e8e3e8d2fd2eb837d769ad` — CSM Ewe → Adja Stage 2 (highest signal: Stage 1 Ewe was intelligible)
- `69e8e3ead2fd2eb837d769af` — Orpheus EN Ewe → Adja Stage 2
- `69e8e3ec2aa1660eaffa8a83` — Orpheus FR Ewe → Adja Stage 2

Same hyperparameters as Stage 1 with two changes: LR dropped 2e-4 → 5e-5, gradient checkpointing OFF.

### Stage reached

- Stage 1 complete for CSM (intelligible Ewe), Orpheus EN/FR (metrics only); Stage 1 rerun in flight for Orpheus ZH + Spark.
- Stage 2 pending listening verdict for all three completed-Stage-1 models.

### Next action

- Wait for the 3 Stage 2 jobs to complete, listen to Adja outputs, rank vs. T3-spark direct-Adja baseline (the current "intelligible Adja" reference).
- If Stage 2 works on CSM: this is the first confirmed recipe for producing intelligible Adja from an English-centric LLM-TTS, and the paper has its headline. Update `results/tts-comparison.md` and the paper-prep doc accordingly.
- If Stage 2 regresses below Stage 1 Ewe quality: the Adja adaptation is destroying the prior; investigate LR, epochs, LoRA merge behavior.

## Cycle 17 — Stage 2 cascade: catastrophic forgetting confirmed, mixed-data recipe submitted — 2026-04-22

- Model: Sesame CSM 1B + Stage 1 Ewe ckpt (T1), Orpheus 3B EN/FR + Stage 1 Ewe ckpt (T2)
- Runtime: HF Jobs l40sx1, `uv run --python 3.11`, inline PEP-723 deps (torch==2.5.1 for CSM / torch==2.6.0 for Orpheus)
- Entrypoint: `scripts/hf_jobs/T1_csm_ewe_adja_stage2.py`, `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py`
- Dataset split policy: Adja `JosueG/adja-tts-orpheus` 80/10/10 seed=42 (~1,276 train / 160 dev / 158 test)
- Package pins: as per inline PEP-723 of each script (transformers 4.52.3 for CSM, 4.56.2 for Orpheus, peft 0.11.x-0.16.x, snac, bitsandbytes)

### Results (native-speaker listening test, Josue 2026-04-22)

| Run | Best eval_loss | Duration | Listening verdict |
|-----|---------------|----------|-------------------|
| T1 CSM Ewe→Adja S2 (`69e93755`) | 6.5115 | 37.5 min (FULL 20-ep budget, early stop never fired) | Noise, all 5 samples hit the 10s cap |
| T2 Orpheus EN Ewe→Adja S2 (`69e9375`) | 5.6156 | 29.5 min (early stop fired at ep 7.19) | Noise, natural durations 1.88-2.9s |
| T2 Orpheus FR Ewe→Adja S2 (`69e9375a`) | 5.6356 | 34.4 min (early stop fired) | Noise, same pattern as EN |

- Failure signature or success artifact: all three Stage 2 runs produced noise despite Stage 1 Ewe outputs being intelligible (verified cross-architecture: CSM/Mimi and Orpheus/SNAC both produced intelligible Ewe in Stage 1). Initial Stage-2 training-step loss was ~17 on Adja — vs ln(4096)=8.3 random-chance — showing the Stage 1 model was confidently producing Ewe-patterned audio tokens when prompted with Adja text. Catastrophic forgetting during adaptation.
- Stage reached: 3 Stage 2 listening-test noise verdicts logged; mixed-data anti-forgetting experiment submitted.
- Next action: Await mixed-data Stage 2 result (`69e977f82aa1660eaffa8d21`). If it produces intelligible Adja, scale to CSM-mixed + Orpheus-FR-mixed + Orpheus-ZH-mixed + Spark-Ewe-Adja. If still noise, lower LR (1e-5), different mix ratio (3:1 or 1:3), or EWC regularization are the next experiments.

### Side results same cycle

- **Orpheus EN/FR Ewe Stage 1 audio** (inference-only jobs `69e8f639`, `69e8f63a`): 5 samples each, Josue listening verdict "Orpheus EWE is also intelligible good stuff" — Gbe bridge confirmed across architectures.
- **Orpheus ZH Ewe Stage 1 audio** (`69e937b6`): only 1/5 samples (sample 2 CUDA device-side assert poisoned CUDA state for samples 3-5). Sample 1 is 14.6s.
- **Spark Ewe Stage 1** (`69e8e3ef` on a100-large, after 4 rounds of infra fixes): completed. Inference-only on L40S (`69e94068`) generated 5 Ewe samples with natural durations. L40S beat a100-large duplicate `69e93aba` which was still SCHEDULING at L40S completion — clean queue-latency win for L40S.

### Supporting documentation

- `docs/gbe-cascade-tts-settings-2026-04-22.md` §8 — mixed-data recipe exact settings + hypothesis.
- `thesis-writing/gbe-cascade-results-2026-04-22.md` — thesis-ready writeup (kept under `thesis-writing/` alongside the undergrad thesis draft; distinct from `research-paper-exploration/paper-one/` which is for the separate research-paper output).
- `session-logs/2026-04-22-tts-failure-modes-debug.md` — infra failure-mode detail (pip→uv migration, libcuda shim, UTF-8 encoding, CSM pad_to_multiple_of, audio-first push pattern).
