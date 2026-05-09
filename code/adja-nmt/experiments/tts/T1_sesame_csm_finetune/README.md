# T1: Sesame CSM Fine-tune for Adja

Status: **baseline completed 2026-04-18**. LoRA r=32 plateaus at eval loss 6.49. Generated audio is noise — dataset likely insufficient for English→Adja language adaptation.

## Tl;dr

- Working canonical script: `scripts/hf_jobs/T1_sesame_csm_finetune.py` (vanilla HuggingFace, no Unsloth)
- Working diagnostic: `scripts/hf_jobs/T1_csm_vanilla.py` (short 5-sample, 2-step sanity check)
- Colab path: `T1_adja_csm_finetune.ipynb` + `UPSTREAM_NOTEBOOK_RUNBOOK.md` (uses Unsloth pins — more fragile)
- HPC path: `RUN_ON_HPC.md` (deprioritized — Colab first)
- Results so far: `https://huggingface.co/JosueG/adja-tts-results/tree/main/T1_long_20ep_earlystop_2026-04-18`

## Recommended path (as of 2026-04-18): vanilla HF Jobs

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T1_sesame_csm_finetune.py)
hf jobs run pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    --flavor l40sx1 \
    --secrets HF_TOKEN \
    --timeout 4h \
    -d \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T1.py && python /tmp/T1.py --push-to-hub --results-prefix <descriptive_name>"
```

Why vanilla over the Unsloth Colab path:
- L40S costs ~$1.80/hr, consistently available (A100 queue is backed up)
- No Unsloth regressions to fight (`ValueError: input_ids/inputs_embeds`, broken `_full_forward` patch, torch.compile incompatibilities)
- Trainer + PEFT is battle-tested
- Gives up ~2x training speed, but irrelevant at 1B + LoRA

## Required config for Trainer + PEFT + CSM

These are the non-obvious fixes from the attempt log. Keep all of them:

1. **Load base model in `torch.float32`.** Loading in bf16 causes CSM's internal `index_put_` dtype mismatch. Trainer handles bf16 via autocast.
2. **Gradient checkpointing OFF.** PEFT's frozen base + gradient checkpointing = `UserWarning: None of the inputs have requires_grad=True` → `RuntimeError: element 0 ... does not require grad`.
3. **`label_names=["labels"]` in TrainingArguments.** Without this, `PeftModel` hides the base model's forward signature, `has_labels=False` in `prediction_step`, eval loop skips `compute_loss`, and `metric_for_best_model="eval_loss"` crashes with `KeyError: 'eval_loss'`.
4. **Filter clips longer than 10 s (`MAX_AUDIO_SAMPLES=240001`).** Otherwise the default data collator crashes stacking mismatched-length `input_values` tensors: `expected sequence of length 240001 at dim 2 (got 339840)`.
5. **NFC-normalize Adja text.** Tone marks (é, è) and special chars (ɛ, ɔ, ŋ, ɖ) must be consistent between train and inference.

## What is actually canonical

- **Canonical training script:** `scripts/hf_jobs/T1_sesame_csm_finetune.py` (vanilla HF)
- **Vendor reference notebook:** `references/unsloth-tts-notebooks/Sesame_CSM_1B_TTS.ipynb` (Unsloth; includes multi-speaker + voice cloning patterns worth reading even if we don't use their training loop)
- **Adja Colab notebook:** `experiments/tts/T1_sesame_csm_finetune/T1_adja_csm_finetune.ipynb` + `UPSTREAM_NOTEBOOK_RUNBOOK.md`. Kept because Colab T4 is free and some users prefer interactive inspection.
- **Short diagnostic:** `scripts/hf_jobs/T1_csm_vanilla.py`. Use when you want a 2-minute sanity check that the pipeline works on a new branch / dependency.

## Results summary (2026-04-18)

| Run | Wall | Train loss | Best dev loss | Generation | Storage |
|---|---|---|---|---|---|
| T1-diagnostic (5 samples × 2 steps) | 2 min | 16.05 | n/a | empty-but-valid wav | `T1_diagnostic/` |
| T1-baseline (120 steps) | 2.8 min | 18.61 | n/a | noise | `T1/` |
| T1-long-20ep-es (early stopped at ep 7) | 24.7 min | ~17.4 | **6.488** | noise | `T1_long_20ep_earlystop_2026-04-18/` |

## Why the audio is still noise

- **1.7h of Adja audio** is ~3-5x below typical TTS language-adaptation needs (5-10h).
- **CSM was trained on English-heavy data.** Its Llama backbone has never seen ɛ, ɔ, ŋ, ɖ, or Adja tone marks.
- **LoRA r=32** can only make small rank-limited updates. Learning a new phoneme→audio mapping likely needs more capacity.

Eval loss plateau at 6.5 then overfit = the model has learned what it can from current (data × capacity). Next levers:

1. Full fine-tune of CSM (current: in progress) — tests capacity ceiling
2. MMS-TTS-Ewe as starting point (Ewe is closest Gbe relative in MMS catalog) — tests if Gbe-family priors help
3. IMS-Toucan — purpose-built low-resource tonal TTS framework
4. XTTS-v2 — multilingual TTS with small-data fine-tune track record

See `ideas/tts-models-to-try.md` for full model matrix and data-augmentation ideas.

## Dry-run success gate

Before any long run, confirm the canonical script / notebook passes:

1. Install + import succeeds (no missing `unsloth_zoo` or stale `trl`)
2. Model loads with `CsmForConditionalGeneration.from_pretrained(..., torch_dtype=torch.float32)`
3. LoRA patch succeeds with gradient checkpointing OFF
4. 5-sample preprocessing yields all of: `input_ids`, `attention_mask`, `labels`, `input_values`, `input_values_cutoffs`
5. 2-step train + first eval produce `eval_loss` in metrics (not just runtime/throughput)
6. One plain-text waveform generated (non-empty)
7. One speaker-conditioned waveform generated (non-empty)

## Hard-stop rule

If a canonical CSM run fails after one clean attempt:

1. Run `scripts/hf_jobs/T1_csm_vanilla.py` for a 2-minute diagnostic
2. If the diagnostic also fails, pivot:
   - to `scripts/hf_jobs/T3_spark_tts_finetune.py`, OR
   - to MMS-TTS-Ewe (see `ideas/tts-models-to-try.md`)

## Tracking

After every TTS cycle, record in all four places:

- `experiments/tts/attempt-log.md` — operational facts (exact hashes, pins, signatures)
- `experiments/registry.md` — registry row if it's a new experiment ID
- `results/run-ledger.md` — chronological entry
- `results/tts-comparison.md` — rank in the leaderboard
