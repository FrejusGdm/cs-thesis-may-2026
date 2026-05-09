# T2: Orpheus 3B Fine-tune for Adja

Status: **planned / ready** — code written, dry-run gate not yet passed. Next action: submit the diagnostic job below.

## What T2 is

A 3.8B-parameter Llama-backbone TTS fine-tune on the same 1277-utterance Adja dataset that T1 used. Capacity test after T1 (Sesame CSM 1B, LoRA r=32) plateaued at eval loss **6.488** with audible noise.

## Why Orpheus

- 3.8x more params than T1's base → more headroom for learning unseen Adja phonology.
- Canopy Labs publishes pre-adapted non-English variants (French, German), which Reddit reports as easier to further-fine-tune than the English base for new languages. See the Reddit thread we logged in `ideas/tts-models-to-try.md`: Kazakh adaptation succeeded on ~350h with 70/30 target/English mixing and LoRA via Unsloth.
- Same SNAC 24kHz codec tokeniser as the CSM ecosystem — preprocessing logic ports over cleanly.

## Why the base defaults to English, not French

You asked us to train "the English version" with the French/German as alternatives. The default is `canopylabs/orpheus-3b-0.1-ft`. Research-backed reasons:

1. **Adja is a Gbe (Niger-Congo) language.** French is Romance. Geographic proximity in West Africa is sociolinguistic, not phonological — French phoneme priors are unlikely to transfer the way Ewe priors would (that's what T6 tests).
2. English Orpheus is the best-documented variant with known working hyperparameters.
3. The French checkpoint is a *research release* — weights exist but stability isn't guaranteed across transformers versions.

Run the French variant as a second, smaller experiment once the English baseline is down. CLI flag: `--base-model canopylabs/3b-fr-ft-research_release`.

## Canonical paths

- **HF Jobs long run:** [scripts/hf_jobs/T2_orpheus_finetune.py](../../../scripts/hf_jobs/T2_orpheus_finetune.py)
- **HF Jobs diagnostic:** [scripts/hf_jobs/T2_orpheus_vanilla.py](../../../scripts/hf_jobs/T2_orpheus_vanilla.py) — thin wrapper that forwards to canonical with `--dry-run`. No drift possible.
- **Read-only Colab reference:** [T2_adja_orpheus_finetune_reference.py](T2_adja_orpheus_finetune_reference.py) — the Kinyarwanda Unsloth notebook you downloaded, with the hardcoded HF token stripped. **Do not run**; it's kept for provenance.
- **Unsloth vendor notebook:** [references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb](../../../references/unsloth-tts-notebooks/Orpheus_3B_TTS.ipynb)

## Dry-run first (mandatory)

Per CLAUDE.md "Code Must Work Before HPC Submission". Takes ~5 min on L40S:

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T2_orpheus_finetune.py)
hf jobs run pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    --flavor l40sx1 --secrets HF_TOKEN --timeout 30m -d \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T2.py && python /tmp/T2.py --dry-run --push-to-hub --results-prefix T2_diagnostic_$(date +%Y-%m-%d)"
```

Diagnostic must produce:
1. Install + imports succeed (`transformers==4.56.2`, `trl==0.22.2 --no-deps`, `snac`, `peft`)
2. Orpheus model loads in fp32, LoRA patch succeeds
3. 5-sample preprocessing yields `input_ids`, `labels`, `attention_mask`
4. 2 train steps + 1 eval produce `eval_loss` in metrics.json (not just runtime)
5. At least one non-empty generated WAV

If any of those fail, fix before submitting the long run.

## LoRA long run (first real experiment)

```bash
SCRIPT_B64=$(base64 < scripts/hf_jobs/T2_orpheus_finetune.py)
hf jobs run pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    --flavor l40sx1 --secrets HF_TOKEN --timeout 4h -d \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T2.py && python /tmp/T2.py --push-to-hub --results-prefix T2_orpheus_en_lora_r64_20ep_$(date +%Y-%m-%d)"
```

Defaults: English base, LoRA r=64 (2x T1's r=32 for the larger backbone; 512 from the upstream Kinyarwanda notebook is overkill), lr 2e-4 cosine, effective batch 8, 20 epochs max with early stopping patience 5.

## Full fine-tune (follow-up)

```bash
hf jobs run pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
    --flavor a100-large --secrets HF_TOKEN --timeout 8h -d \
    -- bash -c "echo '$SCRIPT_B64' | base64 -d > /tmp/T2.py && python /tmp/T2.py --full-finetune --learning-rate 5e-5 --num-epochs 10 --push-to-hub --results-prefix T2_orpheus_en_fullft_10ep_$(date +%Y-%m-%d)"
```

**Requires A100 80GB.** 3.8B params in bf16 + AdamW optimiser states ≈ 50+ GB — will OOM on L40S. lr drops to 5e-5 because full fine-tune is much less forgiving than LoRA.

## French variant (curiosity)

```bash
... --base-model canopylabs/3b-fr-ft-research_release \
    --results-prefix T2_orpheus_fr_lora_r64_20ep_$(date +%Y-%m-%d)
```

## Known risks

1. **1.7h is ~200× below the Reddit-reported Kazakh success case (350h).** If T2-LoRA also plateaus around 6.5, the bottleneck is data, not capacity. Pivot to T6 (MMS-TTS-Ewe).
2. **Llama BPE tokeniser fragments ɛ ɔ ŋ ɖ into byte sequences.** Same issue T1 had. Orpheus doesn't fix this; we'll see whether extra capacity compensates.
3. **Unsloth churn.** T1 burned a week on Unsloth forward-pass regressions for CSM. For Orpheus, Unsloth is more mature (Canopy Labs uses it), but the canonical path stays vanilla PEFT to inherit T1's stability. A `--use-unsloth` flag is intentionally *not* exposed until vanilla is green.
4. **French transfer is unproven for Gbe.** French proximity to Adja is cultural, not phonological. Don't draw conclusions from the French run without T6 (MMS-Ewe, actual Gbe prior) as a comparison.

## Required config (from T1 lessons)

1. **Load base in fp32**, let Trainer handle bf16 via autocast. Avoids index_put_ dtype errors on codec-token paths.
2. **Gradient checkpointing OFF.** PEFT + checkpointing breaks grad flow.
3. **`label_names=["labels"]`** in TrainingArguments. Without it, PeftModel's hidden forward signature causes `KeyError: 'eval_loss'` at first eval.
4. **NFC-normalise Adja text** before tokenisation (ɛ ɔ ŋ ɖ, é, è).
5. **Filter clips <0.5s** (notebook convention). No max-length filter needed like T1 because Orpheus's max_seq_length=2048 covers typical SNAC encodings of Adja clips.

## Hard-stop rule

If the canonical run fails after one clean attempt *after* a passing diagnostic:

1. Check the attempt log entry for what broke.
2. If install/import: fix pins and retry once.
3. If model/forward error: pivot to T6 (MMS-TTS-Ewe) — same hard-stop rule as T1.

## Tracking

After every cycle, update:
- [experiments/tts/attempt-log.md](../attempt-log.md) — exact hashes, pins, signatures
- [experiments/registry.md](../../registry.md) — row update
- [results/run-ledger.md](../../../results/run-ledger.md) — chronological entry
- [results/tts-comparison.md](../../../results/tts-comparison.md) — leaderboard position

## Results (2026-04-18)

Status: **all 5 variants completed**. User listened to generated audio from every variant on 2026-04-18. Conclusion: no intelligible Adja words or phonology in any clip.

### Results table

| Rank | Variant | Job ID | Best dev loss | Best epoch | Train runtime | HF path |
|------|---------|--------|-------------:|----------:|--------------|---------|
| 1 | Full fine-tune (A100-large, lr 5e-5, 10 ep) | `69e3c692ac288e522d8efd9b` | **5.4242** | 1.88 | 23.3 min | `T2_orpheus_en_fullft_10ep_2026-04-18/` |
| 2 | LoRA r=128 (L40S, lr 2e-4, 20 ep) | `69e3c68fac288e522d8efd99` | 5.4596 | 2.82 | 17.8 min | `T2_orpheus_en_lora_r128_20ep_2026-04-18/` |
| 3 | LoRA r=64 (L40S, lr 2e-4, 20 ep) | `69e3c68eac288e522d8efd97` | 5.4823 | 3.75 | 21.6 min | `T2_orpheus_en_lora_r64_20ep_2026-04-18/` |
| 4 | LoRA r=32 (L40S, lr 2e-4, 20 ep) | `69e3c68dcd8c002f31dfeb91` | 5.4976 | 5.00 | 26.8 min | `T2_orpheus_en_lora_r32_20ep_2026-04-18/` |
| 5 | French base LoRA r=64 (L40S, lr 2e-4, 20 ep) | `69e3cba3cd8c002f31dfebdc` | 5.5058 | 3.75 | 21.4 min | `T2_orpheus_fr_lora_r64_20ep_2026-04-18/` |

Baseline: T1 CSM 1B LoRA r=32 best dev loss **6.488**. T2 beats it by ~1.0 nat across the board.

Diagnostic passed earlier: Job `69e3c450cd8c002f31dfeb72` → `T2_diagnostic_2026-04-18_v2/`.

### Audio-listening conclusion

User listened to all 25 generated clips (5 WAVs × 5 variants) on 2026-04-18. Verdict: **no intelligible Adja words or phonology emerged from any variant.** Outputs range from pure noise to non-speech textured sounds — same perceptual outcome as T1 CSM, despite a ~1.0 nat improvement in dev loss. Better numerical fit of English-prior LLM-TTS to 1.7h of Adja data does not produce intelligible Adja speech.

### Key findings

1. **All 5 variants audibly confirmed unintelligible.** User listened; no Adja phonology emerged. Same outcome as T1 CSM. The loss improvement is real but does not cross the intelligibility threshold at 1.7h of data.

2. **Capacity ordering was clean and monotonic** (r=32 > r=64 > r=128 > full-FT). The 3.8B backbone fits the data better than 1B CSM at every rank. Better fit ≠ audible Adja.

3. **Full fine-tune peaked at epoch 1.88 then catastrophically diverged** (dev loss 5.42 → 9.5+ by epoch 3.13). Classic small-data full-FT failure; early stopping caught it. Signals that capacity is not the ceiling — data volume is.

4. **French base lost to English base** at matched LoRA r=64 (5.5058 vs 5.4823, delta 0.02). The "French TTS prior helps because Adja is in francophone West Africa" hypothesis is falsified. Adja is a Gbe (Niger-Congo) language; phonological proximity to Ewe matters more than sociolinguistic proximity to French. The correct cross-lingual transfer vehicle is a Gbe-family model, not a Romance-family one.

5. **Confirmed bottleneck: data volume, not model capacity, not LLM backbone size.** English-prior LLM-TTS (Llama/Orpheus, Sesame CSM) is the wrong model family to push further at 1.7h of target data. Hard-stop rule from the T1 README is triggered.

### Next direction

The remaining high-leverage TTS experiment is **T6 MMS-TTS-Ewe** (Job `69e390f1cd8c002f31dfe9d4`, already running as of 2026-04-18). MMS-TTS-Ewe starts from a model that already speaks Ewe — Adja's closest relative in the Gbe branch of Niger-Congo — giving the fine-tune a phonological prior that English-centric CSM and Orpheus fundamentally cannot provide. If T6 produces intelligible Adja, that result ends the model-choice search. If T6 also plateaus without intelligibility, the next action is data-side: cross-lingual bootstrapping from a Gbe corpus or G2P-assisted phoneme input.

## References

- Orpheus TTS (Canopy Labs): https://github.com/canopyai/Orpheus-TTS
- French Orpheus checkpoint: https://huggingface.co/canopylabs/3b-fr-ft-research_release
- SNAC 24kHz codec: https://github.com/hubertsiuzdak/snac (arXiv:2410.00037)
- LoRA: https://arxiv.org/abs/2106.09685
- Unsloth TTS fine-tuning guide: https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning
- T1 (what we're building on): [experiments/tts/T1_sesame_csm_finetune/README.md](../T1_sesame_csm_finetune/README.md)
