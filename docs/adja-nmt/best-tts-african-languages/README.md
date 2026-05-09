# best-TTS-African-Languages

Ablation suite to find the binding constraint in LLaMA-based TTS for African languages,
using Adja (Gbe family, `aj_Latn`) as the primary case study.

## What This Is

This directory contains experiments, scripts, and documentation for a systematic investigation
into why English-centric LLM-based TTS (Sesame CSM 1B, Orpheus 3B) fails to produce intelligible
speech for Adja when fine-tuned directly on ~1.7 hours of Adja audio — and what architectural,
data, or training interventions can fix it.

The binding-constraint hypothesis space, as of 2026-04-26:

1. **Codec prior** (M0/M1/M2): Mimi or SNAC cannot represent Adja phonemes → codec adaptation needed
2. **LM backbone + data volume** (cascade path): English LM cannot learn Gbe phonology from 1.7h alone
3. **Catastrophic forgetting** (CF1/CF2/CF3): Gbe-family pretraining (Stage 1 Ewe) works, but
   Stage 2 Adja adaptation erases the Gbe prior faster than Adja patterns accumulate
4. **Text tokenizer** (TK1): Llama BPE byte-fragments Adja characters → poor text conditioning
5. **Audio LM pretraining** (AT1): self-supervised pretraining on unlabeled Gbe audio could help
   the LM before supervised TTS fine-tuning

## Current State (2026-04-26)

**What works:**
- Spark TTS 0.5B (Qwen2 BPE + BiCodec/XLSR-53) produces intelligible Adja in ~120 steps directly.
  Multilingual priors cover African phonology out of the box.
- CSM 1B fine-tuned on WaxalNLP Ewe TTS (1,215 clips) produces intelligible Ewe (job `69e830e1`,
  native-speaker listening 2026-04-22). Architecture is not the blocker.
- Orpheus 3B EN/FR fine-tuned on WaxalNLP Ewe TTS → intelligible Ewe (jobs `69e846e8`/`69e846ea`,
  native-speaker listening 2026-04-22).

**What fails:**
- CSM/Orpheus Stage 2 (Ewe checkpoint → Adja): **catastrophic forgetting**. All three Stage 2 runs
  (CSM, Orpheus EN, Orpheus FR) produced noise despite intelligible Stage 1 Ewe checkpoints
  (jobs `69e93755`, `69e93757`, `69e9375a`, native-speaker listening 2026-04-22).
- 1:1 Ewe:Adja mixed Stage 2 (Orpheus EN): still noise, though "better noise" (job `69e977f8`,
  listening 2026-04-23). Loss plateau identical to pure-Adja Stage 2 (5.648 vs 5.616).
- Tokenizer expansion (T1-tokfix, T2-tokfix): noise with rare intelligible fragments, refuting
  the "tokenizer fragmentation is the primary bottleneck" hypothesis.

**Current front:** Catastrophic-forgetting mitigation (CF1/CF2/CF3) is the highest-priority
experimental track. The Gbe-family bridge is confirmed (Stage 1 works). The question is how to
preserve that prior during Stage 2 Adja adaptation.

## Experiment Table

| ID | Hypothesis Being Tested | Status | Instance Type | Est. Cost |
|----|------------------------|--------|---------------|-----------|
| M0 | Mimi reconstruction gate: does Mimi output on raw Adja audio sound clean? If yes, codec is probably fine and CF/AT experiments matter more. | planned | local CPU/GPU | free |
| M1 | Fine-tune Mimi encoder/decoder on Adja audio (acoustic-only, L1+STFT loss). Does codec-adapted Mimi improve CSM Stage 2 intelligibility? | planned | L40S (HF Jobs) | ~$2-4 |
| M2 | Fine-tune Mimi with MMS-300M teacher for codebook-0 semantic distillation on Adja. Does re-distillation add phonological coverage? | planned | A100 80GB (HF Jobs) | ~$5-10 |
| M2b | Ablation of M2: acoustic-only (M1 recipe) vs semantic-distilled (M2 recipe) — same eval budget. | planned | L40S | ~$3-5 |
| CF1 | Anti-forgetting: lower LR (1e-5 or 5e-6) for Stage 2 Adja adaptation from Stage 1 Ewe checkpoint. Does slower drift preserve Gbe prior? | planned | L40S | ~$2 |
| CF2 | Anti-forgetting: EWC (Elastic Weight Consolidation) regularization toward Stage 1 Ewe weights during Stage 2. Penalizes weight drift on parameters important for Ewe. | planned | L40S | ~$3 |
| CF3 | Anti-forgetting: annealing Ewe:Adja mix ratio (start 1:1, end 1:9 over epochs). Does gradual shift from Ewe to Adja prevent forgetting better than static 1:1? | planned | L40S | ~$2 |
| TK1 | Replace Llama BPE with character-level or NLLB subword tokenizer in CSM backbone. Does better text conditioning help when data is sufficient (after Stage 1)? | planned | L40S | ~$3 |
| AT1 | Audio-LM pretraining on unlabeled Gbe audio (WaxalNLP `ewe_asr` 183k clips) before supervised TTS fine-tuning. Does SSL pretraining improve downstream TTS? | planned | A100 80GB | ~$15-30 |

**Experiment dependency order:**
1. Run M0 first (free, local) — if Mimi reconstruction is already clean on Adja, skip M1/M2/M2b
2. Run CF1 and CF2 in parallel (cheapest forgetting mitigations)
3. If CF1/CF2 fail, run CF3 (annealing)
4. TK1 is independent of CF — run in parallel with CF1 if you have budget
5. AT1 is the most expensive; run only after CF results are in

## Quick Start

**Step 1 — Run M0 locally first (free, ~30 min):**

```bash
cd <LOCAL_PATH>
python M0_mimi_reconstruction_gate.py \
    --dataset JosueG/adja-tts-orpheus \
    --num-samples 10 \
    --output-dir /tmp/mimi_gate_outputs
```

Listen to the 10 reconstructed `.wav` files. If the output sounds like a garbled version of the
input (correct phonemes but degraded quality), Mimi can represent Adja and codec adaptation (M1/M2)
is unlikely to be the primary bottleneck. If the output sounds like a completely different language
or pure noise, M1/M2 take priority over CF.

**Step 2 — Dry-run CF2 (EWC) locally:**

```bash
python scripts/sagemaker_jobs/launch.py --experiment CF2 --smoke --dry-run
```

**Step 3 — Submit CF2 to HF Jobs (after dry-run passes):**

```bash
hf jobs uv run --flavor l40sx1 --timeout 4h --python 3.11 --secrets HF_TOKEN \
    -d scripts/hf_jobs/CF2_ewc_stage2.py
```

## Key References

- Sesame CSM (LLM-based TTS, Mimi codec): https://github.com/SesameAILabs/csm
- Orpheus TTS (Llama + SNAC codec): https://github.com/canopyai/Orpheus-TTS
- Spark TTS (Qwen2 + BiCodec/XLSR-53): https://arxiv.org/abs/2503.01710
- Mimi codec (Kyutai, RVQ + WavLM distillation): https://arxiv.org/abs/2410.00037
- WaxalNLP Ewe TTS dataset: https://huggingface.co/datasets/google/WaxalNLP
- Elastic Weight Consolidation (EWC): https://arxiv.org/abs/1612.00796
- Low-resource multilingual TTS: https://aclanthology.org/2022.aacl-main.56.pdf
- Tokenizer impacts on multilingual LMs: https://aclanthology.org/2023.findings-acl.350/
- H-Net (fixed tokenization as limiting assumption): https://arxiv.org/html/2507.07955v2

## Related Docs

- `RESEARCH-QUESTIONS.md` — academic framing of each RQ
- `WHAT-WE-KNOW.md` — synthesis of confirmed/refuted/inconclusive findings
- `NFC-NORMALIZATION.md` — Unicode normalization bug and fix
- `MIMI-FINETUNE-GUIDE.md` — technical guide for Mimi codec fine-tuning (M1/M2)
- `../docs/gbe-cascade-tts-settings-2026-04-22.md` — frozen settings for the 2026-04-22 wave
- `../experiments/registry.md` — per-experiment status and Hub paths
