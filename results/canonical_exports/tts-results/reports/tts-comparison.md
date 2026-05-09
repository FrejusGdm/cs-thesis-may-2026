# Adja TTS Comparison

Tracks Adja speech synthesis runs. Keep this separate from the ASR leaderboard in `comparison.md`.

Last updated: **2026-04-28** — added quantitative reverse-WER auto-eval (see [reverse-wer-summary.md](reverse-wer-summary.md)).

## Reverse-WER auto-eval (2026-04-29 — expanded run, 22 runs scored)

Until now TTS ranking was listening-only. This block adds a quantitative
metric: transcribe each synthesized WAV with our Adja ASRs (E4v4 = Whisper-Ewe→Adja,
C4v2 = Wav2Vec2-XLS-R + CTC, both deployed as dedicated HF Inference Endpoints,
intel-spr x2 CPU) and compare hypotheses to the original prompt. Lower WER/CER
= more intelligible synthesis. **Initial scope (2026-04-28) was 4 runs;
expanded scope (2026-04-29) is 22 runs** — we discovered that
`JosueG/adja-tts-results` exists as both a dataset repo (4 runs) and a model
repo (47 runs), and the older T1/T2/T3 runs all live on the model side.

Sorted by **C4v2 corpus CER (norm)**. Full table + per-utterance dump in
[results/reverse-wer-summary.md](reverse-wer-summary.md).

### Headline (top 6 by C4v2 CER norm)

| Run | C4v2 CER norm | C4v2 WER norm | E4v4 CER norm | Listening verdict |
|---|---|---|---|---|
| **T3 Spark 20-ep early-stop** | **33.73%** | 72.73% | 43.37% | (Spark family — intelligible Adja per listening) |
| **T3 (Spark family canonical)** | **36.14%** | 72.73% | 28.92% | intelligible Adja |
| **T3 Spark 120-step** | 46.99% | 95.45% | 53.01% | **intelligible Adja** (Josue 2026-04-18: "I can understand words") |
| T1 CSM Ewe Stage 1 | 56.03% | 94.79% | 50.86% | intelligible **Ewe** (Gbe-cascade Stage 1 — CSM speaks Gbe phonology) |
| CF2 curriculum stage 2 | 69.64% | 103.85% | 81.25% | 1/5 intelligible Adja (sample 03 = "Tɛnigbe ciyi vayi de ŋweba") |
| CF1 EWC stage 2 | 75.00% | 107.69% | 225.00% | noise |

The Spark family (T3, T3_spark_120steps, T3_spark_20ep_earlystop) all sit in
the 33–47% C4v2 CER band — within ~10 points of C4v2's published test CER on
**real** Adja speech (25.05%). That is the convergence between listening test
and ML metric we wanted: Spark is the only architecture for which the ASR
recognizes meaningful Adja content.

The complete table (22 runs × 2 ASRs) is in
[reverse-wer-summary.md](reverse-wer-summary.md). All Stage-2 cross-language
attempts (T1_csm_ewe_adja_stage2, T2_orpheus_*_ewe_adja_stage2) sit at ≥100%
CER. All T2 Orpheus direct-Adja LoRA variants (r=32/64/128/full-FT, EN+FR)
sit at 200–300% CER — the negative-result floor for English-prior LLM-TTS on
1.6 hours of Adja, exactly as the listening test predicted.

### Old summary block (2026-04-28, 4 runs only)

The original limited-scope summary is preserved here for the audit trail.
Note that the 2026-04-28 reading "T3A Spark underrates it" turned out to be
*specific to T3A* (which is one Spark variant). When the full Spark family is
scored (2026-04-29), it dominates the leaderboard. Listening-test results and
reverse-WER agree.

| Run | C4v2 CER norm | C4v2 WER norm | E4v4 CER norm |
|---|---|---|---|
| CF2 curriculum stage 2 | 69.64% | 103.85% | 81.25% |
| CF1 EWC stage 2 | 75.00% | 107.69% | 225.00% |
| T3A Spark direct Adja | 96.43% | 103.85% | 494.64% |
| CF3 frozen-backbone stage 2 | 104.46% | 123.08% | 155.36% |

**Why E4v4 explodes on T3A:** Whisper hallucination loops on out-of-distribution
TTS audio (e.g. one Spark sample produced infinite "ɖɔ́ ɖɔ́ ɖɔ́…" repetitions
yielding CER >2000%). C4v2's CTC head has no auto-regressive decoder, so it
degrades gracefully. This is the primary motivation for keeping two
architecturally-different ASRs in the eval.

**E4_v5 recovery attempt:** completed 2026-04-28 (HF Job
[`69f0f1c8d2c8bd8662bd235c`](https://huggingface.co/jobs/JosueG/69f0f1c8d2c8bd8662bd235c)),
**dev CER 37.35% at epoch 21** — within 0.2pp of E4v4's 37.18% and far from
the original E4's 24.90%. We **do not swap E4_v5 in** because it would not
move the eval. The original E4 likely trained with the (later-fixed)
EOS-masking bug whose noise acted as regularization; without it, the model
overfits to a worse plateau. The 24.90% Adja Whisper number is, for now,
unreproducible. See [experiments/registry.md](../experiments/registry.md) E4_v5 row.

**Caveats** (in [reverse-wer-summary.md §Ceiling caveat](reverse-wer-summary.md)):

1. Both ASRs have non-trivial test-time error (~25-37% published CER).
   Reverse-WER inherits that floor, which is why noise-tier runs cluster
   around 100-110% rather than the theoretical floor.
2. Eval set is 4 runs × 5 sentences = 20 samples. Sample variance dominates
   small differences. Sort order on close runs (e.g. CF1 vs CF2) is
   meaningful; absolute CER values are not.
3. Listening-test verdicts remain the primary qualitative axis — reverse-WER
   is a quantitative cross-check, not a replacement.

## Original 2026-04-26 ranked summary (listening-test based)

**BREAKTHROUGH: CF2 curriculum annealing produced the first intelligible Adja Stage 2 output. 1/5 sentences confirmed by native speaker.**

## Summary (2026-04-26 update — BREAKTHROUGH)

**Catastrophic forgetting is NOT insurmountable.** CF2 (curriculum annealing, 5:1→1:5 Ewe:Adja over 20 epochs) produced **1 out of 5 intelligible Adja sentences** from a CSM model previously fine-tuned on Ewe. This is the first time any Stage 2 experiment (Ewe → Adja) has produced recognizable Adja speech. All prior Stage 2 attempts (T1-csm-ewe-adja-stage2, T2-orpheus-{en,fr}-ewe-adja-stage2, T2-orpheus-en-ewe-adja-stage2-mixed) were 0/5 intelligible.

**CF1 (EWC)** achieved the best eval loss of all Stage 2 experiments (6.4941), beating both the naive Stage 2 baseline (6.5115) and CF2 (6.6203). All 5 samples have natural durations — listening verdict pending.

**Data setup (both CF1 and CF2):** Ewe = WaxalNLP ewe_tts (1,215 clips, public). Adja = JosueG/adja-tts-orpheus, 80/10/10 split seed=42 (~987 train / ~123 dev / ~124 test). **NFC normalization applied to all text** via `unicodedata.normalize("NFC", text.strip())`. Infrastructure: SageMaker PyTorch 2.5.1/py311/cu124/A10G 24GB.

**Ranked result (audio quality, 2026-04-26):**

1. **T3 Spark 0.5B direct Adja** — intelligible Adja (unchanged)
2. **T1 CSM-Ewe stage1 / T2 Orpheus-Ewe stage1** — intelligible Ewe  
3. **🆕 CF2 curriculum annealing Stage 2** — 1/5 intelligible Adja (native speaker confirmed)
4. **CF1 EWC Stage 2** — best eval loss (6.4941), listening verdict pending
5. T1-csm-tokfix — noise with fragments
6. All prior Stage 2 attempts — noise, 0/5 intelligible

## Summary (2026-04-22 update)

**The Gbe-family bridge hypothesis is confirmed on CSM.** CSM 1B fine-tuned on 1,215 WaxalNLP Ewe TTS clips produces **intelligible Ewe speech** (native speaker confirmation). CSM fine-tuned on ~1.6k Adja clips directly — even with tokenizer expansion (35 Adja tokens added via `add_tokens`) — produces speech-like noise with rare intelligible fragments. **Data volume + Gbe phonology matters more than tokenizer fragmentation.**

**What that retires:** the "Llama BPE fragmenting Adja diacritics is the primary bottleneck" hypothesis. T1-tokfix (expanded tokenizer, direct Adja) is barely better than T1-vanilla (no expansion, direct Adja). Both are noise. CSM *can* produce intelligible Gbe speech — it just needs substantially more than 1.6h of target-language data or a cousin-language bridge.

**F5-TTS and E2-TTS (full fine-tune on 1.6k Adja, ~50 min each)** also produce pure noise — corroborating the data-volume ceiling across architectures. Their 2026-04-21 "samples" have natural durations (2.6–4.3s), but contain no intelligible speech.

**Ranked result (audio quality, listening test by Josue 2026-04-22)**:

1. **T3 Spark 0.5B direct Adja** — intelligible Adja (unchanged from 2026-04-18 verdict)
2. **T1 CSM-Ewe stage1** — intelligible **Ewe** (not Adja; shows CSM architecture works when given Gbe data)
3. **T1 CSM tokfix (Adja + expanded tokenizer)** — speech-like noise, occasional intelligible fragments ("like iron being rammed through the ground" — Josue)
4. **T10 F5 / T11 E2 full fine-tune** — pure noise

**What unblocks now**: Stage 2 (Ewe-pretrained → Adja) for the cascade candidates. CSM-Ewe checkpoint is pushed at `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1` and `T1_csm_ewe_adja_stage2.py` is ready. Orpheus EN-Ewe and FR-Ewe also completed (2026-04-22) — checkpoints at `T2_orpheus_{en,fr}_ewe_stage1`.

## Older Summary (2026-04-18)

**T3 Spark TTS 0.5B is the only model to produce intelligible Adja.** After just 120 training steps (< 1 epoch, 1.4 min), native speaker (Josue) confirmed: *"so, so good. I can understand words. It actually produces coherent words."* Every English-centric LLM-TTS (CSM, Orpheus at all ranks and full-FT) produced unintelligible noise regardless of training duration.

**The capacity hypothesis is dead.** T1 CSM full fine-tune (1.6B trainable params, 30.1 min, job `69e3856e`) reached dev loss 6.496 — essentially identical to LoRA r=32 (6.488). Unlocking 100% of parameters did nothing. The ceiling is representational, not parameter count.

**Why Spark works:** BiCodec uses wav2vec2-XLSR-53 semantic tokens (multilingual by construction — 53 languages including African ones), vs. Mimi which is English-heavy. Qwen2 tokenizer handles Adja diacritics natively; Llama 3.2 BPE fragments ɛ/ɔ/ŋ/ɖ into bytes. Full analysis: `research-paper-exploration/paper-one/why-spark-worked.md`.

**English-prior LLM-TTS (CSM, Orpheus) direction is hard-stopped.** Dev loss metrics are not cross-comparable between model families (different codec token CE). Audio quality is the only valid ranking criterion.

**Longer Spark training (20-epoch eval+early-stop run, job `69e39bc2`) was submitted 2026-04-18 but outcome not yet logged.** Spark's 120-step result was already intelligible, so the question is whether more epochs refine quality further or plateau like CSM did.

**T2 Orpheus 3B beats T1 CSM 1B by ~1.0 nat numerically** (5.4242 vs 6.488), and the capacity sweep (r=32→r=128→full-FT) is clean and monotonic — but none of it translated to audible speech. French base did not beat English at matched rank.

**Qwen3-TTS 1.7B** is operational end-to-end on HF Jobs; listening verdict not yet recorded.

## Results Table

Note: dev loss metrics are NOT cross-comparable between model families (Mimi token CE vs BiCodec token CE vs VITS GAN). Rank by audio quality first; dev loss rank is within-family only.

| Rank (audio) | Exp ID | Model | Runtime | Status | Train Loss (end) | Best Dev Loss | Epoch of Best | Generation Quality | Notes |
|------|--------|-------|---------|--------|------------------|---------------|---------------|--------------------|-------|
| **🆕 2** | **CF2-curriculum** | **CSM 1B, LoRA r=32, Ewe→Adja, curriculum 5:1→1:5** | **SageMaker A10G** | ✅ **completed (2026-04-26)** | — | **6.6203** | **19 (still improving)** | **1/5 INTELLIGIBLE Adja — native speaker (Josue 2026-04-26): "Ŋ nyanyɔ mɔ wo anu ahán !" confirmed intelligible. 4/5 noise.** | Job `adja-cf2-full-20260426-161752`. 49.9 min training. LoRA r=32, lr=5e-5. Ewe:Adja ratio annealed 5:1→1:5 over 20 epochs. Ewe=WaxalNLP ewe_tts (1215 clips), Adja=adja-tts-orpheus 80/10/10 split (~987 train). NFC normalized. **First intelligible Adja Stage 2 output in any experiment.** Natural durations (2.3–7.8s, no 10s cap). Audio: [CF2_curriculum_stage2/generated_audio](https://huggingface.co/datasets/JosueG/adja-tts-results/tree/main/CF2_curriculum_stage2/generated_audio). |
| **🆕 4** | **CF1-ewc** | **CSM 1B, LoRA r=32, Ewe→Adja, EWC λ=1000** | **SageMaker A10G** | ✅ **completed (2026-04-26)** | — | **6.4941** | **best of all Stage 2** | **All noise — Josue 2026-04-26: "no intelligible sentence, crispy noise, you could maybe hear some words in the background but not intelligible"** | Job `adja-cf1-full-20260426-183144`. 71.5 min. EWC Fisher diagonal on 500 Ewe samples. Pure Adja in training loop (no Ewe data — EWC penalty enforces Ewe priors via regularization). best_eval_loss=6.4941 beats naive Stage 2 baseline (6.5115) and CF2 (6.6203). Natural durations on all 5 samples (1.76–5.6s). Audio: [CF1_ewc_stage2/generated_audio](https://huggingface.co/datasets/JosueG/adja-tts-results/tree/main/CF1_ewc_stage2/generated_audio). |
| **1** | **T3-spark-120step** | **Spark TTS 0.5B, FULL fine-tune (Qwen2 + BiCodec/XLSR)** | **L40S HF Jobs** | ✅ **completed** | **6.43** | **—** | **< 1 ep (120 steps)** | **INTELLIGIBLE Adja — native speaker confirmed coherent words (2026-04-18)** | Job `69e3987d`. Only model to produce intelligible Adja. 1.4 min training. Train loss 7.2 → 6.43. BiCodec XLSR-53 multilingual semantic tokens + Qwen2 tokenizer = multilingual priors. Analysis: `research-paper-exploration/paper-one/why-spark-worked.md`. |
| **2 (Ewe)** | **T1-csm-ewe-stage1** | **Sesame CSM 1B, LoRA r=32, WaxalNLP Ewe TTS (1215 clips)** | **L40S HF Jobs** | ✅ **completed (2026-04-21)** | **12.08** | **4.326** | **3.86** | **INTELLIGIBLE Ewe — Josue 2026-04-22: "CSM produces intelligible Ewe (which is good!)"** | Job `69e830e1ac288e522d8f0782`. 17.4 min, 25.52 GB VRAM. Best checkpoint at epoch 3.86; diverges after (final eval=4.65). `load_best_model_at_end=True` means pushed checkpoint IS best. Confirms Gbe-family transfer works with CSM architecture. Audio: [T1_csm_ewe_stage1/generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T1_csm_ewe_stage1/generated_audio). Checkpoint: `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1`. |
| 3 (noise) | T1-csm-tokfix | Sesame CSM 1B, LoRA r=32, Adja only + 35 added text tokens (ɔɛŋɖ + diacritics) | L40S HF Jobs | ✅ completed (2026-04-22) | — | — | — | Speech-like noise, occasional human-voice fragments — Josue 2026-04-22: "just noise but a tiny better noise... you could hear someone speaking sometime in the background but unintelligible (lots of crisp noise like iron being rammed through the ground)" | Job `69e857f3cd8c002f31e016a6`. **Refutes the "Llama BPE fragmentation is the primary bottleneck" hypothesis.** Tokenizer expansion alone — without the Gbe-family data bridge — leaves the model near the same quality as T1-vanilla. Required `config.vocab_size` restore patch because CSM's backbone_loss uses that field for the Mimi codebook space (2051), not text vocab. Results: [T1_csm_tokfix](https://huggingface.co/JosueG/adja-tts-results/tree/main/T1_csm_tokfix). |
| 4 (noise) | T10-f5-full-l40s | F5-TTS full fine-tune on Adja only (1234 clips, char-level vocab) | L40S HF Jobs | ✅ completed (2026-04-21) | — | — | — | Pure noise — Josue 2026-04-22: "E2 and F5 (which I assume were only trained on our 2 hours of adja) only produce noise for now" | Job `69e8318fac288e522d8f0787`. 48.9 min. 5 samples with natural durations (2.6–4.3s) but no intelligible content. Results: [T10_f5_full_2026-04-21_l40s](https://huggingface.co/JosueG/adja-tts-results/tree/main/T10_f5_full_2026-04-21_l40s). |
| 4 (noise) | T11-e2-full-l40s | E2-TTS full fine-tune on Adja only (1234 clips, char-level vocab) | L40S HF Jobs | ✅ completed (2026-04-21) | — | — | — | Pure noise (same listening verdict as T10) | Job `69e831e4ac288e522d8f078b`. 50.9 min. Same audio character as F5. Results: [T11_e2_full_2026-04-21_l40s](https://huggingface.co/JosueG/adja-tts-results/tree/main/T11_e2_full_2026-04-21_l40s). |
| — | T2-orpheus-en-ewe-stage1 | Orpheus 3B EN base, LoRA r=64, WaxalNLP Ewe TTS (1215 clips) | L40S HF Jobs | ✅ completed (2026-04-22) | 0.45 | 0.16 | 14 | Not yet evaluated (audio generation deferred to Stage 2) | Job `69e846e8ac288e522d8f084f`. Required `model.enable_input_require_grads()` fix after `get_peft_model()` to prevent "loss does not require grad" crash under gradient_checkpointing. Checkpoint: `JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1`. |
| — | T2-orpheus-fr-ewe-stage1 | Orpheus 3B FR base, LoRA r=64, WaxalNLP Ewe TTS (1215 clips) | L40S HF Jobs | ✅ completed (2026-04-22) | 0.29 | 0.16 | 14 | Not yet evaluated (audio generation deferred to Stage 2) | Job `69e846eaac288e522d8f0851`. Same grad fix as EN. Checkpoint: `JosueG/adja-tts-checkpoints/T2_orpheus_fr_ewe_stage1`. |
| — | T2-orpheus-tokfix | Orpheus 3B, Adja only + expanded tokenizer | L40S HF Jobs | ✅ completed (2026-04-21) | — | — | — | Completed but produced no audio samples (0 wavs in output; likely generation path was skipped) | Job `69e83cf1ac288e522d8f07e9`. Results folder has metrics.json but generation didn't run. Need to investigate before drawing conclusions. Results: [T2_orpheus_tokfix](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_tokfix). |
| **2 (Ewe, confirmed)** | **T2-orpheus-en-ewe-stage1** | **Orpheus 3B EN, LoRA r=64, WaxalNLP Ewe TTS** | **L40S HF Jobs** | ✅ **completed (2026-04-22)** | 0.45 | **0.16** | 14 | **INTELLIGIBLE Ewe** — native-speaker listening test 2026-04-22: *"Orpheus EWE is also intelligible good stuff"* | Stage 1 training job `69e846e8ac288e522d8f084f`. Inference-only job `69e8f639d2fd2eb837d76a72` generated 5 Ewe samples for the listening test. Checkpoint: `JosueG/adja-tts-checkpoints/T2_orpheus_en_ewe_stage1`. Audio: [T2_orpheus_en_ewe_stage1_inference/generated](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_ewe_stage1_inference/generated). **Confirms the Gbe-family bridge also works on Orpheus (SNAC codec, not just Mimi).** |
| **2 (Ewe)** | **T2-orpheus-fr-ewe-stage1** | **Orpheus 3B FR, LoRA r=64, WaxalNLP Ewe TTS** | **L40S HF Jobs** | ✅ **completed (2026-04-22)** | 0.29 | 0.16 | 14 | **INTELLIGIBLE Ewe (same verdict as EN)** | Stage 1 training job `69e846eaac288e522d8f0851`. Inference-only job `69e8f63ad2fd2eb837d76a74`. Audio: [T2_orpheus_fr_ewe_stage1_inference/generated](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_fr_ewe_stage1_inference/generated). Paired with EN. |
| 3 (noise) | T1-csm-ewe-adja-stage2 | Sesame CSM 1B, Stage 1 Ewe ckpt → Adja LoRA r=32 | L40S HF Jobs | ✅ completed (2026-04-22) | 17.54 | 6.5115 | ~19 (full 20-ep budget, early stop NEVER fired) | **Noise, all 5 samples hit the 10s cap** | Job `69e93755d2fd2eb837d76d4e`. 37.5 min — **ran full 20-epoch budget with no early stopping**. Catastrophic forgetting of the Gbe prior learned in Stage 1. Critical evidence that patience is NOT the bottleneck (CSM finished full budget and still noise). Audio: [T1_csm_ewe_adja_stage2/generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T1_csm_ewe_adja_stage2/generated_audio). |
| 3 (noise) | T2-orpheus-en-ewe-adja-stage2 | Orpheus 3B EN, Stage 1 Ewe ckpt → Adja LoRA r=64 | L40S HF Jobs | ✅ completed (2026-04-22) | 5.89 | 5.6156 | 5.63 (early-stopped ep 7.19) | **Noise, natural durations 1.88-2.9s** — Josue 2026-04-22: *"noise - i am super surprised because the EWE had good EWE speaking stuff! now that we finetuned it on adja - it is just noise"* | Job `69e937572aa1660eaffa8c5a`. Initial training-step loss 17 on Adja (vs ln(4096)=8.3 random-chance for SNAC) — Stage 1 model was confidently producing Ewe tokens when prompted with Adja. Audio: [T2_orpheus_en_ewe_adja_stage2/generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_ewe_adja_stage2/generated_audio). |
| 3 (noise) | T2-orpheus-fr-ewe-adja-stage2 | Orpheus 3B FR, Stage 1 Ewe ckpt → Adja LoRA r=64 | L40S HF Jobs | ✅ completed (2026-04-22) | 5.81 | 5.6356 | ~7 (early stop patience=5) | Noise, same pattern as EN | Job `69e9375a2aa1660eaffa8c5c`. 34.4 min. FR and EN land within 0.02 nats of each other at the plateau — FR sociolinguistic prior does NOT meaningfully differ from EN under this recipe. Audio: [T2_orpheus_fr_ewe_adja_stage2/generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_fr_ewe_adja_stage2/generated_audio). |
| 2 (Ewe, Spark) | T3-spark-ewe-stage1 | Spark TTS 0.5B, LoRA r=128, WaxalNLP Ewe TTS | a100-large HF Jobs | ✅ completed (2026-04-22) | — | — | — | 5 Ewe samples with natural 6.8-25.3s durations; listening verdict pending | Stage 1 job `69e8e3efd2fd2eb837d769b3` (after 3 rounds of infra debugging: torch 2.5→2.6 for CVE, python3-dev for Python.h, libcuda.so symlink, bf16+full_finetuning=False fix for L40S OOM, finally completed on a100-large). Inference-only job `69e94068d2fd2eb837d76d98` (L40S) generated 5 samples. Audio: [T3_spark_ewe_stage1_inference_l40s/generated](https://huggingface.co/JosueG/adja-tts-results/tree/main/T3_spark_ewe_stage1_inference_l40s/generated). |
| 3 (noise) | T2-orpheus-en-ewe-adja-stage2-mixed | Orpheus 3B EN, Stage 1 Ewe ckpt → Adja+Ewe 1:1 mix (anti-forgetting) | L40S HF Jobs | ✅ completed (2026-04-23) | 2.47 | 5.648 | 6.79 (early stop) | **Better noise than baseline Stage 2, but still noise** — Josue 2026-04-23: *"better noise than before but still noise — this is crazy — it's insane that we can't get great adja yet"* | Job `69e977f82aa1660eaffa8d21`. 53.7 min. 3/5 samples. Best_eval_loss 5.648 vs baseline 5.616 — virtually identical plateau. 1:1 Ewe reinforcement marginally improves audio texture but does not recover intelligibility. Catastrophic forgetting is not addressed by naive data mixing. Stronger intervention needed: higher Ewe ratio, elastic weight consolidation, lower Stage-2 LR, or more Adja data. Audio: [generated_audio](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_ewe_adja_stage2_mixed/generated_audio). |
| — | T3-spark-20ep | Spark TTS 0.5B, FULL fine-tune, 20 ep eval+early-stop | L40S HF Jobs | ⚪ outcome not logged | — | — | — | Not recorded | Job `69e39bc2`. Submitted 2026-04-18. Whether longer training improved on the 120-step intelligible baseline is still an open question. |
| 2–6 (noise) | T2-orpheus-fullft | Orpheus 3B, FULL fine-tune | A100-large HF Jobs | ✅ completed | 4.49 | 5.4242 | 1.88 | Unintelligible noise (user-confirmed 2026-04-18) | Job `69e3c692ac288e522d8efd9b`. Best dev loss numerically. Diverged 9.5+ by ep 3.13. 23.3 min wall. `T2_orpheus_en_fullft_10ep_2026-04-18/` |
| — | T2-orpheus-r128 | Orpheus 3B, vanilla PEFT LoRA r=128 | L40S HF Jobs | ✅ completed | 5.44 | 5.4596 | 2.82 | Unintelligible noise (user-confirmed 2026-04-18) | Job `69e3c68fac288e522d8efd99`. 17.8 min wall. `T2_orpheus_en_lora_r128_20ep_2026-04-18/` |
| — | T2-orpheus-r64 | Orpheus 3B, vanilla PEFT LoRA r=64 | L40S HF Jobs | ✅ completed | 5.46 | 5.4823 | 3.75 | Unintelligible noise (user-confirmed 2026-04-18) | Job `69e3c68eac288e522d8efd97`. 21.6 min wall. `T2_orpheus_en_lora_r64_20ep_2026-04-18/` |
| — | T2-orpheus-r32 | Orpheus 3B, vanilla PEFT LoRA r=32 | L40S HF Jobs | ✅ completed | 5.47 | 5.4976 | 5.00 | Unintelligible noise (user-confirmed 2026-04-18) | Job `69e3c68dcd8c002f31dfeb91`. 26.8 min wall. `T2_orpheus_en_lora_r32_20ep_2026-04-18/` |
| — | T2-orpheus-fr-r64 | Orpheus 3B French base, LoRA r=64 | L40S HF Jobs | ✅ completed | 5.48 | 5.5058 | 3.75 | Unintelligible noise (user-confirmed 2026-04-18) | Job `69e3cba3cd8c002f31dfebdc`. French prior hypothesis failed. 21.4 min wall. `T2_orpheus_fr_lora_r64_20ep_2026-04-18/` |
| — | T1-long-20ep-es | Sesame CSM 1B, LoRA r=32 | L40S HF Jobs | ✅ completed | 17.4 | 6.488 | 3.85 | Noise / Mimi artifacts | Early stopped ep 7. 24.7 min wall. `T1_long_20ep_earlystop_2026-04-18/` |
| — | **T1-full-ft** | **Sesame CSM 1B, FULL fine-tune** | **L40S HF Jobs** | ✅ **completed** | ~22–28 (noisy) | **6.496** | **1.3** | **Noise — capacity hypothesis REFUTED** | **Job `69e3856e`. 30.1 min wall, peak 34.6 GB VRAM. Essentially identical dev loss to LoRA r=32 (6.488). Unlocking 1.6B params did nothing. Ceiling is representational, not param count.** |
| — | T2-orpheus-diagnostic | Orpheus 3B, LoRA r=64 (dry-run) | L40S HF Jobs | ✅ completed | — | — | — | n/a | Job `69e3c450cd8c002f31dfeb72`. Pipeline gate. |
| — | T1-baseline | Sesame CSM 1B, LoRA r=32 | L40S HF Jobs | ✅ completed | 18.61 | n/a | n/a | Noise | 120-step proof-of-pipeline. `T1/` |
| — | T1-diagnostic | Sesame CSM 1B, LoRA r=32 | L40S HF Jobs | ✅ completed | 16.05 | n/a | n/a | Noise | 5 samples × 2 steps. `T1_diagnostic/` |
| — | T5-qwen3-1p7b | Qwen3-TTS 1.7B smoke | L40S HF Jobs | ✅ completed | 10.42 | n/a | n/a | Sample WAV generated; listening not recorded | Patched rerun `69e4da24ac288e522d8f001a`. |
| — | T5-qwen3-1p7b-full | Qwen3-TTS 1.7B full pilot | L40S HF Jobs | ✅ completed | 4.74 (step 630) | n/a | n/a | Pilot metadata; listening not recorded | Patched rerun `69e4da24ac288e522d8f0018`. Loss 9.09→4.74 by step 630. |
| — | T6-mms-ewe | MMS-TTS-Ewe (VITS) via ylacombe/finetune-hf-vits | L40S HF Jobs | 🟡 running (2026-04-18) | — | — | — | — | Job `69e390f1`. **Highest-leverage experiment**: closest Gbe-family prior. |
| — | T8-ims-toucan | IMS-Toucan multilingual low-resource fine-tune | A100 HF Jobs | 🟥 blocked in corpus caching | — | — | — | — | Latest completed retry `69e43075ac288e522d8efe98` cleared earlier `matplotlib`, `transphone`, `phonepiece`, and Hub-API blockers, then failed later in corpus caching when `speechbrain/spkrec-ecapa-voxceleb` 404ed on `custom.py`. |
| — | T9-xtts-v2 | XTTS-v2 GPT encoder fine-tune | L40S / A100 HF Jobs | 🟨 compatibility fix still in progress | — | — | — | — | The older `BeamSearchScorer` blocker is fixed, but the latest L40S retry `69e42ddccd8c002f31dfefb1` exposed the next `transformers` API drift around `SampleOutput` and legacy `transformers.generation_utils`. |
| — | T7-voxcpm-smoke | VoxCPM2 LoRA dry-run smoke | A100 HF Jobs | ✅ completed | — | — | — | Non-empty audio generation during validation | Job `69e3ea21cd8c002f31dfed1e` ran 2 real train steps plus validation/audio generation, wrote a LoRA checkpoint, and uploaded metrics/checkpoints. |
| — | T7-voxcpm-fullft | VoxCPM2 full fine-tune | A100 HF Jobs | 🟡 running | — | — | — | — | Full-FT job `69e43943ac288e522d8efea4` was submitted with `--full-finetune --learning-rate 1e-5 --num-iters 1000` for a fairer architecture evaluation than the short LoRA pilot. |
| — | T10-f5 / T11-e2 | F5-TTS / E2-TTS flow-matching fine-tune | A100 HF Jobs | ✅ smoke reached training | — | — | — | — | Latest retries `69e42b13ac288e522d8efe87` (F5) and `69e42b13ac288e522d8efe8b` (E2) cleared the earlier EMA/deepcopy and dataset-path bugs, loaded the Adja bundle, and logged checkpoint saves. |

## How to listen to generated audio

All generated WAVs are on the Hub at `JosueG/adja-tts-results`. Click any link, then the play button on the file page.

**T2 Orpheus outputs** (5 WAVs each × 5 variants = 25 clips):

- [T2 full-FT (best, 5.4242)](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_fullft_10ep_2026-04-18/generated_audio)
- [T2 LoRA r=128 (5.4596)](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_lora_r128_20ep_2026-04-18/generated_audio)
- [T2 LoRA r=64 (5.4823)](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_lora_r64_20ep_2026-04-18/generated_audio)
- [T2 LoRA r=32 (5.4976)](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_en_lora_r32_20ep_2026-04-18/generated_audio)
- [T2 LoRA r=64 **French base** (5.5058)](https://huggingface.co/JosueG/adja-tts-results/tree/main/T2_orpheus_fr_lora_r64_20ep_2026-04-18/generated_audio)

Each folder contains `plain_00.wav` through `plain_04.wav` — five Adja text prompts decoded through the SNAC 24kHz codec.

Bulk download for A/B listening:

```bash
hf download JosueG/adja-tts-results \
  --include "T2_orpheus_*/generated_audio/*.wav" \
  --local-dir /tmp/adja-t2-audio
```

## Evaluation Notes

### T1-long-20ep-es (canonical baseline)

- Training: 1234 train samples after filtering clips > 10s (dropped 43 of 1277). 140 dev / 160 test. Seed 42, 80/10/10 split.
- Hyperparameters: LoRA r=32 (29M trainable / 1632M = 1.78%), lr 2e-4 cosine, effective batch 16, bf16 mixed precision, gradient checkpointing OFF.
- Eval loss trajectory (dev set, every 50 steps):
  - Monotonic decrease from 6.776 (ep 0.65) to 6.488 (ep 3.85, step 300)
  - Then monotonic increase 6.488 → 6.785 for 5 consecutive evals → early stopping
- Output WAVs saved at 24 kHz, 10 seconds each, PCM.
- Plain-text generation: all 5 clips produced non-empty waveforms, but perceived as noise/Mimi codec artifacts — no identifiable Adja phonemes.
- Speaker-conditioned generation: same story. Reference audio was not retained in output voice.
- Handling of Adja special characters: NFC-normalized at preprocess time. Llama BPE tokenizer fragments them into byte sequences; model needs to learn the byte→phoneme mapping from scratch (this is part of why 1.7h isn't enough).

### What eval loss 6.5 means in context

For reference, well-trained TTS on in-distribution English typically reaches 2–4 on this kind of next-audio-token cross-entropy. 6.5 is "learned something but far from fluent."

## Open Questions / Next Experiments

1. **LoRA r=128 / Full fine-tune** — Is the ceiling capacity or data? If r=128 barely moves the needle, data is the blocker.
2. **MMS-TTS-Ewe starting point** — Ewe is the closest language to Adja that Meta's MMS-TTS supports. Fine-tuning from MMS-Ewe should give us a model that already "speaks Gbe-family phonology." Much better starting point than English-centric CSM.
3. **IMS-Toucan** — Built for low-resource tonal languages with language embeddings from Glottolog. Possibly the most principled approach for our exact constraint.
4. **XTTS-v2** — Multilingual TTS with cross-language voice cloning; known to fine-tune with 1-2h data.
5. **Data augmentation** — Pitch shift, speed perturbation, SpecAugment on Mimi codec latents could 2-3x effective dataset size.
6. **NLLB tokenizer swap** — Research idea in `ideas/nllb-tokenizer-for-tts.md`. Replace Llama BPE with NLLB SentencePiece, which knows Gbe-family languages. Architecture surgery required.

See `ideas/tts-models-to-try.md` for deeper model-choice discussion.
