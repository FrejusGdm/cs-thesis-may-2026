# What We Know

Synthesis of confirmed findings, refuted hypotheses, inconclusive results, untested assumptions,
and open questions for the best-TTS-African-Languages ablation suite.

All job IDs and dates are drawn directly from `experiments/registry.md`.

---

## Confirmed

**M0 PASS: Mimi encode→decode reconstructs Ewe audio with excellent fidelity (SC = 0.1733, L1 = 0.00637).**
Run 2026-04-26, local CPU, 5 WaxalNLP ewe_tts samples at 24kHz.
Per-sample SC: 0.1931, 0.1578, 0.1879, 0.1520, 0.1758 — all well below the 0.30 "excellent" threshold.
Significance: `kyutai/mimi` is NOT the bottleneck for Gbe/tonal audio. The off-the-shelf codec
preserves tonal contrasts and vowel contrasts (ɛ, ɔ, ɖ, tone marks) faithfully. M1/M2/M2b
(codec fine-tuning experiments) are therefore **deprioritized** — the CF and AT experiment groups
are the correct focus for fixing Stage 2 Adja synthesis. Resources should go to CF1/CF2/CF3 and AT1.

**CSM 1B + 1,215 WaxalNLP Ewe TTS clips → intelligible Ewe.**
Job `69e830e1ac288e522d8f0782`, 2026-04-21. 17.4 min, L40S, best eval_loss=4.326 @ epoch 3.86.
Native-speaker listening verdict (Josue Godeme, 2026-04-22): intelligible Ewe.
Significance: architecture (CSM 1B, Mimi codec, Llama BPE) is not the bottleneck. Given adequate
data volume (1,215 clips) in a Gbe-family language, the model learns coherent synthesis.

**Orpheus 3B EN + WaxalNLP Ewe TTS → intelligible Ewe.**
Job `69e846e8ac288e522d8f084f`, 2026-04-22, LoRA r=64. Best eval_loss ~0.16 @ epoch 14.
Native-speaker listening verdict (Josue Godeme, 2026-04-22, via inference job `69e8f639d2fd2eb837d76a72`):
"Orpheus EWE is also intelligible good stuff."
Significance: same confirmation as CSM above, for the Orpheus 3B + SNAC architecture.

**Orpheus 3B FR + WaxalNLP Ewe TTS → intelligible Ewe.**
Job `69e846eaac288e522d8f0851`, 2026-04-22, LoRA r=64. Best eval_loss ~0.16 @ epoch 14.
Native-speaker listening verdict (Josue Godeme, 2026-04-22, via inference job `69e8f63ad2fd2eb837d76a74`):
confirmed intelligible. EN and FR base models achieve identical Stage 1 eval loss; Stage 2 Adja
quality is the real comparison between them.

**Spark TTS 0.5B (Qwen2 BPE + XLSR-53 BiCodec) → intelligible Adja in ~120 training steps on ~1.7h
direct Adja.**
Job `69e3987d`, 2026-04-18. No Ewe stage required. Native-speaker listening verdict (Josue Godeme,
2026-04-18): coherent Adja words.
Significance: multilingual priors (Qwen2 BPE covers African characters natively; BiCodec includes
XLSR-53 multilingual acoustic features) are sufficient for Adja TTS with the same data volume where
CSM and Orpheus fail completely.

**CSM Stage 2 (Ewe checkpoint → Adja) ran full 20-epoch budget without early stopping and still
produced noise — training duration is not the bottleneck.**
Job `69e93755d2fd2eb837d76d4e`, 2026-04-22. 37.5 min. eval_loss_best=6.5115. Early stop did NOT
fire despite patience=5. All 5 generated samples hit the 10s cap (silence-termination not learned).
Significance: this is the critical evidence distinguishing "the model needed more training time"
from "the model cannot learn coherent Adja generation with this training setup." Budget was not the
constraint.

**Orpheus EN and FR Stage 2 (Ewe checkpoint → Adja) both produced noise with nearly identical loss
plateaus.**
Jobs `69e937572aa1660eaffa8c5a` (EN, best_eval_loss=5.6156) and `69e9375a2aa1660eaffa8c5c`
(FR, best_eval_loss=5.6356), both 2026-04-22. Natural durations (1.96-3.24s). Early stop fired at
patience=5. Native-speaker listening verdict (Josue Godeme, 2026-04-22): noise.
Significance: both base models fail Stage 2 identically. EN vs FR base distinction does not matter
at Stage 2; the forgetting dynamic is the same for both.

**Initial Stage-2 training loss on Adja was ~17 nats/token (catastrophic forgetting signature).**
Orpheus EN Stage 2, job `69e937572aa1660eaffa8c5a`. For a SNAC codebook of ~4096 entries,
random-chance baseline is `ln(4096) ≈ 8.3`. A loss of 17 means the Stage-1 model was confidently
generating Ewe-patterned audio tokens when prompted with Adja text — not ignorant, actively wrong.
Significance: this is canonical catastrophic forgetting, not random-initialization noise.

**1:1 Ewe:Adja mixed Stage 2 is insufficient to prevent catastrophic forgetting.**
Job `69e977f82aa1660eaffa8d21`, 2026-04-23. 53.7 min, L40S. best_eval_loss=5.648 @ ep 6.79.
Native-speaker listening verdict (Josue Godeme, 2026-04-23): "better noise than before, but still
noise." Loss plateau virtually identical to pure-Adja Stage 2 (5.648 vs 5.616). Naive 1:1 data
mixing is not sufficient to preserve the Gbe prior during Stage 2 adaptation.
Significance: the forgetting mitigation bar is higher than simple data replay at 1:1 ratio.

---

## Refuted

**"Tokenizer fragmentation is the primary bottleneck in LLaMA-based TTS for Adja."**
This was the leading hypothesis before 2026-04-22.
Refutation: T1-tokfix (CSM 1B + expanded Llama tokenizer with 35 Adja characters added,
direct Adja fine-tune) produced noise with only rare intelligible fragments.
Job `69e857f3cd8c002f31e016a6`, 2026-04-22. Required `config.vocab_size` restore patch.
Listening verdict (Josue Godeme, 2026-04-22): noise with rare speech-like fragments —
only marginally better than vanilla T1. Data volume and Gbe prior dominate; tokenizer
fragmentation is a secondary effect.
Note: this refutation applies to the data-insufficient regime (~1.7h direct Adja). Whether
tokenizer matters in the data-sufficient regime (after Stage 1 Ewe pretraining) is still open —
see TK1 in the experiment table and RESEARCH-QUESTIONS.md RQ4.

**"The Mimi codec / architecture is the blocker for CSM on any non-English language."**
Refutation: CSM 1B fine-tuned on Ewe (job `69e830e1`) produces intelligible Ewe, a Gbe-family
tonal language, using the same Mimi codec and Llama BPE architecture as the failing direct-Adja runs.
The architecture handles at least one African tonal language given sufficient data. The original
hypothesis was formed before any non-English CSM result was available.

**"The French Orpheus base should outperform the English base for Adja due to phonological proximity
of French to African languages."**
Refutation: Orpheus EN and FR Stage 1 achieve identical eval_loss (~0.16 @ ep 14) and both Stage 2
variants produce noise with indistinguishable loss plateaus (5.6156 vs 5.6356). The French base
offers no measurable advantage at either stage.

---

## Inconclusive

**T1-tokfix: tokenizer expansion on ~1.7h Adja data → noise, but data volume confounds the result.**
Job `69e857f3cd8c002f31e016a6`, 2026-04-22. The tokenizer expansion was tested in the same
data-insufficient regime (~1.7h Adja) where all LLaMA-based TTS fails. The experiment shows
tokenizer is not the *primary* bottleneck in this regime, but it does not answer whether a better
tokenizer would help in the data-sufficient regime (e.g., after Stage 1 Ewe pretraining). TK1 is
the planned experiment to disentangle these.

**Orpheus FR vs EN Stage 2 Adja loss plateau (5.6356 vs 5.6156) — too close to interpret.**
Both are noise. The 0.02 difference in eval loss is within run-to-run noise and does not support
any conclusion about base model language proximity.

**Orpheus ZH (Mandarin) Stage 1 — grad-graph issue, not yet resubmitted with fix.**
T2-orpheus-zh-ewe-stage1: original submission from 2026-04-21 wave failed with the same grad-graph
issue that EN/FR fixed via `model.enable_input_require_grads()`. Not yet resubmitted. Hypothesis
(unconfirmed): the ZH base, having been pretrained on a 4-tone language (Mandarin), might bring
stronger tonal LM prior to Gbe family languages than EN or FR. This is theoretically motivated
but untested.

**Spark TTS Stage 1 Ewe — infra-blocked, not yet completed.**
T3-spark-ewe-stage1: as of 2026-04-26, failing due to infra issues (triton OOM on L40S, latest
retry `69e8e3efd2fd2eb837d769b3` moved to A100 large). Outcome pending.
Implication: if Spark Stage 1 Ewe completes and produces intelligible Ewe (as expected given Spark's
direct-Adja success), Spark Stage 2 Ewe→Adja is the highest-probability candidate for a positive
Stage 2 result, because Spark already has a multilingual prior that the cascade only reinforces.

---

## Assumption (not proven)

**"WaxalNLP `ewe_tts` curated clips are better than `ewe_asr` multi-speaker clips for Stage 1
TTS training."**
Rationale: TTS models benefit from clean, consistent single-speaker audio; ASR configs include
diverse speakers, noise, and varied conditions. The WaxalNLP team separated the configs
deliberately. We trusted their curation choice and used `ewe_tts` (1,215 clips) rather than
`ewe_asr` (15,054 clips). This choice is documented in
`docs/gbe-cascade-tts-settings-2026-04-22.md §2.2`, but the `ewe_asr` ablation has not been run.
It is possible that 12x more data outweighs the quality disadvantage; also possible that it hurts.

**"LoRA r=32 is sufficient capacity for Stage 1 and Stage 2 CSM adaptation."**
CSM Stage 1 used LoRA r=32 and produced intelligible Ewe. Stage 2 used the same rank and produced
noise. It is not known whether higher rank (e.g., r=64 or r=128, matching Orpheus) would change
the Stage 2 outcome. The CSM Stage 2 failure was attributed to forgetting, not rank.

---

## Untested

The following experiments from the `README.md` experiment table have not been run:

| ID | What it tests |
|----|--------------|
| M1 | Acoustic Mimi fine-tune on Adja (L1 + STFT loss) — deprioritized: M0 passed |
| M2 | Semantic Mimi fine-tune with MMS-300M teacher (codebook-0 distillation) |
| M2b | Ablation: acoustic-only (M1) vs semantic-distilled (M2) |
| CF1 | Lower LR Stage 2 (1e-5 or 5e-6) anti-forgetting |
| CF2 | EWC regularization toward Stage 1 weights during Stage 2 |
| CF3 | Annealing Ewe:Adja mix ratio (1:1 → 1:9 over epochs) |
| TK1 | Character-level or NLLB tokenizer in CSM backbone, Stage 1+2 cascade |
| AT1 | Audio-LM SSL pretraining on 183k unlabeled Ewe clips before supervised TTS |

M0 has run and passed (SC=0.1733). M1/M2/M2b are deprioritized.
**Highest-priority paid experiments: CF2 (curriculum annealing), CF3 (frozen backbone), CF1 (EWC).**
