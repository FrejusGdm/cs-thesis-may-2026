# Research Questions

Academic framing for the best-TTS-African-Languages ablation suite.

The overarching question is:

> What is the binding constraint in LLaMA-based TTS for Adja (and African low-resource languages
> more broadly), and which interventions are sufficient to overcome it?

---

## RQ1: Is catastrophic forgetting the primary bottleneck in Gbe-family cascade transfer?

**Operationalization**: Gbe-family cascade (Stage 1 Ewe → Stage 2 Adja) produces intelligible Ewe
after Stage 1 but noise after Stage 2. Is the failure explained by the Stage 2 model unlearning
Ewe phonology before accumulating Adja phonology, or by some other failure mode?

**Evidence motivating this question**: CSM 1B Stage 2 ran its full 20-epoch budget without early
stopping and still produced noise (job `69e93755`). Orpheus EN Stage 2 initial loss was ~17 nats/token
vs a random baseline of `ln(4096) ≈ 8.3` — meaning the Stage-1 model was confidently generating
Ewe-patterned audio tokens when prompted with Adja text. 1:1 mixed Stage 2 (job `69e977f8`)
reduced loss marginally but still produced noise.

**What a positive result looks like**: any of CF1 (lower LR), CF2 (EWC), or CF3 (annealed mix)
yields a Stage 2 Adja listening verdict of "intelligible." This would confirm forgetting as the
binding constraint and establish a repeatable recipe for cross-lingual Gbe-family TTS transfer.

**What a negative result means**: forgetting mitigation does not rescue Stage 2 regardless of LR
or regularization. This would redirect the hypothesis toward insufficient Gbe-family prior strength
in Stage 1 (need more Ewe data — AT1 track), codec mismatch for Adja specifically (M1/M2 track),
or a fundamental data-volume ceiling (~1.7 hours is not enough for any LLM-based TTS to learn Adja
regardless of pretraining). A negative result on CF is informative and publishable if the analysis
is sharp.

**Paper claim if positive**: "Elastic Weight Consolidation / mixed-data replay during Stage 2
preserves the Gbe phonological prior accumulated in Stage 1, enabling intelligible Adja TTS from a
model family that produces only noise when fine-tuned on Adja directly."

**Paper claim if negative**: "Catastrophic forgetting mitigation alone is insufficient; the data
volume ceiling for Stage 2 Adja adaptation dominates even when Gbe phonology is preserved."

**Experiments**: CF1, CF2, CF3 (see `README.md` experiment table)

---

## RQ2: Does codec adaptation (Mimi fine-tuning) improve African phoneme representation?

**Operationalization**: Mimi (Kyutai, used in CSM) was trained on English-heavy data. Does
fine-tuning the Mimi encoder/decoder on Adja audio improve the reconstructed audio quality, and
does a CSM pipeline using the adapted Mimi produce better TTS output than one using the original
Mimi?

**Evidence motivating this question**: The original hypothesis ("Mimi cannot represent Adja
phonemes") was partially refuted by the observation that Orpheus Mandarin (3b-zh, same SNAC
architecture pattern) produces intelligible Mandarin — a 4-tone language — suggesting codec priors
are not the primary failure mode. However, this is indirect evidence. Mimi is a different codec
from SNAC, and direct reconstruction quality on Adja has not been measured.

**What a positive result looks like**: M0 reconstruction gate reveals audible phoneme distortion on
Adja audio. M1/M2 fine-tuning reduces that distortion. CSM Stage 2 with adapted Mimi improves
listening verdict relative to CSM Stage 2 with original Mimi.

**What a negative result means**: M0 reconstruction is already clean (Mimi handles Adja fine).
This does not mean the codec hypothesis was wrong — it means the codec is not the binding
constraint at this stage. Focus shifts entirely to CF/AT experiments.

**Paper claim if positive**: "Mimi's acoustic representation of Adja phonemes is degraded relative
to English, and domain adaptation of the codec encoder/decoder is a necessary prerequisite for
intelligible Adja synthesis with CSM-family models."

**Paper claim if negative (and M0 clean)**: "Mimi's reconstruction quality on Adja is sufficient;
the bottleneck lies in the LM backbone's token-to-acoustic mapping, not in the codec's phoneme
coverage." This is itself a useful ablation result for the paper.

**Gate experiment**: Run M0 before M1 or M2. M0 is free and local; if it shows clean
reconstruction, skip M1/M2 entirely.

**Experiments**: M0 (gate), M1 (acoustic-only Mimi fine-tune), M2 (semantic-distilled Mimi
fine-tune), M2b (ablation M1 vs M2)

---

## RQ3: Does audio-LM pretraining on unlabeled Gbe audio improve downstream supervised TTS?

**Operationalization**: Does self-supervised pretraining of the CSM LM backbone on unlabeled Gbe
speech (WaxalNLP `ewe_asr` 183,920 unlabeled clips + any available Adja unlabeled data) before
supervised TTS fine-tuning improve the intelligibility of the final Adja output?

**Evidence motivating this question**: The Stage 1 Ewe supervised fine-tune uses only 1,215 labeled
clips. WaxalNLP ships 183,920 unlabeled Ewe clips that are not used in any current experiment.
Self-supervised audio-LM pretraining (predicting masked Mimi tokens from context) is a known
technique for domain adaptation of speech LMs with unlabeled data.

**What a positive result looks like**: AT1 (audio-LM pretrained backbone) + Stage 2 Adja produces
intelligible output where the Stage 1-only cascade failed, or produces better listening scores
than Stage 1-only when Stage 1 already partially works.

**What a negative result means**: 183k unlabeled clips does not overcome the Adja supervised-data
ceiling. This would suggest that the binding constraint is truly the supervised Adja data volume,
not the LM's prior over Gbe phonology. Important negative result for papers on low-resource TTS.

**Paper claim if positive**: "Unlabeled Gbe audio provides sufficient phonological prior for LM-TTS
backbone pretraining, reducing the supervised data requirement for Adja adaptation below the 1.7-hour
threshold where direct fine-tuning fails."

**Paper claim if negative**: "The 183k-clip unlabeled-audio LM pretraining stage does not overcome
the 1.7-hour Adja data ceiling, confirming that supervised TTS quality is currently gated by labeled
data volume rather than acoustic prior."

**Note**: AT1 is the most expensive experiment in this suite (~$15-30 on A100 80GB). Run only after
CF results are in to avoid duplicating work if CF resolves the forgetting problem independently.

**Experiments**: AT1

---

## RQ4: Does text representation quality (tokenizer) affect TTS quality when data is sufficient?

**Operationalization**: The T1-tokfix experiment (CSM + expanded Llama tokenizer + direct Adja)
produced noise, which was interpreted as refuting the "tokenizer is the primary bottleneck"
hypothesis. However, that experiment had only ~1.7 hours of direct Adja data — the same regime
where even full fine-tune with the original tokenizer fails. Does tokenizer quality matter once
data volume is no longer the bottleneck, i.e., after Stage 1 Ewe pretraining provides a Gbe prior?

**Operationalization (revised)**: Compare CSM Stage 2 with original Llama BPE vs character-level
or NLLB tokenizer, starting from the same Stage 1 Ewe checkpoint. Same Adja data, same training
budget, same LoRA configuration. Only the text tokenizer changes.

**Evidence motivating this question**: T1-tokfix refuted tokenizer as the *primary* bottleneck
when data is insufficient. It does not answer whether tokenizer matters in the regime where data
is sufficient. The H-Net paper (`https://arxiv.org/html/2507.07955v2`) argues that fixed
tokenization is a non-trivial design choice for diverse languages. Adja's ɛ, ɔ, ŋ, ɖ, and tone
marks are byte-fragmented under Llama BPE.

**What a positive result looks like**: TK1 (character or NLLB tokenizer + Stage 1 pretraining)
produces better intelligibility or lower eval loss than baseline Stage 2 with Llama BPE. This
would rehabilitate the tokenizer hypothesis for the data-sufficient regime.

**What a negative result means**: Tokenizer does not matter once the Gbe prior is in place. The
Stage 1 Ewe pretraining is sufficient to teach the LM to map Adja byte-fragments to coherent
Adja audio tokens. Negative result is consistent with T1-tokfix and strengthens the paper's
"data volume and Gbe prior, not tokenizer" narrative.

**Dependency**: TK1 should be run after at least one successful CF variant (otherwise we are
still in the data-insufficient regime where T1-tokfix already showed tokenizer doesn't help).

**Paper claim if positive**: "Text representation quality affects TTS intelligibility even when
Gbe-family prior is in place: character or multilingual subword tokenization reduces the text
conditioning error and improves Adja synthesis over Llama BPE byte-fragments."

**Paper claim if negative**: "Tokenizer choice does not materially affect Stage 2 TTS quality
once the Gbe-family prior is established, suggesting that the LM learns to compensate for
byte-fragment representations when phonological coverage is adequate."

**Experiments**: TK1
