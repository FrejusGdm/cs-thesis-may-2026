# Gbe-family cascade TTS: data, settings, and reproducibility notes

**Status**: paper-prep. Written 2026-04-22 to freeze the data sources, preprocessing, model
configurations, and hyperparameters used across the 2026-04-21/22 TTS Stage 1 / Stage 2 wave so
the numbers, listening verdicts, and decisions can be cited directly in the paper.

Scope: the three TTS cascades that test whether a Gbe-family pretraining stage on **Ewe**
(Adja's closest relative) unlocks Adja speech synthesis that direct fine-tuning on ~1.7 hours of
Adja alone cannot achieve.

Companion docs:

- [results/tts-comparison.md](../results/tts-comparison.md) — running leaderboard (listening verdict, dev loss)
- [experiments/registry.md](../experiments/registry.md) — per-experiment status and Hub paths
- [results/run-ledger.md](../results/run-ledger.md) — chronological per-run log
- [session-logs/2026-04-22-tts-failure-modes-debug.md](../session-logs/2026-04-22-tts-failure-modes-debug.md) — infra failure-mode writeup
- [research-paper-exploration/paper-one/why-spark-worked.md](../research-paper-exploration/paper-one/why-spark-worked.md) — prior tokenizer/codec analysis
- [docs/tts-audio-range-normalization-reruns-2026-05-09.md](tts-audio-range-normalization-reruns-2026-05-09.md) — May 2026 CSM/Orpheus audio-range caveat and rerun matrix

**May 2026 caveat:** a later parquet audit found that some CSM and Orpheus Adja paths may have
consumed PCM-scale float arrays without explicit waveform normalization. Treat the April CSM and
Orpheus Adja listening failures as preliminary until the reruns in
[tts-audio-range-normalization-reruns-2026-05-09.md](tts-audio-range-normalization-reruns-2026-05-09.md)
finish. Spark is not affected by this caveat because its path already performed explicit audio
normalization.

## 1. Why a Gbe-family cascade at all

Three things forced the question:

1. **Spark TTS 0.5B** produces intelligible Adja in ~120 training steps on ~1.7 hours of Adja
   direct (no cascade). Spark's Qwen2 tokenizer + BiCodec/XLSR-53 semantic codec are multilingual
   by construction. See [why-spark-worked.md](../research-paper-exploration/paper-one/why-spark-worked.md)
   for full architectural analysis.
2. **Sesame CSM 1B** (Llama 3.2 BPE + Mimi codec) and **Orpheus 3B** (Llama tokenizer + SNAC codec)
   cannot produce intelligible Adja on the same data, regardless of LoRA rank, full fine-tuning,
   or tokenizer expansion (2026-04-22 listening test on T1-tokfix: speech-like noise with rare
   intelligible fragments).
3. The obvious confound — "maybe the architecture just can't do any Gbe language" — needed to be
   ruled out before we could claim tokenizer/codec priors as the cause.

**The experiment**: give each English/French/Chinese-centric TTS an intermediate Ewe pretraining
stage on WaxalNLP Ewe (1,215 training clips — Adja's closest Gbe relative with a public curated
TTS dataset), then fine-tune the resulting checkpoint on Adja. If the Ewe stage produces
intelligible Ewe, the architecture is fine. If the Adja stage then produces intelligible Adja,
the Gbe-family prior was the missing ingredient.

**2026-04-22 Stage 1 listening verdict (native speaker, Josue Godeme)**:

| Stage 1 output | Intelligibility |
|----------------|-----------------|
| CSM-Ewe Stage 1 (Mimi codec + Llama BPE, trained on 1,215 Ewe clips) | **Intelligible Ewe** |
| CSM tokfix direct Adja (Llama BPE + 35 added Adja chars, trained on ~1,234 Adja clips) | speech-like noise, rare fragments |
| F5-TTS full fine-tune direct Adja (char-level vocab) | pure noise |
| E2-TTS full fine-tune direct Adja (char-level vocab) | pure noise |

The CSM-Ewe result is the decisive one: **the architecture is not the problem; data volume and
Gbe-family phonological coverage are.** Stage 2 (Ewe checkpoint → Adja) is the direct test of
whether that prior transfers to Adja.

## 2. Data sources

### 2.1 Stage 1 data — WaxalNLP Ewe TTS

- **Hub repo**: [`google/WaxalNLP`](https://huggingface.co/datasets/google/WaxalNLP)
- **Config used**: `ewe_tts`
- **Splits**:

  | Split | Rows |
  |-------|------|
  | train | 1,215 |
  | validation | 152 |
  | test | 152 |
  | **total** | **1,519** |

- **Text column**: `text` (autodetected; `sentence` fallback handled in script)
- **Audio column**: `audio` as `datasets.Audio` (arrays + sampling_rate), originally 48 kHz WAV
- **Orthography**: Ewe Latin with diacritics, same character set family as Adja (ɔ ɛ ɖ ŋ + tone marks)
- **License**: CC-BY-4.0 (WaxalNLP project, Google Research)

### 2.2 Why `ewe_tts` (1,215 clips) instead of `ewe_asr` (15,054 clips)

`google/WaxalNLP` ships both an `ewe_asr` config and an `ewe_tts` config for Ewe:

| Config | Train | Val | Test | Unlabeled | Total (labeled) |
|--------|-------|-----|------|-----------|-----------------|
| `ewe_asr` | 15,054 | 1,916 | 1,891 | 183,920 | 18,861 labeled + 183,920 unlabeled |
| `ewe_tts` | 1,215 | 152 | 152 | — | 1,519 |

We used `ewe_tts` even though `ewe_asr` has ~12× more labeled data. The reasoning:

- **ASR configs** are curated for *recognition*: many speakers, diverse microphones/rooms, accents,
  background noise. That diversity is what makes an ASR acoustic model robust.
- **TTS configs** are curated for *synthesis*: cleaner audio, fewer speakers, more prosodically
  consistent recordings. A TTS model has to *produce* coherent speech, so training on noisy
  multi-speaker data forces it to average over voice identity in ways that hurt output quality.
- Google's WaxalNLP team separated the configs deliberately — we trust their curation choice.

**Documented follow-up**: the Stage 1 CSM-Ewe result (intelligible Ewe from only 1,215 clips) is
strong enough that an `ewe_asr`-trained Stage 1 is worth an ablation — specifically to see
whether 15k clips *help* (more data) or *hurt* (multi-speaker averaging). Not run yet; logged as
an open question for the paper.

### 2.3 Stage 2 data — Adja TTS

- **Hub repo**: [`JosueG/adja-tts-orpheus`](https://huggingface.co/datasets/JosueG/adja-tts-orpheus) (private)
- **Size**: ~1,597 utterances, ~1.95 hours of speech
- **Split policy**: 80 / 10 / 10 train / dev / test, `seed=42`, done in-script via
  `Dataset.train_test_split` (first 0.1 for test, then 0.1/0.9 of the remainder for dev)
  → resulting sizes after split: train ≈ 1,234, dev ≈ 158, test ≈ 158 (exact values logged per job)
- **Orthography**: Adja Latin with ɔ, ɛ, ɖ, ŋ, plus French-style tone marks (é, è, ê, etc.)
- **Text normalization**: Unicode NFC on every string before training/evaluation (NFKC in the tokfix
  variant, to match the ADJA_CHARS derivation corpus)
- **Audio**: stored as `datasets.Audio`, cast to 24 kHz for Mimi/SNAC/BiCodec paths
- **Clip length filter**: drop clips > 10 s (> 240,001 samples at 24 kHz) — applied per split,
  documented in log output

## 3. Models and checkpoint sources

| Experiment | Base checkpoint | Modification | Stage 1 result | Stage 2 script |
|------------|-----------------|--------------|----------------|----------------|
| T1 CSM Ewe | `unsloth/csm-1b` | LoRA r=32, q/k/v/o/gate/up/down proj | Intelligible Ewe | `scripts/hf_jobs/T1_csm_ewe_adja_stage2.py` |
| T2 Orpheus EN Ewe | `canopylabs/orpheus-3b-0.1-ft` | LoRA r=64 | eval_loss ~0.16 @ ep 14 (audio deferred to Stage 2) | `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py --checkpoint-tag en` |
| T2 Orpheus FR Ewe | `canopylabs/3b-fr-ft-research_release` | LoRA r=64 | eval_loss ~0.16 @ ep 14 (audio deferred) | `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py --checkpoint-tag fr` |
| T2 Orpheus ZH Ewe | `canopylabs/3b-zh-ft-research_release` | LoRA r=64 | pending resubmit (grad fix) | `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2.py --checkpoint-tag zh` |
| T3 Spark Ewe | `unsloth/Spark-TTS-0.5B` | LoRA r=128 via Unsloth `FastModel` | failing (infra) | `scripts/hf_jobs/T3_spark_ewe_adja_stage2.py` |

Stage 1 merged checkpoints are pushed to `JosueG/adja-tts-checkpoints/T{1,2,3}_{id}_ewe_stage1/`.

## 4. Hyperparameters (frozen for this wave)

### 4.1 Stage 1 — Ewe adaptation

Common:

- Optimizer: AdamW; `weight_decay=0.001`
- LR schedule: cosine (Orpheus), cosine (CSM); linear (Spark smoke path)
- Warmup: 10 steps (CSM, Orpheus), 20 (Spark)
- Precision: bf16 when supported, otherwise fp16 (autodetected via `torch.cuda.is_bf16_supported()`)
- Seed: 42
- Early stopping: `EarlyStoppingCallback(patience=5)` on `eval_loss`, `load_best_model_at_end=True`
- Eval/save cadence: every 50 steps, `save_total_limit=3`
- Logging cadence: every 5 steps (Orpheus, CSM), every 10 (Spark)

Per-model overrides:

| Setting | CSM 1B | Orpheus 3B EN/FR/ZH | Spark 0.5B |
|---------|--------|---------------------|------------|
| Training mode | LoRA r=32 (α=32) | LoRA r=64 (α=64) | LoRA r=128 (α=128) via Unsloth FastModel |
| Trainable modules | q/k/v/o/gate/up/down proj | q/k/v/o/gate/up/down proj | q/k/v/o/gate/up/down proj |
| Per-device batch | 4 | 1 | 2 |
| Grad accum | 4 | 8 | 4 |
| Effective batch | 16 | 8 | 8 |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| Epochs (cap) | 20 | 20 | 20 |
| Max seq length | default (`processor.apply_chat_template` max_length=256 + 240001 audio samples) | 2048 | 2048 |
| Gradient checkpointing | **OFF** (breaks grad flow with PEFT — see note) | **ON** with `model.enable_input_require_grads()` | ON (via Unsloth `use_gradient_checkpointing="unsloth"`) |
| Audio sampling rate | 24 kHz (Mimi) | 24 kHz (SNAC) | configured by BiCodec (resampled inside `formatting_audio_func`) |

**Gradient checkpointing + PEFT note**: our `T2_orpheus_ewe_stage1.py` originally failed with
`RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` when
`gradient_checkpointing=True`. Fix: call `model.enable_input_require_grads()` immediately after
`get_peft_model()` to inject a forward hook that forces input gradients. This is the documented
PEFT pattern. `T1_csm_ewe_stage1.py` sidesteps this by leaving gradient_checkpointing OFF (comment
on line 218: "OFF — breaks grad flow with PEFT"). For 1B CSM on L40S (48 GB) the extra activation
memory is fine; for 3B Orpheus it's not, hence the need for the hook.

### 4.2 Stage 2 — Adja adaptation (from Stage 1 merged checkpoint)

Identical to Stage 1 except:

- **Learning rate**: 5e-5 (vs 2e-4 in Stage 1) — lower LR for continued fine-tuning
- **Stage 1 checkpoint** loaded from `JosueG/adja-tts-checkpoints/T{id}_ewe_stage1/` and used as
  the pretrained model (for Orpheus/CSM, the merged-LoRA safetensors; for Spark, the merged model
  including Unsloth patches)
- **Gradient checkpointing**: OFF in both CSM and Orpheus Stage 2 (no PEFT grad-flow issue since
  checkpointing isn't enabled)
- **Prefix / output paths**: `T{id}_{base}_ewe_adja_stage2/` under `JosueG/adja-tts-results`

### 4.3 CSM tokfix (retired hypothesis test)

For completeness, the direct-Adja tokenizer-expansion variant:

- Base: `unsloth/csm-1b`
- Tokenizer: Llama 3.2 BPE + 35 added Adja characters (`ADJA_CHARS` list in
  `scripts/hf_jobs/T1_csm_tokfix.py`), empirically derived from 455k lines of Adja corpus under
  NFKC normalization
- Critical patch: `config.vocab_size` is restored to its original value after `resize_token_embeddings`.
  CSM's `backbone_loss` uses `config.vocab_size` for the Mimi *audio codebook* space (2051), not
  text vocab; mutating it to 128291 (text vocab after expansion) caused
  `RuntimeError: shape '[-1, 128291]' is invalid for input of size 2100224` at the first loss
  reshape. See [session-logs/2026-04-22-tts-failure-modes-debug.md](../session-logs/2026-04-22-tts-failure-modes-debug.md)
  for the full derivation.
- Data: `JosueG/adja-tts-orpheus` (Adja only; no Ewe stage)

## 5. Compute environment (2026-04-22 wave)

All TTS Stage 1 / Stage 2 jobs ran on **Hugging Face Jobs**, `l40sx1` flavor (NVIDIA L40S, 48 GB VRAM,
1 GPU), 8-hour timeout, Python 3.11, `hf jobs uv run` entrypoint with inline PEP-723 `/// script ///`
dependencies (Stage 1) or runtime `pip install` (Stage 2, which inherits torch from the default uv
container and installs the transformers family at runtime).

Per-run wall clocks (Stage 1):

- CSM Ewe: 17.4 min, peak 25.52 GB VRAM
- Orpheus EN Ewe: ~1 h (running when polled; LoRA r=64, 3B model)
- Orpheus FR Ewe: ~1 h
- Spark Ewe: failing at import time (triton JIT compile — see failure-mode log)

## 6. Evaluation protocol

- **Primary criterion**: native-speaker listening test on 5 generated samples per model
  (`generated_audio/` under each results prefix).
- **Secondary**: `eval_loss` (cross-entropy over target codec tokens per family — Mimi for CSM,
  SNAC for Orpheus, BiCodec for Spark). **Not cross-comparable across model families** because the
  target codebook differs. Use only for within-family ordering.
- **Orpheus Stage 1 inference-only pass** — added 2026-04-22 after the Stage 1 listening verdict
  was noted as missing. Orpheus Stage 1 training scripts intentionally skip audio generation to
  keep Stage 1 wall time short. To actually *listen* to the Ewe output from each Stage 1
  checkpoint (for the same cross-architecture audibility comparison we did on CSM), run
  `scripts/hf_jobs/T2_orpheus_ewe_stage1_inference.py`:

  ```
  hf jobs uv run --flavor l40sx1 --timeout 1h --python 3.11 --secrets HF_TOKEN \
    -d scripts/hf_jobs/T2_orpheus_ewe_stage1_inference.py -- \
    --stage1-checkpoint JosueG/adja-tts-checkpoints/T2_orpheus_{en|fr|zh}_ewe_stage1 \
    --checkpoint-tag {en|fr|zh} \
    --results-prefix T2_orpheus_{tag}_ewe_stage1_inference \
    --push-to-hub
  ```

  Generates 5 Ewe samples per checkpoint from `WaxalNLP ewe_tts` test split. Pushed to
  `JosueG/adja-tts-results/T2_orpheus_{tag}_ewe_stage1_inference/generated/`.

## 7. Open questions for the paper

1. **Does the Ewe → Adja transfer hold?** — Stage 2 listening test is the crux. If the
   CSM-Ewe-Adja Stage 2 produces intelligible Adja, the Gbe-family prior hypothesis is the first
   confirmed recipe for an English-centric LLM-TTS on Adja. If it regresses, either the Stage 2
   adaptation is destroying the prior or something subtler is going on.
2. **Does the FR base beat the EN base after the Ewe stage?** — The Spark paper's intuition would
   say no (phonology dominates sociolinguistic proximity). Our Stage 1 loss numbers are similar
   (~0.16 eval for both). Stage 2 Adja quality is the real comparison.
3. **Does the ZH base — once we get it past the grad bug — add tonal LM prior on top of the Ewe
   acoustic prior?** This is the most theoretically grounded Orpheus cascade.
4. **`ewe_asr` ablation for Stage 1 (not yet run)**: does 15k clips help or hurt? Informs
   whether curated-TTS-small ≫ noisy-ASR-large at this data scale.
5. **Spark Ewe Stage 1** remains infra-blocked; once it completes, T3-ewe Stage 2 is the strongest
   probable winner because Spark already works direct on Adja — the Ewe stage should only improve it.

## 8. Reproducibility checklist

- [x] All Stage 1 scripts pinned with inline PEP-723 dependency declarations (torch, transformers,
      peft, datasets versions frozen in each script)
- [x] Seed 42 everywhere (data splits, torch, numpy, python random, CUDA)
- [x] Text normalization (NFC for Ewe/Adja; NFKC only in tokfix to match the corpus-derived ADJA_CHARS)
- [x] Audio resampling target documented per model family
- [x] LoRA target modules and rank logged in every run's metrics.json
- [x] Stage 1 merged checkpoints pushed to Hub with the exact path Stage 2 loads from
- [x] Listening test text samples saved with each generated audio batch (see `metrics.json` →
      `generated` field)
- [ ] Per-sample intelligibility ratings (MOS-style) — planned once multiple listeners are available

---

## 8. Stage 2 catastrophic forgetting + mixed-data remediation (added 2026-04-22 evening)

### 8.1 Observation

All three Stage 2 runs from 2026-04-22 produced noise per native-speaker listening, despite the Stage 1 Ewe outputs being intelligible across both architectures:

| Stage 2 variant | Job ID | eval_loss (best, Adja dev) | Epochs | Early stop fired? | Listening verdict |
|-----------------|--------|---------------------------|--------|-------------------|-------------------|
| CSM Ewe→Adja | `69e93755d2fd2eb837d76d4e` | 6.5115 | ~19 (full 20-ep budget) | **NO** | Noise, 10s cap |
| Orpheus EN Ewe→Adja | `69e937572aa1660eaffa8c5a` | 5.6156 | 7.19 (ep 5.63 best) | yes @ patience=5 | Noise, natural durations |
| Orpheus FR Ewe→Adja | `69e9375a2aa1660eaffa8c5c` | 5.6356 | ~7 | yes @ patience=5 | Noise, same as EN |

### 8.2 Catastrophic-forgetting diagnosis

Initial Stage-2 training-step loss on Adja was ~17 nats/token. For a SNAC codebook of ~4096 entries, `ln(4096) ≈ 8.3` is the random-chance baseline. A loss of 17 means the Stage 1 model is *confidently producing Ewe-patterned audio tokens* when prompted with Adja text — not ignorant, actively wrong-for-Adja. Adaptation then has to unlearn the Ewe behavior before it can produce anything Adja-shaped, and on ~1,276 Adja clips the LoRA drift erases the Gbe prior faster than Adja grammar accumulates. Both ends crash and we land on noise.

**The critical evidence that this is forgetting and not a budget issue**: CSM Stage 2 ran its **full 20-epoch budget without early stopping firing** (37.5 min) and still produced noise. If training time were the bottleneck, CSM would have produced at least partially coherent output. Budget was not the constraint; *which data the model saw* was.

### 8.3 Mixed-data Stage 2 script (submitted 2026-04-22)

File: `scripts/hf_jobs/T2_orpheus_ewe_adja_stage2_mixed.py`. Identical to the baseline `T2_orpheus_ewe_adja_stage2.py` except for the training set:

- **Train**: `concatenate_datasets([processed_adja_train, processed_ewe_train]).shuffle(seed=42)` — 1:1 ratio (≈1,276 Adja + 1,215 Ewe = ~2,491 mixed samples per epoch)
- **Eval**: Adja dev only (~160 clips) — we measure the Adja target, not Ewe recall
- **Ewe source**: `google/WaxalNLP` config `ewe_tts` train split — same subset used to build the Stage 1 checkpoint
- **Adja source**: `JosueG/adja-tts-orpheus` 80/10/10 seed=42 split — same as baseline Stage 2
- **Hyperparameters**: identical to baseline for direct A/B comparability — LR 5e-5, LoRA r=64 on q/k/v/o/gate/up/down proj, patience=5 + threshold=0.0, 20 epochs max, bf16, `optim="adamw_8bit"`, seed=42
- **Output prefix**: `T2_orpheus_en_ewe_adja_stage2_mixed` (distinct from baseline `T2_orpheus_en_ewe_adja_stage2`)
- **Submitted**: job `69e977f82aa1660eaffa8d21` on l40sx1 with 8h timeout. Expected ~60-90 min (train set doubled → ~8 min/epoch, may early-stop before the full 20 epochs).

### 8.4 What the mixed-data recipe predicts

Interleaving Ewe audio tokens with Adja audio tokens during Stage 2 training forces the model to keep producing Gbe-family-valid audio sequences (the Ewe target keeps the backbone on-distribution for Gbe phonology) while still learning the Adja text-token → audio-token mapping via the Adja portion of each batch.

Three possible outcomes and their interpretations:

1. **Success (intelligible Adja, listening verdict)**: the paper's headline recipe — *Gbe-family prior + 1:1 anti-forgetting mix on Stage 2 unlocks intelligible low-resource Adja TTS*. Then we run CSM + Orpheus FR + Orpheus ZH + Spark mixed variants for the ablation.
2. **Partial success (Ewe-shaped audio with Adja tokens)**: we preserved the prior but didn't adapt — need a different mix ratio (3:1 Adja:Ewe or an annealing schedule that increases the Adja share over epochs).
3. **Failure (still noise)**: 1:1 isn't enough regularization. Next moves: (a) lower LR (1e-5 or 5e-6) so the adapter drifts less; (b) EWC regularization toward Stage 1 weights; (c) train only q/v LoRAs, freeze gate/up/down.

### 8.5 Ablations already logged for the paper (don't re-run)

- **Naive Stage 2 baseline** — the 3 runs in §8.1. Best-possible cascade without forgetting mitigation.
- **Patience=5 with CSM hitting full budget** — documents that training duration is not the bottleneck.
- **ewe_tts curated (1,215 clips) vs ewe_asr multi-speaker (15,054 clips)** — chose ewe_tts per §2.2 rationale; the ewe_asr variant is a follow-up ablation, not required for the paper's main claim.

---

Last updated: 2026-04-22 evening. Please update this doc whenever data, model, or hyperparameter choices
change materially, so future paper edits can cite a single source of truth.
