# TTS Models to Try for Adja

Written 2026-04-18 after T1-CSM baseline plateau (best dev loss 6.488, audio still noise).

## The constraint

- **~1.7 hours** of Adja audio (1277 utterances, avg ~5 s)
- **Gbe-family, tonal language** with ɛ, ɔ, ŋ, ɖ and tone marks (é, è)
- **No chance of 10x more data** in a useful timeframe — have to be creative
- Primary target: paper-quality demo + usable generations, not production service

## Tier 1 — Most likely to move the needle for Adja

### 1. MMS-TTS-Ewe → fine-tune on Adja  ⭐ top recommendation

**Why**: Meta's MMS-TTS was trained on 1100+ languages including **Ewe**. Ewe and Adja are both Gbe-family — share most phonemes, similar tonal system, overlapping vocabulary. Starting from a model that already speaks Ewe skips the "learn new language from scratch" problem that CSM/Orpheus face.

Ewe is arguably the single closest available pretraining prior for Adja TTS anywhere in the open ecosystem.

**What to run**:
- Model: `facebook/mms-tts-ewe` (VITS architecture)
- Fine-tune on Adja dataset directly (VITS supports this out of the box)
- Phoneme input via MMS tokenizer should handle Gbe phonemes cleanly

**Key refs**:
- https://huggingface.co/facebook/mms-tts-ewe
- https://arxiv.org/abs/2305.13516 (MMS paper)

**Risks**:
- Fewer moving parts, but also fewer levers — VITS is harder to fine-tune than LLM-TTS
- Has to accept the MMS tokenizer's phoneme set

### 2. IMS-Toucan

**Why**: Purpose-built for low-resource tonal TTS. Uses language embeddings from Glottolog; explicitly supports cross-lingual transfer with ~1h of target data. Published baselines on African languages.

**What to run**:
- Repo: https://github.com/DigitalPhonetics/IMS-Toucan
- Start from their pretrained multilingual checkpoint
- Fine-tune with Adja audio + text, using Glottolog ID for Adja (`ajgb1240` or parent `gbee1241`)

**Key refs**:
- https://arxiv.org/abs/2206.12229 (IMS-Toucan paper, updated 2024)
- Has a dedicated low-resource recipe

**Risks**:
- Smaller research community, less plug-and-play than HF ecosystem
- Needs Glottolog ID selection — pick wrong parent and transfer suffers

## Tier 2 — Strong multilingual, proven on small data

### 3. XTTS-v2 (Coqui)

**Why**: 16-language multilingual TTS designed for voice cloning with 1-2 hours of target data. Cross-lingual transfer is a first-class feature. Community has fine-tuned it for dozens of out-of-distribution languages.

**What to run**:
- `coqui/XTTS-v2` on HuggingFace
- Fine-tune on Adja dataset
- Use Adja reference audio for speaker conditioning at inference

**Key refs**:
- https://huggingface.co/coqui/XTTS-v2
- https://arxiv.org/abs/2406.04904

**Risks**:
- Coqui shut down in 2024; repo is in maintenance mode. Weights still work but tooling less supported.

### 4. F5-TTS / E2-TTS

**Why**: Flow-matching TTS from 2024 — newer generation beyond autoregressive codec LMs. F5 reported strong low-resource results. Architecturally simpler than VALL-E / CSM.

**What to run**:
- `SWivid/F5-TTS` on HuggingFace
- Fine-tune on Adja dataset
- Trainable with smaller effective batch than LLM-TTS

**Key refs**:
- F5-TTS repo: https://github.com/SWivid/F5-TTS
- F5 paper: https://arxiv.org/abs/2410.06885 (2024)

**Risks**:
- Very new, less community experience with low-resource fine-tune
- May still be English-dominant in pretraining

## Tier 3 — Exhaust current CSM capacity first

### 5. Full fine-tune Sesame CSM  (in progress as of 2026-04-18)

**Why**: LoRA r=32 plateaued at 6.488. Before concluding the base model / data is the blocker, see what happens when we unfreeze all 1.66B params. VRAM cost: ~10x, training time: ~3x. L40S 48GB should still fit at bf16 with smaller batch.

**What to run**:
- Existing `scripts/hf_jobs/T1_sesame_csm_finetune.py` with a `--full-finetune` flag (to be added)
- Same early stopping protocol

**Signal we're looking for**:
- If full FT eval loss drops to ~4-5 → capacity was the blocker, worth investing in bigger LoRA next
- If full FT still plateaus near 6.5 → data is the actual blocker, pivot to Tier 1 models

### 6. LoRA r=128 CSM  (intermediate)

Cheaper sibling of #5. If full FT helps and is expensive, r=128 is the middle ground.

## Tier 4 — Already tried (by user or in-repo)

### Orpheus TTS (Canopy Labs)
- **Status: exhausted as of 2026-04-18.** 5-variant sweep (LoRA r=32/r=64/r=128 on L40S + full fine-tune on A100 + French base LoRA r=64 on L40S) all completed and audibly confirmed unintelligible — user listened to all 25 generated clips, no Adja words or phonology in any variant. Same perceptual outcome as T1 CSM despite a ~1.0 nat improvement in dev loss (best: 5.4242 full-FT vs T1's 6.488).
- Capacity ordering was clean and monotonic, full-FT diverged catastrophically after ep 1.88 (small-data classic). French base lost to English base at matched LoRA rank (5.5058 vs 5.4823). French-prior hypothesis falsified.
- Root cause: English-dominant pretraining (Llama BPE tokenizer fragments Adja ɛ/ɔ/ŋ/ɖ into byte sequences; 1.7h is ~200x below Reddit success cases). Data is the bottleneck, not capacity or backbone size.
- Do not invest further in Orpheus or any other English-prior LLM-TTS at 1.7h of Adja data.
- Full results in [experiments/tts/T2_orpheus_finetune/README.md](../experiments/tts/T2_orpheus_finetune/README.md).

### Spark TTS
- Repo script exists (`scripts/hf_jobs/T3_spark_tts_finetune.py`) but not run on Adja yet
- Smaller (0.5B) — compute cheap
- Architecturally similar issues to CSM (LLM-style audio codec modeling)

## Data-side levers (orthogonal to model choice)

Given the 1.7h constraint, these can multiply effective dataset size without collecting more audio:

### A. Audio augmentation (cheap)
- **Speed perturbation** (±10%): 3x data
- **Pitch shift** (±2 semitones): 3x data
- **SpecAugment on Mimi codec latents**: regularizer, reduces overfitting

Implementation: `torchaudio.transforms.SpeedPerturbation`, `librosa.effects.pitch_shift`. Apply at preprocess time, save augmented dataset to disk, train on the expanded set. Should reduce eval_loss gap.

### B. Cross-lingual bootstrapping (medium)
- **Train on Ewe first, then fine-tune on Adja**
- Requires Ewe TTS corpus. Common Voice has some Ewe; ALFFA/Bible TTS has more. 5-10h Ewe + 1.7h Adja >> 1.7h Adja alone.

### C. G2P → phoneme input (medium)
- Replace raw text input with IPA phonemes from a Gbe G2P model or a rule-based Adja G2P
- Avoids Llama tokenizer fragmentation of ɛ/ɔ/ŋ/ɖ
- Lets the model share phoneme embeddings across clip boundaries → more effective supervision per phoneme

### D. Synthetic scale-up (risky)
- Use a different TTS (XTTS-v2 zero-shot Adja) to generate "more" Adja audio, train on it
- Circular: we synthesize bad audio, model learns bad audio
- Only useful as a regularizer / warm-start, not as real data

## Decision matrix (my recommendation)

> **Update 2026-04-18**: Orpheus 5-variant sweep completed. All variants audibly confirmed unintelligible (user-confirmed). Data bottleneck confirmed at 1.7h for English-prior LLM-TTS. **Prioritize T6 MMS-Ewe results over any further English-prior LLM-TTS experiments.** Orpheus confirmed not the blocker's solution.

Current priority order:

1. **Now**: await T6 MMS-TTS-Ewe results (already running, Job `69e390f1cd8c002f31dfe9d4`). This is the highest-leverage model-side experiment remaining — Gbe-family prior vs English-prior.
2. **Then**: IMS-Toucan if MMS-Ewe wasn't enough. Purpose-built for low-resource tonal languages.
3. **If all else fails**: data augmentation + G2P to buy more effective signal from existing 1.7h. Orthogonal to model choice — applies to all of the above.

## Related files

- `ideas/nllb-tokenizer-for-tts.md` — earlier research idea, tokenizer surgery on CSM
- `ideas/nllb-tokenizer-for-tts-reading-list.md` — 25-paper reading list across TTS / low-resource / tokenizer
- `results/tts-comparison.md` — leaderboard to update after each new experiment
