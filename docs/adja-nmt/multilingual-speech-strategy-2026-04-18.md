# Multilingual Speech Strategy for Adja

Written 2026-04-18 to decide what to run next for both TTS and ASR on the
`JosueG/adja-tts-orpheus` dataset (~1.95h, 1,597 utterances, NFC-normalized).

This is the practical ranking, not a generic leaderboard. The goal is to pick
the highest-leverage paths for **low-resource Adja** specifically.

## TTS: what is most promising for Adja?

### Rank 1 — IMS-Toucan

**Why it is the best conceptual fit**

- Built for massively multilingual, low-resource TTS.
- Explicit language embeddings and ISO-639-3 handling are a better match for
  Adja than English-centric codec LMs.
- Its text frontend can fall back to Transphone for unsupported languages, so
  Adja (`ajg`) is not blocked at the tokenizer level.
- The architecture is designed around phonological generalization rather than
  only memorizing text-token-to-codec-token mappings.

**Why it matters for Adja**

- Adja is tonal and low-resource.
- The existing English-centric TTS paths in this repo (CSM, Orpheus, Spark)
  mostly learn "some audio behavior" but not intelligible Adja phonology.
- IMS-Toucan is the cleanest test of whether explicit multilingual phonology
  and language embeddings beat raw LLM capacity under the same data budget.

**Primary risk**

- Operational complexity is higher than the Hugging Face-native stacks.
- The first smoke run is partly a tooling validation: repo install, text
  frontend, corpus preparation, and checkpoint resume path.

**References**

- IMS-Toucan repo: https://github.com/DigitalPhonetics/IMS-Toucan
- Lux et al., massively multilingual Toucan work
- Repo language inventory: `Utility/language_list.md`

### Rank 2 — MMS-TTS-Ewe -> Adja

**Why it is still a top-tier bet**

- Ewe is the closest open-source TTS prior to Adja in the current stack.
- This repo already has a live T6 track for it.
- It avoids English-only tokenizer and phonology mismatch.

**What to learn from it**

- If Ewe transfer works and English-centric models do not, that is strong
  evidence that **language-family prior matters more than raw model size**.

**References**

- MMS: https://jmlr.org/papers/v25/23-1318.html
- `experiments/tts/T6_mms_tts_ewe_finetune/README.md`

### Rank 3 — XTTS-v2

**Why it deserves a serious run**

- Explicitly positioned for cross-lingual voice cloning with very little target
  speech.
- Community usage around 1-2h is much more common than for VoxCPM or Toucan.
- Practical fine-tuning interface is simpler than IMS-Toucan.

**Why it is not rank 1**

- XTTS-v2 only fine-tunes the GPT encoder in the public recipe.
- Adja is outside the officially supported language list, so this is still an
  out-of-distribution adaptation.
- It is a practical small-data baseline, not the most linguistically
  principled solution.

**References**

- XTTS-v2 model card: https://huggingface.co/coqui/XTTS-v2
- Coqui TTS XTTS training docs

### Rank 4 — VoxCPM2

**Why it is attractive**

- Latest multilingual OpenBMB release, trained on very large multilingual
  speech corpora.
- Tokenizer-free and diffusion-autoregressive, so it tests a genuinely
  different representation family from XTTS and CSM.
- Supports LoRA and full SFT in the official training docs.

**Why it is not above IMS-Toucan / XTTS-v2**

- The repo is strongest as a large multilingual capability test, not a proven
  low-resource tonal adaptation recipe.
- The most accessible open fine-tune examples are still new.
- Model download/runtime cost is higher than XTTS-v2 and F5.

**Practical positioning**

- Best "large multilingual capacity" ablation after the more principled
  low-resource paths.
- Useful if IMS-Toucan and XTTS-v2 still fail to become intelligible.

**References**

- VoxCPM repo: https://github.com/OpenBMB/VoxCPM
- Fine-tune guide: `docs/finetune.md` in the upstream repo

### Rank 5 — F5-TTS / E2-TTS

**Why they are worth trying**

- Flow-matching is a real architectural contrast with codec-LM TTS.
- The upstream repo exposes a clean fine-tune CLI and supports custom vocab
  extension for unseen symbols.
- F5 and E2 let us test whether the bottleneck is the autoregressive codec-LM
  paradigm itself.

**Why they are lower priority**

- The released checkpoints are English/Chinese-oriented.
- Adja requires custom vocabulary handling and checkpoint embedding expansion.
- This makes them excellent research ablations, but not the safest
  first-success path.

**References**

- F5-TTS repo: https://github.com/SWivid/F5-TTS
- F5-TTS paper: https://arxiv.org/abs/2410.06885
- E2-TTS paper: https://arxiv.org/abs/2406.18009

## TTS operational ranking for this repo

If the question is "what should we run first, second, third in this codebase?"
the answer is:

1. **T6 MMS-TTS-Ewe** (already running)
2. **T8 IMS-Toucan**
3. **T9 XTTS-v2**
4. **T7 VoxCPM**
5. **T10/T11 F5-TTS / E2-TTS**

This ordering balances:

- linguistic prior for Adja
- likelihood of working with ~2h audio
- implementation risk
- compute cost

## ASR: what is best for low-resource multilingual Adja?

### What has already become clear in this repo

- Encoder-decoder ASR is winning over CTC in the current data regime.
- Related-language transfer (Ewe -> Adja) helps.
- A character LM helps CTC models, but it does not fully close the gap to the
  best Whisper-family runs.

### Best next ASR bets

#### Rank 1 — Continue Whisper-Ewe (E4) + LM

- This is the best measured path in the repo's current evidence base.
- The Ewe prior is directly relevant to Adja.
- Continuing from the best checkpoint is lower risk than restarting another
  Whisper run from scratch.

#### Rank 2 — Whisper-large-v3 (and, if possible, Ewe-adapted large Whisper)

- More capacity should help if we preserve the working decoder setup.
- The main constraint is compute, not architecture uncertainty.

#### Rank 3 — AfriHuBERT / African-pretrained SSL encoders

- Better prior than English-only wav2vec2 or generic XLS-R for African tonal
  languages.
- Strongest non-Whisper comparison if we want an encoder-only ASR baseline with
  a more relevant pretraining distribution.

#### Rank 4 — Omnilingual ASR zero-shot / HPC path

- Valuable as a frontier comparison.
- Especially useful if it handles unseen or nearly unseen languages well.
- The repo already documents the CUDA/runtime blocker on HF Jobs, so this is an
  HPC experiment, not a cloud-first one.

#### Rank 5 — More CTC ablations only after the above

- CTC is still useful for controlled comparisons and LM experiments.
- But the repo's current results say it should not consume the majority of the
  next cycle.

## ASR improvements that matter most

### Highest-leverage

1. **Resume E4 from the best known checkpoint**
2. **Keep LM shallow-fusion / beam-rescoring in the loop**
3. **Move to larger Whisper capacity before inventing new loss functions**
4. **Use related-language priors whenever possible**

### Good follow-ups after that

1. AfriHuBERT or another African-pretrained encoder baseline
2. Better normalized WER reporting
3. Per-character error analysis for tone-bearing and rare Adja symbols
4. Targeted augmentation for rare orthographic patterns

### Lower priority than they sound

- new CTC loss variants before stabilizing the best Whisper path
- broad architecture churn without better priors
- tokenizer surgery that is not clearly tied to a measured failure mode

## Shared principles for both TTS and ASR

### 1. Related-language priors beat generic multilingual claims

For Adja, **Ewe/Fon/Gbe-family transfer** is more important than generic
"supports many languages" marketing.

### 2. Unicode handling is non-negotiable

Always preserve and normalize:

- `ɛ`
- `ɔ`
- `ŋ`
- `ɖ`
- tone marks

### 3. Smoke jobs must prove more than imports

For Hugging Face submissions, the first useful smoke signal is:

- dataset load succeeds
- preprocessing completes
- at least one training step actually runs

Just downloading weights is not enough.

### 4. Record exact failure signatures quickly

The current repo is already strong at this. Keep updating:

- `experiments/tts/attempt-log.md`
- `experiments/registry.md`
- `results/run-ledger.md`
- `results/tts-comparison.md`

## Recommended next-week order

1. Monitor active T6
2. Submit IMS-Toucan smoke
3. Submit XTTS-v2 smoke
4. Submit VoxCPM smoke
5. Submit F5-TTS smoke
6. Submit E2-TTS smoke
7. Keep ASR effort focused on E4 continuation + larger Whisper + African
   encoder comparisons

## References

- Whisper: https://cdn.openai.com/papers/whisper.pdf
- MMS: https://jmlr.org/papers/v25/23-1318.html
- SeamlessM4T v2 / related speech stack: https://arxiv.org/abs/2312.05187
- F5-TTS: https://aclanthology.org/2025.acl-long.313/
- E2-TTS: https://arxiv.org/abs/2406.18009
- VoxCPM: https://github.com/OpenBMB/VoxCPM
- IMS-Toucan: https://github.com/DigitalPhonetics/IMS-Toucan
- XTTS-v2: https://huggingface.co/coqui/XTTS-v2
