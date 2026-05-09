# T6: MMS-TTS-Ewe fine-tune for Adja

Created 2026-04-18.

Status: **scaffolded 2026-04-18, not yet run**. This is the first T-series experiment that starts from a model already trained on a Gbe-family language (Ewe) rather than English-heavy priors.

## Hypothesis

Starting from `facebook/mms-tts-ewe` should beat starting from English-centric TTS models (CSM, Orpheus, Spark) on 1.7 h of Adja audio. The reasoning:

- **Adja and Ewe are both Gbe-family Niger-Congo tonal languages.** They share a large fraction of phoneme inventory — including ɛ, ɔ, ŋ, ɖ and an analogous tone system — and have strong lexical/phonotactic overlap.
- **Starting from a model that already outputs Ewe-family speech is structurally worth more than training English priors on 1.7 h.** T1-T3 showed that 1.7 h is 3-5x below the data budget required to re-home an English-centric TTS to a tonal Gbe language. If we start from Ewe we skip that re-homing step.
- Empirical rule of thumb from prior low-resource TTS work: cross-lingual transfer from a same-family language is **~5-10x more efficient** per hour of target-language audio than transfer from a distant-family language.

## Why character-level tokenizer is our bet

MMS-TTS uses a **character-level tokenizer with no G2P step.** That is the core bet of T6:

- No grapheme-to-phoneme pipeline has to be built for Adja (there is no public Adja G2P).
- ɛ, ɔ, ŋ, ɖ and tone-marked vowels (é, è, ê, etc.) feed directly into the tokenizer as raw characters after NFC normalization.
- The VITS backbone will learn the Adja character → phoneme → audio mapping directly during fine-tuning, starting from a checkpoint that already does this for Ewe characters.

If the character vocabulary of `mms-tts-ewe` is missing any Adja character, the ylacombe recipe's vocab-extension step handles that (new rows get added to the text embedding).

## What is canonical

| Thing | Location |
|---|---|
| Vendor training repo | https://github.com/ylacombe/finetune-hf-vits |
| HF base model | https://huggingface.co/facebook/mms-tts-ewe |
| Adja dataset | `JosueG/adja-tts-orpheus` (private, ~1597 utterances, ~1.7 h) |
| Canonical script | `scripts/hf_jobs/T6_mms_tts_ewe_finetune.py` |
| Config file | `experiments/tts/T6_mms_tts_ewe_finetune/conf/config.yaml` |
| Results destination | `JosueG/adja-tts-results/T6_mms_ewe_20ep_2026-04-18/` |

## License note

`facebook/mms-tts-ewe` is released under **CC-BY-NC 4.0**. That is fine for research publication and internal demos, **but not for commercial deployment.** Flag this clearly if T6 becomes the recommended production path.

## Assumptions (2026-04-18)

The experiment is built on these assumptions. If any are wrong, the plan changes.

1. **Character tokenizer is sufficient for Adja.** No Adja-specific G2P is needed — the ylacombe recipe can extend the tokenizer vocab if any Adja character is missing from the Ewe base tokenizer.
2. **`ylacombe/finetune-hf-vits` works on `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel`.** Same container as T1 (success), T3 (success). No exotic CUDA/torch pins expected.
3. **1.7 h of Adja audio is sufficient for same-family transfer.** This is the core bet. If it fails, fall back to augmenting Adja audio with Ewe audio for joint training.
4. **MMS-TTS-Ewe's 16 kHz sample rate is acceptable.** Lower than CSM (24 kHz) / Orpheus (24 kHz), but MMS-TTS was trained at 16 kHz and re-sampling the output upward is cheap at inference time. For research MOS comparisons this is fine.
5. **CC-BY-NC 4.0 is acceptable for research use.** Confirmed above.
6. **ylacombe's generator-loss + discriminator-loss GAN trainer is load-bearing.** Vanilla `transformers.Trainer` will not produce competitive VITS quality — VITS needs the adversarial loss.
7. **L40S 48 GB is enough.** VITS-base is ~83 M params, far smaller than CSM 1 B. 48 GB is plenty with batch 4 × grad-accum 4.
8. **HF_TOKEN has read access to `JosueG/adja-tts-orpheus` and write access to `JosueG/adja-tts-results`.** Same token as T1/T3.

## Dry-run success gate

Before a long run, the canonical script must clear all of these:

1. `ylacombe/finetune-hf-vits` clones and installs cleanly (no package resolver explosion).
2. `facebook/mms-tts-ewe` loads via `transformers.VitsModel.from_pretrained(...)`.
3. Tokenizer accepts every Adja training string after NFC normalization without an `UNK` flood. If any Adja character is not in the Ewe vocab, the vocab-extension step runs without error.
4. Dataset loads, resamples to 16 kHz, and splits 80/10/10 with seed 42.
5. Training step 1 produces a finite generator loss and a finite discriminator loss (VITS-GAN smoke test).
6. First eval step produces a non-empty synthesized waveform (not all zeros, not NaN).
7. End-of-run `VitsModel.from_pretrained(<output_dir>).generate(...)` produces a non-empty wav for at least one held-out Adja test sentence.

## Hard-stop rule

If ylacombe's repo fails to clone OR fails to install OR its entry point crashes on step 1:

- **Fallback:** vanilla `transformers.VitsModel` + custom generator-only fine-tune (no discriminator). Will produce lower-quality audio but at least something runs. Flag clearly that this is a structural downgrade from the canonical VITS-GAN recipe.

## Tracking

After every T6 cycle, update all four docs:

- `experiments/registry.md` — registry row
- `experiments/tts/attempt-log.md` — operational facts (pins, hashes, crash signatures)
- `results/run-ledger.md` — chronological entry with audio samples + notes
- `results/tts-comparison.md` — rank in the leaderboard

## References

- MMS paper: https://jmlr.org/papers/v25/23-1318.html
- MMS-TTS-Ewe model card: https://huggingface.co/facebook/mms-tts-ewe
- VITS paper: https://arxiv.org/abs/2106.06103
- ylacombe/finetune-hf-vits: https://github.com/ylacombe/finetune-hf-vits
- Gbe languages overview: Capo, H. B. C. (1991). *A Comparative Phonology of Gbe.*
