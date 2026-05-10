# Training Gotchas for Adja NMT

Learned during the neurosymbolic/ACL paper experiments (2025-2026).

## TTS — Always Range-Normalize HF Audio Arrays Before Codec/Processor Input

Do not assume `datasets.Audio` arrays or locally downloaded parquet audio arrays are already
normalized to `[-1, 1]`. The Adja TTS parquet audit on 2026-05-09 found PCM-scale float arrays in
`JosueG/adja-tts-orpheus`. Any TTS path that passes `example["audio"]["array"]` directly into a
processor or codec can silently train on invalid waveform magnitudes.

Affected historical paths: CSM direct/tokfix/Stage 2, Orpheus direct/tokfix/Stage 2/mixed Stage 2,
and SageMaker CF1/CF2/CF3. Spark was not affected because it already had explicit audio
normalization.

Guardrail: before CSM `processor.apply_chat_template(...)`, Orpheus SNAC encoding, Mimi/SNAC
preprocessing, or any custom WAV write, apply a waveform range guard:

- Leave peaks `<= 2.0` unchanged.
- Divide PCM-like peaks by `65536.0`.
- Peak-scale anything larger.
- Re-check after resampling.

The May 2026 reruns and job IDs are documented in
[`docs/tts-audio-range-normalization-reruns-2026-05-09.md`](../docs/tts-audio-range-normalization-reruns-2026-05-09.md).

## TTS — Codec vs Tokenizer Failure (Important Correction)

Early analysis blamed CSM (Mimi codec) and Orpheus (SNAC codec) failures on the codecs being "English-centric." This was wrong.

**Both Mimi and SNAC are waveform-level acoustic codecs — they are language-agnostic.** Tones (F0 contour), ATR vowels, nasals are all acoustic features that encode correctly. This is confirmed by: Orpheus `3b-zh` (Mandarin, 4 tones) works, which uses the exact same SNAC codec as the English model.

**The real bottleneck for CSM/Orpheus on Adja was Layer 1: the Llama 3.2 BPE text tokenizer.** Adja/Ewe diacritics (ɛ, ɔ, ŋ, ɖ, è, é, ɔ̀, ɛ́) are not in the Llama 3.2 vocabulary — they fall back to multi-byte sequences. The LM cannot learn phoneme→audio mapping from byte fragments with 1.7h of data.

**Diagnostic**: tokenize a sample Adja sentence with `tok.tokenize("ɛnyi wɛ dze è")`. If you see byte fragments instead of whole characters, fix the tokenizer before fine-tuning.

**Fix options (in order of complexity):**
1. `tokenizer.add_tokens([...missing_chars...])` + `model.resize_token_embeddings()` — fast, keeps base model
2. Switch to a tokenizer with native coverage (Qwen2 BPE, NLLB SentencePiece)

Spark TTS succeeded because it uses Qwen2 tokenizer (native Adja char support) + XLSR-53 BiCodec (multilingual semantic tokens). It got all three layers right; CSM/Orpheus only failed on one.

## Model / Tokenizer

- **`aj_Latn` is a custom token** — must call `fix_tokenizer()` + `model.resize_token_embeddings()` every time the model/tokenizer is loaded. Forgetting this = silent garbage output.
- **`as_target_tokenizer()` is deprecated** in transformers 4.44+ but still works. Pin `transformers==4.44.2`. If it breaks, replace with: set `tokenizer.src_lang = TGT_LANG`, tokenize, then restore `src_lang`.
- **Initializing `aj_Latn` embeddings from a related language** (Ewe/Fon) worked better than random init.

## Training Loop

- **`Seq2SeqTrainer` is fragile across transformers versions.** Custom training loop with Adafactor + manual step loop is more reliable. Don't go back to Trainer.
- **Adafactor optimizer** with constant-with-warmup schedule worked well for low-resource fine-tuning.
- **Early stopping on validation chrF** (not loss) — chrF correlates better with translation quality than loss in low-resource.

## Data

- **Group-aware train/val/test splits are critical** — `groupby(base_sentence_id)` prevents minimal-pair leakage across splits. Without this, metrics are inflated.
- **Always run decontamination** after preparing baseline/ablation splits — we found 3.8-29.3% contamination in early splits.
- **Stratified sampling** in dataset construction: `groupby(['pronoun', 'verb']).apply(sample)` ensures balanced coverage.
