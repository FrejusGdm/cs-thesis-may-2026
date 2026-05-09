# Phase 1: Repository Forensics — Qwen3-TTS & Qwen3-ASR Deep Analysis

> **Note**: Your links point to Qwen3-ASR, but your goal is TTS fine-tuning for a native language.
> Both models come from the same Qwen3 speech family and share architectural DNA (Qwen3-Omni foundation).
> This document covers **both** systems comprehensively so you understand the full ecosystem.

---

## 1. Architecture Deep Dive

### Qwen3-TTS Architecture

Qwen3-TTS uses a **discrete multi-codebook language model** architecture — this is fundamentally different from
the Tacotron/FastSpeech lineage (autoregressive mel-prediction) or the newer diffusion/flow-matching approaches
(like Voicebox, NaturalSpeech 3). Here's how the pieces fit together:

#### Component 1: Qwen3-TTS-Tokenizer-12Hz (Speech Codec)

This is the bridge between continuous audio waveforms and discrete tokens the LM can process.

| Parameter              | Value                          |
|------------------------|--------------------------------|
| Frame rate             | 12.5 Hz (80ms per frame)       |
| Total codebook layers  | **16**                         |
| Semantic codebook      | **1 layer** (WavLM-guided)     |
| Acoustic RVQ layers    | **15 layers** (residual VQ)    |
| Encoder/Decoder        | Fully causal ConvNet           |
| Training method        | GAN + multi-scale mel loss     |
| Streaming latency      | ~97ms first-packet             |

**How it works:**
1. The **encoder** takes raw waveform → extracts features at 12.5 Hz
2. **Semantic quantizer** (layer 0): Trained with WavLM teacher distillation. Captures *what* is being said
3. **Acoustic RVQ** (layers 1-15): Each layer quantizes the *residual* error from all previous layers.
   Captures *how* it sounds — timbre, prosody, room acoustics, breathing
4. The **decoder** takes all 16 codebook indices → reconstructs the waveform

**Why this matters for fine-tuning:** When you fine-tune, the tokenizer is **frozen**. You're training the
LM to predict better codebook sequences for your language. The tokenizer's reconstruction quality sets
a ceiling on output quality.

#### Component 2: The Backbone LM (Transformer)

The main model that generates speech tokens autoregressively:

| Variant    | Parameters | Size on Disk |
|------------|-----------|-------------|
| 1.7B-Base  | 1.7B      | ~4.54 GB    |
| 0.6B-Base  | 0.6B      | ~2.52 GB    |

**Hierarchical prediction scheme:**
1. **Backbone** ingests aggregated features from all codebook layers → predicts codebook 0 (semantic)
2. **MTP (Multi-Token Prediction) module** → generates codebooks 1-15 (acoustic) in a single forward pass

This is the key architectural innovation: traditional LM+DiT approaches predict semantic tokens first,
then use a separate diffusion model to generate acoustic details. Qwen3-TTS does it all in one model,
avoiding cascading errors and information bottlenecks.

#### Component 3: Speaker Encoder

Extracts speaker identity from a reference mel-spectrogram:
- Input: Mel-spectrogram of reference audio (24kHz, 128 mel bins, 1024 FFT, 256 hop)
- Output: Speaker embedding vector
- Used during inference to condition generation on a target voice

**For fine-tuning:** The speaker encoder weights are **excluded** from saved checkpoints (see `sft_12hz.py`).
This means fine-tuning adapts the LM's generation patterns, not the speaker representation.

### Qwen3-ASR Architecture

Qwen3-ASR is built on Qwen3-Omni and uses a **thinker** architecture (encoder-decoder with audio features):

| Component              | Details                              |
|------------------------|--------------------------------------|
| Foundation             | Qwen3-Omni base model               |
| Audio encoder          | Whisper-style feature extraction     |
| Audio sample rate      | 16kHz mono                           |
| Architecture pattern   | `model.thinker.forward()` dispatch   |
| Language support       | 30 languages + 22 Chinese dialects   |
| Max audio length       | 5 minutes per inference call         |

The ASR model uses a `patch_outer_forward` pattern (see `qwen3_asr_sft.py:37-60`) that routes the
outer model's forward call to the inner `thinker` module. This is necessary because the HuggingFace
Trainer expects a standard `forward()` signature with `labels` parameter.

---

## 2. Training Pipeline Analysis

### TTS Fine-tuning Pipeline (from `finetuning/sft_12hz.py`)

```
Raw Data (JSONL) → prepare_data.py → Tokenized Data (JSONL) → sft_12hz.py → Fine-tuned Model
```

**Stage 1: Data Preparation** (`prepare_data.py`)
- Loads `Qwen3TTSTokenizer` from pretrained checkpoint
- Processes audio files in batches of 32
- Encodes each audio → 16 codebook index sequences
- Saves tokenized codes alongside original text/audio paths

**Stage 2: Training** (`sft_12hz.py`)
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Gradient accumulation**: 4 steps
- **Precision**: bfloat16 (with Flash Attention 2)
- **Loss**: Combined cross-entropy over predicted codebook tokens
  - The model predicts codebook-0 tokens via backbone
  - MTP module loss over codebooks 1-15
  - These losses are combined internally by the model's forward pass

**Training loop pseudocode:**
```python
for batch in dataloader:
    text_embeds = batch["text_embeddings"]      # Tokenized text
    codec_embeds = batch["codec_embeddings"]    # From prepare_data.py
    ref_mels = batch["ref_mels"]                # Speaker reference
    attention_masks = batch["attention_masks"]

    speaker_embed = model.speaker_encoder(ref_mels)
    input_embeds = model.combine_embeddings(text_embeds, codec_embeds, speaker_embed)
    loss = model.forward(input_embeds, attention_mask)

    loss.backward()
    optimizer.step()
```

**Checkpoint saving:**
- Saves after each epoch: `output/checkpoint-epoch-{N}/`
- Excludes speaker encoder weights
- Saves model config with custom speaker metadata

### ASR Fine-tuning Pipeline (from `finetuning/qwen3_asr_sft.py`)

Uses HuggingFace `Trainer` with custom data collator:

**Data Collator** (`DataCollatorForQwen3ASRFinetuning`):
1. Loads audio at 16kHz mono via librosa
2. Constructs full text = `prefix_text + target + eos_token`
3. Tokenizes with padding
4. Creates labels: `-100` for prefix tokens (don't compute loss on input), actual token IDs for target

**Key architectural decisions:**
- `CastFloatInputsTrainer`: Custom trainer that casts float inputs to model dtype (bf16/fp16)
- `MakeEveryCheckpointInferableCallback`: Copies tokenizer/config files to each checkpoint so they're self-contained
- Resume support: Both explicit checkpoint path and auto-detect latest

---

## 3. Official Fine-tuning Support

### TTS: YES — Official scripts in `finetuning/` directory

| File              | Purpose                                    |
|-------------------|--------------------------------------------|
| `README.md`       | Step-by-step fine-tuning guide             |
| `prepare_data.py` | Audio → codebook tokenization              |
| `dataset.py`      | `TTSDataset` class with collation logic    |
| `sft_12hz.py`     | Full SFT training script                   |

**Limitations noted:**
- Single-speaker fine-tuning only (current release)
- Multi-speaker support planned for future release

### ASR: YES — Official scripts in `finetuning/` directory

| File                | Purpose                                  |
|---------------------|------------------------------------------|
| `README.md`         | Fine-tuning guide with examples          |
| `qwen3_asr_sft.py`  | Full SFT script using HF Trainer        |

**Features:**
- Single and multi-GPU (torchrun) support
- Resume from checkpoint
- Evaluation during training

---

## 4. Data Format Requirements

### TTS Data Format

**Input JSONL** (before tokenization):
```json
{"audio": "./data/utt0001.wav", "text": "The transcript of the audio.", "ref_audio": "./data/ref_speaker.wav"}
```

| Field       | Required | Description                                    |
|-------------|----------|------------------------------------------------|
| `audio`     | Yes      | Path to training utterance WAV file             |
| `text`      | Yes      | Exact transcript of the audio                   |
| `ref_audio` | Yes      | Path to reference speaker audio (use same file for all samples) |

**Audio specifications:**
- Format: WAV (uncompressed PCM)
- Sample rate: 24kHz (the tokenizer resamples internally, but native 24kHz is best)
- Channels: Mono
- Bit depth: 16-bit or 32-bit float
- Duration: No hard limit, but practical range is 1-30 seconds per utterance

**Mel-spectrogram parameters** (from `dataset.py:extract_mels()`):
- Sample rate: 24,000 Hz
- FFT size: 1024
- Mel bins: 128
- Hop size: 256 (gives ~93.75 frames/second)

**After tokenization** (`prepare_data.py` output):
```json
{"audio": "./data/utt0001.wav", "text": "...", "ref_audio": "./data/ref.wav", "codes": [[1,42,7,...], [9,3,88,...], ...]}
```
The `codes` field contains 16 lists (one per codebook layer), each with `ceil(audio_duration * 12.5)` indices.

### ASR Data Format

**Input JSONL:**
```json
{"audio": "/data/wavs/utt0001.wav", "text": "language English<asr_text>This is a test."}
```

| Field   | Required | Description                                          |
|---------|----------|------------------------------------------------------|
| `audio` | Yes      | Path to WAV file                                     |
| `text`  | Yes      | Language-prefixed transcript (see format below)       |

**Language prefix format:**
- With language detection: `"language English<asr_text>The actual transcript"`
- Without language info: `"language None<asr_text>The actual transcript"`

**Audio specifications:**
- Format: WAV
- Sample rate: 16kHz (resampled by librosa in data collator)
- Channels: Mono
- Duration: Up to 5 minutes, but shorter is better for training

---

## 5. Compute Requirements

### TTS Fine-tuning

| Resource              | Minimum          | Recommended           |
|-----------------------|------------------|-----------------------|
| GPU VRAM (1.7B)       | ~16 GB (bf16)    | 24+ GB (A5000/A6000)  |
| GPU VRAM (0.6B)       | ~8 GB (bf16)     | 16 GB                 |
| System RAM            | 32 GB            | 64 GB                 |
| Storage               | 50 GB + dataset  | 200+ GB               |
| Precision             | bfloat16         | bfloat16 + FlashAttn2 |

**Estimated training time** (rough, depends on dataset size):
- 1000 utterances, 1.7B, single A100: ~1-2 hours per epoch
- 10000 utterances, 1.7B, single A100: ~8-12 hours per epoch

**Multi-GPU:** The TTS script uses `accelerate` but the current release is primarily designed for single-GPU.
You can adapt it for multi-GPU with PyTorch DDP or FSDP.

### ASR Fine-tuning

| Resource              | Minimum          | Recommended           |
|-----------------------|------------------|-----------------------|
| GPU VRAM (1.7B)       | ~20 GB (bf16)    | 40+ GB (A100)         |
| GPU VRAM (0.6B)       | ~10 GB (bf16)    | 24 GB                 |
| System RAM            | 32 GB            | 64 GB                 |

**Multi-GPU:** Native `torchrun` support:
```bash
torchrun --nproc_per_node=2 qwen3_asr_sft.py ...
```

---

## 6. Language Support Analysis

### TTS Languages (10 languages)
Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

**Dialect support:** Mandarin, Hokkien, Wu, Cantonese, Sichuanese, Beijing, Nanjing, Tianjin, Shaanxi

### ASR Languages (30 languages + 22 dialects)
Arabic, German, English, Spanish, French, Indonesian, Italian, Japanese, Korean, Dutch, Portuguese,
Russian, Swedish, Thai, Turkish, Vietnamese, Chinese, Cantonese, Danish, Finnish, Hindi, Hungarian,
Malay, Macedonian, Polish, Czech, Filipino, Persian, Greek, Romanian

### Adding a New Language via Fine-tuning

**Challenges:**
1. **Tokenizer coverage**: The speech tokenizer was trained on multi-language data. If your language has
   phonemes outside the training distribution, reconstruction quality may suffer. Test by encoding and
   decoding sample audio first.
2. **Text tokenizer**: The text side uses a BPE tokenizer. Check if your language's characters are in the
   vocabulary. If not, rare characters will be split into byte-level tokens (still works, but less efficient).
3. **Phonological distance**: Languages phonologically close to the supported 10 will fine-tune more easily.
   For example, adding Portuguese (supported) vs. adding Xhosa (clicks not in training data) are very
   different challenges.
4. **Data quantity**: For a language with similar phonology, 1-5 hours of paired audio+text may suffice.
   For a phonologically distant language, expect to need 10-50+ hours.

**Recommendation:** Start with the 0.6B-Base model for experimentation, graduate to 1.7B once you've
validated your pipeline works end-to-end.

---

## Source Repositories

- **Qwen3-TTS**: https://github.com/QwenLM/Qwen3-TTS
- **Qwen3-ASR**: https://github.com/QwenLM/Qwen3-ASR
- **Technical Report**: https://arxiv.org/abs/2601.15621
- **HuggingFace Collection**: https://huggingface.co/Qwen
