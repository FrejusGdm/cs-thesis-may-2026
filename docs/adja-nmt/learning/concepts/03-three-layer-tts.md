# 03 — The three-layer mental model for LLM-based TTS

Last updated: **2026-04-21**
Prereq: `01-what-is-a-tokenizer.md`, `02-what-is-an-audio-codec.md`.

Every modern LLM-based TTS model (CSM, Orpheus, Spark, Llasa, F5-TTS, E2-TTS,
VoxCPM, Inworld…) reduces to this pipeline:

```
TEXT → [1. Text Tokenizer] → [2. LM Backbone] → [3. Audio Codec decoder] → AUDIO
```

When you debug a failure, *always* ask: which layer broke?

## Layer 1 — Text Tokenizer

- Input: string.
- Output: list of int IDs.
- Must cover every character in the target language.
- Failure mode: byte fragmentation (see concept 01).
- Fix: `tokenizer.add_tokens()` + `model.resize_token_embeddings()` OR swap
  tokenizer.

## Layer 2 — LM Backbone

- Input: interleaved text tokens + audio code tokens.
- Output: next audio code token at each step.
- This is where LoRA applies — you're teaching the LM a new text→audio mapping.
- Transfer wins: if a related language was in pretraining (e.g. Mandarin
  tones, French phonotactics), the LM starts closer to target.
- Capacity is rarely the bottleneck. We verified this on CSM: a full
  fine-tune of all 1.6B params did no better than LoRA r=32. The architecture
  had hit a representational ceiling, not a parameter-count ceiling.

## Layer 3 — Audio Codec

- Input: codec token sequence.
- Output: waveform.
- Language-agnostic (trained on waveforms, not phonemes).
- Usually frozen during fine-tuning.
- Failure mode: if the codec truly cannot represent an acoustic feature of
  your language, you lose signal. Test with a reconstruction experiment.

## Diagnostic decision tree

Bad output from fine-tuning?

```
Is the text reaching the LM as clean tokens?
├── No → Layer 1 is broken. Fix the tokenizer.
└── Yes → Does the codec reconstruct your language's audio cleanly?
    ├── No → Layer 3 is broken. Try a different codec (XLSR-53 BiCodec, DAC).
    └── Yes → Layer 2 is broken. Either
        • Wrong LM prior (swap base checkpoint, e.g. en→zh for tonal)
        • Not enough training data / steps
        • LoRA rank too low to learn new embedding vectors
```

## Why Spark won where CSM and Orpheus lost

| Layer | CSM | Orpheus | Spark |
|-------|-----|---------|-------|
| 1. Tokenizer | Llama 3.2 BPE ❌ | Llama 3.2 BPE ❌ | Qwen2 BPE ✅ |
| 2. LM | Llama 3.2 | Llama 3.2 | Qwen2 |
| 3. Codec | Mimi ✅ | SNAC ✅ | BiCodec (XLSR-53) ✅ |

Spark got all three right. CSM and Orpheus failed at Layer 1 only. Once we
fix Layer 1, both should work — that's the `tokfix` experiment hypothesis.

## Related

- `../experiments-explained/T1-csm-why-it-failed.md`
- `../experiments-explained/T3-spark-why-it-worked.md`
- `../../ideas/tts-models-guide.md` (longer version)
