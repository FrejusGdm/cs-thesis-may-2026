# T3: Spark TTS Fallback

Immediate Unsloth fallback if the canonical Sesame CSM path remains blocked after:

1. one clean CSM attempt
2. one short vanilla CSM diagnostic

## Canonical script

- `scripts/hf_jobs/T3_spark_tts_finetune.py`

## Why Spark is the fallback

Spark's Unsloth flow tokenizes audio into text-like semantic/global tokens before training.
That means a generic `SFTTrainer` flow is structurally correct for Spark in a way it is not for CSM.

## Runtime defaults

- Primary runtime: Google Colab T4
- Escalate to A100 only for VRAM/runtime pressure
- Keep `transformers==4.56.2`
- Keep float32 training
- Keep `use_gradient_checkpointing="unsloth"` unless a concrete Spark-specific error appears

## Success gate

The fallback is considered successful only if it produces:

- one short training run without tokenization shape errors
- at least one saved Adja waveform from `audio_tokenizer.detokenize(...)`
- metrics + generated sample metadata
