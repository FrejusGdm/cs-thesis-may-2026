# T3 (Spark TTS) — why it worked where CSM/Orpheus didn't

Last updated: **2026-04-21**

## What we tried

Fine-tune Spark TTS (0.5B params, SparkAudio/Spark-TTS-0.5B) on the same 1.6k
Adja utterances that broke CSM and Orpheus. Single L40S job, 120 training
steps, ~1.4 min wall time.

## What we got

**Intelligible Adja.** Native-speaker listener (Josue): "so, so good. I can
understand words. It actually produces coherent words." Captured as the
headline TTS result on 2026-04-18.

## Why it worked — the three layers

See `../concepts/03-three-layer-tts.md` for the full mental model. Spark got
all three layers right:

| Layer | CSM | Orpheus | **Spark** |
|-------|-----|---------|-----------|
| Tokenizer | Llama 3.2 BPE ❌ | Llama 3.2 BPE ❌ | **Qwen2 BPE ✅** |
| LM backbone | Llama 3.2 1B | Llama 3.2 3B | **Qwen2 0.5B** |
| Audio codec | Mimi | SNAC | **BiCodec + XLSR-53 semantic** |

### Layer 1 — Tokenizer
Qwen2 was trained on a much more multilingual corpus than Llama 3.2,
including African languages. When you tokenize a sample Adja sentence with
Qwen2:

```python
['ɛnyi', 'wɛ', 'dze', 'è']     # clean, native tokens
```

vs. Llama 3.2:
```python
['Ġ', '<0xC9>', '<0x9B>', 'nyi', 'Ġ', ...]   # byte fragments
```

That's the primary difference. The LM sees clean characters and can learn
the phoneme→code mapping with 1.6k examples.

### Layer 2 — LM backbone
Qwen2 is also trained more multilingually than Llama 3.2. Not a huge factor
compared to Layer 1, but likely a positive.

### Layer 3 — Audio codec
Spark's BiCodec uses XLSR-53 for semantic tokens. XLSR-53 is a wav2vec 2.0
variant pretrained on 53 languages, many of them African. This gives the
semantic layer of the codec strong priors for Adja-like phonetics right out
of the gate.

## What this tells us

1. **The tokenizer was the main bottleneck for CSM/Orpheus.** Not the codec.
2. **Smaller is fine.** Spark (0.5B) beat Orpheus (3B) by producing
   intelligible output. Capacity wasn't the issue.
3. **Qwen2's multilingual pretraining matters.** This is why the `tokfix`
   experiments for CSM/Orpheus try to paper over Layer 1 — if it works, we
   recover those models. If not, we need bigger architectural changes.

## What's still unknown

- Is Spark's win primarily Layer 1 (tokenizer) or Layer 3 (XLSR-53 semantic)?
  Our tokfix experiments will partially isolate Layer 1. To isolate Layer 3,
  we'd need to swap codecs, which is more work.
- Does Spark's voice quality match CSM/Orpheus's quality after we fix their
  tokenizers? TBD — comparing ceilings is a later question.

## What we did next

1. Wrote `T3_spark_ewe_stage1.py` to train Spark on WaxalNLP Ewe first, then
   Adja — bigger training signal in a related language should help quality.
2. Wrote `T1_csm_tokfix.py` and `T2_orpheus_tokfix.py` to test the tokenizer
   hypothesis directly on CSM and Orpheus.
3. Ran Spark at 20 epochs with early stopping (vs 120 steps for the initial
   proof-of-concept). Need to listen to the output to judge if longer
   training improved or plateaued.

## Files

- Registry: `../../experiments/registry.md` row T3-spark
- Run ledger: `../../results/run-ledger.md` — 2026-04-18 T3-spark entries
- Script: `../../scripts/hf_jobs/T3_spark_tts_finetune.py` (original)
- Ewe bridge: `../../scripts/hf_jobs/T3_spark_ewe_stage1.py` + `_adja_stage2.py`
- Analysis: `../../research-paper-exploration/paper-one/why-spark-worked.md`
