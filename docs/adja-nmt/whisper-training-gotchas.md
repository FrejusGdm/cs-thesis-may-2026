# Whisper Fine-Tuning Gotchas (Adja ASR Session, 2026-04-17)

Record of fixes applied during the Adja ASR retrain so we don't
rediscover them later.

---

## 0. The real reason Whisper fine-tuning hallucinates infinite loops

This is the root cause of both the E4v2 (CER=76-90%, WER=120%+) and
E4v3 (CER=400-900%, loss→0.01) failures. Padding was a red herring —
3000-frame mel padding IS required by the encoder, but it does NOT
cause the repetitive-hallucination failure mode we kept seeing. This
does.

**Symptom (verified in E4v3 epoch 5 on run `69e27574cd8c002f31dfdf07`):**

```
REF: Enu maku enyi
HYP: Enu maku enyi. Ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka
     ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ɖeka ...
```

The model produces the correct transcript, then **keeps generating
forever** until `max_new_tokens` is hit. This is not a content failure —
it's a **stop-token failure**. Training loss collapses to ~0.01 while
CER explodes, because teacher-forced training sees very short targets
and the generator at inference has no learned "stop" signal.

### Why it happens

Whisper's tokenizer uses the **same token id for `pad_token` and
`eos_token`**:

```python
>>> from transformers import WhisperProcessor
>>> p = WhisperProcessor.from_pretrained("openai/whisper-small")
>>> p.tokenizer.pad_token_id
50257
>>> p.tokenizer.eos_token_id
50257
>>> p.tokenizer.pad_token
'<|endoftext|>'
```

This is intentional in Whisper — the pretraining used `<|endoftext|>`
both as the sequence-final marker and as padding for shorter chunks in
a 30-second batch.

When you tokenize a training label:

```python
>>> p.tokenizer("Enu maku enyi", padding=True).input_ids[0]
[50258, 50363,  36, 16241, 963, 84, 465, 8461,
 50257,  # <-- REAL EOS, the only target that teaches "stop here"
 50257, 50257, 50257, 50257, ...]  # <-- padding (same id, different role)
```

The widespread pattern in community Whisper fine-tuning code is to
replace "pad" with `-100` so that `CrossEntropyLoss(ignore_index=-100)`
skips those positions:

```python
# BROKEN: erases the real EOS along with the padding.
label_ids = label_ids.masked_fill(label_ids == pad_token_id, -100)
```

Because `pad_token_id == eos_token_id`, this substitutes `-100` for
**every** occurrence of 50257, including the single legitimate EOS
that ends the real transcript. The model never receives a gradient
signal telling it "after `yi`, emit `<|endoftext|>`." At inference, with
no forced prefix (`forced_decoder_ids=None`) and greedy/beam search, it
decodes the correct content, never emits EOS, and fills the remaining
`max_new_tokens` with whatever low-perplexity tokens cross-attention
produces — commonly a single-token n-gram loop.

### The fix

Use the tokenizer's `attention_mask` — which marks **real content
positions (including the terminal EOS) as 1** and post-EOS padding
positions as 0:

```python
labels = processor.tokenizer(
    texts, return_tensors="pt", padding=True, truncation=True
)
label_ids  = labels.input_ids
attn_mask  = labels.attention_mask   # <-- 1 for EOS, 0 only for padding

# Strip the added <|startoftranscript|> (model inserts it internally)
if (label_ids[:, 0] == sot_id).all():
    label_ids = label_ids[:, 1:]
    attn_mask = attn_mask[:, 1:]

# Mask ONLY the padded positions, preserving the real EOS.
label_ids = label_ids.masked_fill(attn_mask == 0, -100)
```

Empirical verification: `attention_mask[0]` for
`tokenize("Enu maku enyi")` is `[1,1,1,1,1,1,1,1, 1, 0,0,0,…]` — the
9th position (the real `<|endoftext|>`) has mask 1, and only the
subsequent padding copies have mask 0.

### Why this ever worked at all in the original E4 run

E4 (CER=24.9%) was submitted before any `transformers` version pin
existed in the script, so it resolved against whatever version was
latest at the time (likely ≥4.49, unpinned). It hit this bug too, but
the failure mode was milder because:

1. Whisper encoder is pretrained to expect 3000 frames → at inference,
   with batched dev samples, the decoder sometimes emits `<|endoftext|>`
   from prior distribution (pretrained weight for the EOS token is
   non-zero) before saturating `max_new_tokens`.
2. WER=73.09% at the end tells us E4 was still partially hallucinating
   (WER is bounded below by insertion count when HYP is longer than
   REF), but characters happened to align well enough for CER=24.9%.

In other words, E4 "worked" but was underperforming its potential. The
attention-mask fix should improve it further — expect both WER and CER
lower than the original E4 baseline.

### References

- **Whisper paper** — Radford et al., *Robust Speech Recognition via
  Large-Scale Weak Supervision*, OpenAI 2022.
  https://cdn.openai.com/papers/whisper.pdf
  (Section 2.3 "Multitask Format" describes the prefix/suffix token
  scheme and the `<|endoftext|>` dual role.)

- **Original sequence-to-sequence paper** — Sutskever, Vinyals & Le,
  *Sequence to Sequence Learning with Neural Networks*, NeurIPS 2014.
  https://arxiv.org/abs/1409.3215
  (Section 3.2 describes the role of the `<EOS>` symbol in terminating
  generation, which is the invariant this bug violates.)

- **Teacher forcing and exposure bias** — Ranzato, Chopra, Auli,
  Zaremba, *Sequence Level Training with Recurrent Neural Networks*,
  ICLR 2016. https://arxiv.org/abs/1511.06732
  (Explains why training-time loss can look great while inference-time
  generation degenerates — exactly what we saw with loss→0.01 and
  CER→400%.)

- **Whisper hallucination analysis** — Koenecke et al., *Careless
  Whisper: Speech-to-Text Hallucination Harms*, FAccT 2024.
  https://dl.acm.org/doi/10.1145/3630106.3658996
  (Documents that Whisper routinely hallucinates phantom text during
  silence or low-SNR regions; fine-tuning with broken EOS supervision
  amplifies this behavior.)

- **Correct Whisper fine-tuning pattern** — Hugging Face blog,
  *Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers*,
  Sanchit Gandhi, Nov 2022.
  https://huggingface.co/blog/fine-tune-whisper
  (The official `DataCollatorSpeechSeq2SeqWithPadding` masks with
  `attention_mask.ne(1)`, not with `labels == pad_token_id` — exactly
  the fix we applied. See the "Define a Data Collator" section.)

- **Whisper tokenizer source** — see
  `transformers/models/whisper/tokenization_whisper.py` in the
  `transformers` library for the default `add_special_tokens=True`
  behavior that appends `<|endoftext|>`.

---

## 1. The "3000 mel frames" error

**Symptom:**
```
ValueError: Whisper expects the mel input features to be of length 3000,
but found 1038. Make sure to pad the input mel features to 3000.
```

**Root cause:** Whisper's audio encoder is hard-coded to expect exactly
3000 frames (30 seconds at 16kHz with 10ms hop). In `transformers >= 4.49`
(and possibly 4.48.x late patches), the Whisper processor no longer
automatically pads to 3000 when called with `padding=True` — it pads to
the longest in the batch, which is usually less than 3000 for short clips.

**Wrong fix** (what I tried, what broke things):
- `padding="max_length"` works to produce 3000 frames. I originally
  attributed the hallucination loops to this choice — that was WRONG.
  The loops come from the EOS-masking bug (see section 0); `padding=
  "max_length"` may amplify the symptom slightly by making the decoder
  attend over silence-mel frames, but the primary cause is the label
  mask erasing the EOS token. Either `"max_length"` or the manual
  `torch.nn.functional.pad` approach below will work once the label
  mask is fixed.

**Right fix** (what actually works):
Pad manually in the collate function:
```python
input_features = processor(
    audios, sampling_rate=16000, return_tensors="pt", padding=True
).input_features

# Pad to exactly 3000 frames for all transformers versions
if input_features.shape[-1] < 3000:
    pad_width = 3000 - input_features.shape[-1]
    input_features = torch.nn.functional.pad(
        input_features, (0, pad_width), value=0.0
    )
elif input_features.shape[-1] > 3000:
    input_features = input_features[..., :3000]
```

This preserves the "short-batch padding" behavior of `padding=True` (the
label sequences aren't inflated to match 30s of silence) while satisfying
Whisper's hard requirement.

**File**: `scripts/hf_jobs/whisper_finetune.py` (around line 107)

---

## 2. Save BOTH model and processor (CTC)

**Symptom:** After training, trying to load the saved model for inference
fails with "no preprocessor_config.json" or "no ctc_vocab.json".

**Root cause:** We originally only called `model.save_pretrained()`.
For CTC models, the feature extractor is part of the acoustic pipeline
and must be saved too. For the custom CTC head to work, we also need
our char→index mapping.

**Fix in `scripts/hf_jobs/ctc_finetune.py` (around line 253):**
```python
save_dir = f"/tmp/best_{EXP_ID}"
model.save_pretrained(save_dir)
feature_extractor.save_pretrained(save_dir)           # <-- add this
with open(f"{save_dir}/ctc_vocab.json", "w") as f:    # <-- and this
    json.dump(char2idx, f, ensure_ascii=False)
```

`whisper_finetune.py` already does both correctly (saves `processor`).

**Lesson**: always save the full artifact set needed for re-loading.
If you can't do `from_pretrained(save_dir)` and reproduce inference,
the save is incomplete.

---

## 3. kenlm / pyctcdecode build failures on HF Jobs

**Symptom:**
```
error: 'PyLongObject' has no member named 'ob_digit'
hint: indicates you need to install a library that provides "Python.h"
```

**Root cause:** The `kenlm` package on PyPI is source-only (no prebuilt
wheels). The HF Jobs default `ghcr.io/astral-sh/uv:python3.12` image
doesn't include Python development headers, so the C++ extension
won't compile. `pypi-kenlm` has the same problem.

**Workarounds:**

1. **Pure-Python ARPA scorer** (what we did): wrote `ArpaLM` class in
   `scripts/hf_jobs/decode_with_lm.py` that parses ARPA and does Katz
   backoff. Works but slower and less accurate than KenLM's Kneser-Ney
   smoothing.

2. **Use pyctcdecode without kenlm**: pyctcdecode's beam search engine
   does NOT hard-require kenlm. You can plug in a custom `LanguageModel`
   subclass that calls our ArpaLM under the hood. This gives us proper
   CTC prefix beam search (collapses blanks correctly) with our pure-
   Python LM scoring.

3. **Custom Docker image**: Build an image from `pytorch/pytorch:*-devel`
   (which has Python.h), install kenlm inside, use via `--image` flag.
   Heavier lift, deferred.

**Current status**: using approach #1 (pure-Python). If LM gains plateau,
upgrade to #2 (pyctcdecode + custom LM class) for better beam search.

---

## 4. MMS adapter state_dict mismatch when loading for inference

**Symptom (D4_C3v3_lm job):**
```
RuntimeError: Error(s) in loading state_dict for Wav2Vec2ForCTC:
  size mismatch for lm_head.weight: ... from checkpoint: torch.Size([154, ...])
  while the current model expects torch.Size([115, ...]).
```

**Root cause:** MMS (`facebook/mms-1b-all`) loads with language-specific
adapter heads. During training we set `target_lang="fra"` and overrode
`vocab_size` to match our Adja alphabet (115 chars). At inference, we
loaded without specifying `target_lang`, so the model defaulted back to
its full 154-char English head and rejected our weights.

**Fix (not applied yet)**: in `decode_with_lm.py`, when loading an MMS
model, also pass `target_lang=os.environ.get("MMS_TARGET_LANG", "fra")`
to `Wav2Vec2ForCTC.from_pretrained`. Requires understanding that
MMS adapters mutate the tensor shapes.

**Workaround**: skip MMS LM decoding for now; XLS-R works fine since
it doesn't have language adapters.

---

## 5. Python version pinning for kenlm (if ever used)

**Symptom:** `pip install kenlm` works on 3.11 but fails on 3.12 with
an `ob_digit` compilation error.

**Root cause:** Python 3.12 changed internals of `PyLongObject` that
kenlm's Cython bindings reference.

**Fix**: In PEP 723 headers, if kenlm is needed, pin:
```python
# /// script
# requires-python = "==3.11.*"
# dependencies = ["kenlm", ...]
# ///
```
And pass `--python 3.11` to `hf jobs uv run` (doesn't hurt to do both).

Still requires Python.h headers to be present in the runner image.

---

## Recommended Defaults for Whisper Fine-Tune on Low-Resource

From our C2 (CER=27%) and E4 (CER=24.9%) runs:
- `BATCH_SIZE=8`, `GRAD_ACCUM=4` (effective batch 32)
- `LR=1e-5`, `WARMUP_STEPS=500`
- `MAX_EPOCHS=50`, `PATIENCE=20` — DO NOT go below 20 on tiny datasets;
  CTC and Whisper both need 20-30 epochs to converge on ~1k utterances
- `no_repeat_ngram_size=3` at inference (prevents loops)
- Pad input_features to exactly 3000 (see fix #1)
- Save both model AND processor (see fix #2)

Our original E4 with these settings reached CER=24.9% at epoch 50,
still improving. Plan for longer training if dev CER is still dropping.
