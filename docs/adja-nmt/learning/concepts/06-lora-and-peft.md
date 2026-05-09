# 06 — LoRA and parameter-efficient fine-tuning

Last updated: **2026-04-21**

## The one-line version

LoRA lets you fine-tune a 3 B-parameter model on a single consumer GPU by
training a tiny rank-r update matrix instead of the full weight matrix.

## The math

For a pretrained weight matrix W ∈ R^{d×k}, LoRA parameterizes the update as:

```
W_updated = W + B · A,  where A ∈ R^{r×k}, B ∈ R^{d×r}, r << min(d, k)
```

Only A and B are trained; W stays frozen. If d = k = 4096 and r = 32, you
update 2 × 32 × 4096 = 262k params instead of 16.7M. 64× smaller footprint.

## Why this works

Empirical finding from Hu et al. (https://arxiv.org/abs/2106.09685): the
"update needed to adapt a pretrained LLM to a new task" usually lies in a
low-dimensional subspace. You don't need full-rank updates; r = 8-64 is
enough for most tasks.

## Hyperparameters we tune

| Param | Typical | What it controls |
|-------|---------|------------------|
| `r` (rank) | 8, 16, 32, 64 | Capacity. Larger r = more flexibility. |
| `lora_alpha` | r or 2r | Scale of the update. Usually set equal to r. |
| `target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | Which matrices get LoRA applied. The common 7 in Llama-style models. |
| `lora_dropout` | 0 or 0.05 | Regularization. 0 is fine for small tasks. |

## When LoRA is NOT enough

If the downstream task requires **new tokens in the embedding layer** (e.g.
Adja characters added via `tokenizer.add_tokens()`), LoRA on attention/MLP
isn't enough — you also need to update the embedding matrix, because the new
tokens start with random vectors. Our `tokfix` scripts handle this by calling
`model.resize_token_embeddings()` *before* wrapping in LoRA. The newly-added
embedding rows are trainable (since they're in the base model but initially
random), and the LoRA adapters learn the downstream mapping.

## `full_finetune` vs `lora`

- LoRA: fast, cheap, usually as good. Our default.
- Full fine-tune: updates every parameter. We tested this on CSM and Orpheus
  (T1-fullft, T2-fullft). Neither beat LoRA by a meaningful margin on Adja.
  That was the control that showed CSM/Orpheus's ceiling wasn't about
  trainable-parameter count — it was about the tokenizer (see concept 01).

## Papers to read

- Hu et al., "LoRA" — https://arxiv.org/abs/2106.09685
- Dettmers et al., "QLoRA" (4-bit quantization + LoRA) — https://arxiv.org/abs/2305.14314
- Liu et al., "DoRA" (weight-decomposed LoRA) — https://arxiv.org/abs/2402.09353 —
  newer and often better; we haven't tried it yet.

## Code reference

Inside any of our HF Jobs scripts:

```python
from peft import LoraConfig, get_peft_model
model = get_peft_model(model, LoraConfig(
    r=32, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none",
))
```
