# E1: Training Curve (crashed — no epochs completed)

Pulled from `hf jobs logs 69e03bb9ac288e522d8eedae`.

| Epoch | Loss | Dev CER | Dev WER |
|-------|------|---------|---------|
| — | — | — | — |

**No epoch completed.** The job crashed inside the first forward/backward pass of epoch 1 with a `torch.OutOfMemoryError`:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 220.00 MiB.
GPU 0 has a total capacity of 79.25 GiB of which 53.88 MiB is free.
Process 1264511 has 79.19 GiB memory in use.
```

A previous attempt (not shown in this specific job) had also failed with a `ValueError` because the Ewe adapter ships with `vocab_size=55` while the Adja vocab is 115 — the model's output head was smaller than the target labels. That was patched by setting `model.config.vocab_size = vocab_size` before training; the run captured here is the post-patch attempt, which then crashed on OOM.

Stack trace excerpt:
```
File ".../ctc_finetune.py", line 196, in <module>
  outputs = model(input_values=iv, attention_mask=am, labels=targets)
...
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 220.00 MiB.
```
