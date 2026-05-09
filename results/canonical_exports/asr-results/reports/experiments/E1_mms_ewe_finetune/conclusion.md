# E1: Conclusion

## Verdict
**crashed**

## What happened
Two sequential failures on the same experiment:

1. **First crash — ValueError on label range**. The Ewe adapter in `facebook/mms-1b-all` ships with `vocab_size=55`, but our Adja vocabulary has 115 tokens. When the CTC loss received label indices above 54, it raised `ValueError` on invalid label values. We patched this by explicitly overriding `model.config.vocab_size = vocab_size` (115) before training so the output head and the label space agree.
2. **Second crash — CUDA OOM**. After the fix, the job made it past construction and entered the first training forward pass, but the A100 80GB GPU hit `torch.OutOfMemoryError: Tried to allocate 220.00 MiB ... 79.19 GiB already in use`. The trainable parameter count reached ~961M, and activations for a single batch plus gradient state evidently exceeded the GPU's capacity at our batch/seq-length settings.

## Why
- **Vocab-mismatch crash** is a known MMS-adapter gotcha: each adapter has a language-specific vocab size, and the `Wav2Vec2ForCTC` head is sized to match, so a Swap-in-Adja-labels workflow needs an explicit reinit of `lm_head` + config override. C3 had the same issue for the French adapter (154 → 115); E1 has it for Ewe (55 → 115) with a larger mismatch.
- **OOM** likely from a combination of (a) gradient checkpointing being off or partially applied on the adapter path, (b) the CTC head reinit creating new parameters not tracked by the usual memory estimate, and (c) batch_size / max audio length not tuned down for the Ewe adapter path. The same A100 ran C3 (fra adapter) successfully at the same batch size, so the extra memory cost is specific to how the Ewe adapter interacts with our code path.

## What to do next
Investigated. Root causes: (1) Ewe adapter `vocab_size=55` < Adja `vocab_size=115` label-range crash (fixed); (2) OOM on A100 after the fix. Decision: **skip further E1 reruns for now**. The Ewe-adapter path has unique technical issues that aren't blocking our main research question, and the E4 track (Whisper-Ewe → Adja) is already producing the best numbers on the project (CER=24.9% at ep 50). If E4v2 continues to improve Ewe-initialized transfer, we may learn enough from that approach to decide whether fixing E1's OOM is worth it.

## Known follow-ups already queued
- **E4v2-s42** is queued (Whisper-Ewe → Adja, 100 epochs, patience=25). E4 is already the best model in the project; finishing E4v2 is higher priority than fixing E1's memory issues.
- No E1 rerun scheduled. If we revisit: cut batch size in half and confirm gradient checkpointing is applied along the adapter path.
