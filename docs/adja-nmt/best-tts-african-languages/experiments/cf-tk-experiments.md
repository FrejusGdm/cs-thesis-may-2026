# CF/TK Experiments: Catastrophic Forgetting Mitigation and Tokenizer Expansion

Adja TTS — CSM 1B. Written 2026-04-26.

---

## Background: Why Stage 2 Fails

The Gbe-family cascade strategy (Stage 1: Ewe → Stage 2: Adja) produced intelligible
Ewe audio (T1-csm-ewe-stage1, job `69e830e1`, 2026-04-22) but catastrophic forgetting
in every Stage 2 attempt:

- **T1-csm-ewe-adja-stage2** (job `69e93755`): ran full 20-epoch budget, early stopping
  never fired, best_eval_loss=6.5115, all 5 generated samples hit the 10s silence cap.
- **T2-orpheus-fr-ewe-adja-stage2** (job `69e9375a`): noise, best_eval_loss=5.6356.
- **T2-orpheus-en-ewe-adja-stage2-mixed** (job `69e977f8`, 1:1 Ewe/Adja mix): "better
  noise", loss plateau 5.648 — virtually identical to pure-Adja Stage 2. Naive data
  mixing insufficient.

Root cause: initial Stage 2 step loss ~17 nats vs ln(2051)=7.6 random chance for the
Mimi codebook (or 8.3 for SNAC). The Stage-1 model fires Ewe token distributions when
prompted with Adja text — textbook catastrophic forgetting. The backbone has not seen
enough Adja text to distinguish it from Ewe.

Scripts: `scripts/sagemaker_jobs/train_CF1_ewc_stage2.py`,
`train_CF2_curriculum_stage2.py`, `train_CF3_frozen_backbone_stage2.py`,
`train_TK1_tokfix_ewe_stage1.py`.

---

## CF1: Elastic Weight Consolidation (EWC) Stage 2

**Script**: `scripts/sagemaker_jobs/train_CF1_ewc_stage2.py`

### Hypothesis

The Llama backbone has learned Ewe-specific weight configurations during Stage 1.
Penalising changes to the weights that most mattered for Ewe (measured by Fisher
information) should let the model adapt to Adja without erasing the Ewe prior.

### Why Prior Approaches Failed

Naive Stage 2 (job `69e93755`) applied no constraints on weight changes. The LoRA
adapter was free to overwrite the Ewe-specific directions in the attention and FFN
layers. The 1:1 mixed training (job `69e977f8`) sampled Ewe data but placed no
penalty on the magnitude of weight changes — the optimizer still moved weights away
from the Stage-1 optimum.

### Technical Approach

EWC (Kirkpatrick et al. 2017) adds an L2 penalty weighted by the diagonal Fisher
information matrix:

```
L_total = L_adja + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2
```

where `theta*_i` is the Stage-1 parameter value and `F_i = E[(d log p / d theta_i)^2]`
is estimated over 500 Ewe examples. The penalty is implemented in `EWCTrainer`, a
subclass of HF `Trainer` that overrides `compute_loss`.

Key implementation detail: with LoRA, only adapter weights are trainable, so the
Fisher is only computed and applied over adapter parameters. This is intentional —
we penalise changes to the LoRA deltas that encode Ewe phonology.

```python
# From EWCTrainer.compute_loss:
ewc_penalty = torch.tensor(0.0, device=device, dtype=task_loss.dtype)
param_dict = dict(model.named_parameters())
for name, (anchor, fisher) in self.ewc_params.items():
    if name not in param_dict:
        continue
    param = param_dict[name]
    if param.requires_grad:
        ewc_penalty += (fisher * (param - anchor).pow(2)).sum()
total_loss = task_loss + (self.ewc_lambda / 2.0) * ewc_penalty
```

### Expected Outcomes

- `--ewc-lambda=1000` (default): strong EWC. Stage 2 train loss should decrease
  more slowly than naive Stage 2 but without the forgetting signature. If Adja train
  loss cannot decrease below ~6 nats, lambda is too high — retry with 100 or 10.
- Audio quality: expect gradual improvement over epochs rather than immediate noise.
- Ablation: compare best_eval_loss against T1-csm-ewe-adja-stage2 (6.5115). Any
  result below ~5.5 at the same epoch count would indicate meaningful improvement.

### Cost Estimate

Fisher computation on 500 Ewe samples: ~5-10 min on L40S (500 forward+backward
passes, batch size 1). Full Stage 2 training: ~35-45 min (similar to job `69e93755`
which was 37.5 min). Total: ~45-55 min on L40S. Approx $0.50-0.70 at HF Jobs rates.

### References

- Kirkpatrick et al. 2017 PNAS "Overcoming catastrophic forgetting in neural networks"
  https://www.pnas.org/doi/10.1073/pnas.1611835114
- Sesame CSM: https://github.com/SesameAILabs/csm
- Mimi codec: https://arxiv.org/abs/2410.00037

---

## CF2: Curriculum Annealing Stage 2

**Script**: `scripts/sagemaker_jobs/train_CF2_curriculum_stage2.py`

### Hypothesis

Starting from a Ewe-heavy dataset and gradually shifting to Adja gives the model
time to retain Ewe phonological priors in early epochs while converging on Adja
prosody by the final epochs. This is less aggressive than EWC (no explicit penalty)
but more principled than naive 1:1 mixing.

### Why Prior Approaches Failed

The 1:1 mixed training (job `69e977f8`) sampled Ewe and Adja at a fixed ratio
throughout training. A fixed ratio does not account for the asymmetric difficulty:
the model is starting from Ewe priors and needs early epochs to retain them before
shifting. A static 1:1 ratio updates the model with equal Adja pressure from step 1,
which is sufficient to corrupt the Ewe prior (loss plateau unchanged vs pure Adja).

### Technical Approach

Custom training loop (not HF Trainer) with per-epoch dataset resampling. The Ewe:Adja
ratio follows a linear schedule:

```python
def get_epoch_datasets(ewe_preprocessed, adja_preprocessed, epoch, total_epochs,
                       samples_per_epoch=400):
    alpha = epoch / max(total_epochs - 1, 1)
    ewe_frac = 5 - alpha * 4   # 5.0 → 1.0
    adja_frac = 1 + alpha * 4  # 1.0 → 5.0
    ...
```

Epoch 0: 5:1 Ewe:Adja (333 Ewe + 67 Adja per 400-sample epoch).
Last epoch: 1:5 (67 Ewe + 333 Adja).

Both datasets are preprocessed once before the loop (expensive processor call done
once, not per epoch). Evaluation uses a held-out Adja dev set (10% of Adja dataset).

Training uses AdamW + cosine LR schedule + grad clipping, with 10-step linear warmup.

### Expected Outcomes

- Earlier epochs: low Adja dev loss degradation (Ewe retention), potentially slightly
  higher than Stage 1 Ewe eval_loss.
- Later epochs: Adja dev loss should decrease as Adja data dominates the curriculum.
- Key diagnostic: plot `dev_loss` per epoch from `training_history` in metrics.json.
  A U-shaped or monotonically decreasing Adja dev curve would indicate success.
- Compare best_epoch against T1 naive Stage 2 (20 epochs, never improved).

### Cost Estimate

Preprocessing: ~5 min (once for each dataset). Training loop: 20 epochs × 400
samples/epoch = 8,000 gradient steps total (effective). Similar wall-clock to
T1-csm-ewe-adja-stage2 (37.5 min). Estimate ~40-50 min on L40S. Approx $0.45-0.60.

### References

- Bengio et al. 2009 "Curriculum Learning"
  https://dl.acm.org/doi/10.1145/1553374.1553380
- McCloskey & Cohen 1989 "Catastrophic interference in connectionist networks"
- Sesame CSM: https://github.com/SesameAILabs/csm

---

## CF3: Frozen Llama Backbone Stage 2

**Script**: `scripts/sagemaker_jobs/train_CF3_frozen_backbone_stage2.py`

### Hypothesis

The catastrophic forgetting originates in the Llama backbone (text-to-context
layers). If we freeze the backbone entirely and only train the audio generation
heads (Mimi codebook projections, audio token embeddings, depth decoder), the
backbone cannot unlearn Ewe while the heads adapt to Adja phoneme-to-audio
alignment.

CSM has a natural architectural separation: the Llama backbone handles
text-to-context mapping, and dedicated audio heads handle codec token generation.
Freezing the backbone preserves the Ewe text priors while allowing the audio
heads to adapt to Adja phonology.

### Why Prior Approaches Failed

In all Stage 2 runs (jobs `69e93755`, `69e9375a`, `69e977f8`), LoRA was applied to
Llama attention and FFN layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`). These LoRA adapter weights live inside the backbone's
attention/FFN stack. Updating them with Adja loss directly overwrites the Ewe
phonological directions that made Stage 1 intelligible.

### Technical Approach

1. Load Stage 1 checkpoint.
2. Freeze ALL parameters (`requires_grad_(False)` for every param).
3. Re-enable grad for parameters whose name contains any of `AUDIO_HEAD_SUBSTRINGS`
   (codebook, audio_embed, audio_head, mimi, projection, depth_decoder, etc.).
4. Raise `ValueError` if zero trainable params result (heuristic failure).
5. Train with standard HF Trainer on Adja data.

**IMPORTANT**: Run `--print-params` before SageMaker submission to verify that the
freeze heuristic matches actual CSM parameter names:

```bash
python train_CF3_frozen_backbone_stage2.py --print-params
```

This downloads the Stage 1 checkpoint, prints all `(name, shape, requires_grad)`
tuples, and exits. Update `AUDIO_HEAD_SUBSTRINGS` based on the output before
submitting. All freeze decisions in the script are marked with
`# TODO: verify with --print-params`.

### Expected Outcomes

- Very few trainable parameters (~1-10% of model). Training will be fast.
- If the audio heads are sufficient to learn Adja: dev_loss should decrease and audio
  should show intelligible Adja phonemes.
- If the backbone is also required for Adja adaptation (possible): dev_loss will
  plateau at a high value regardless of epochs. In that case, a partial-freeze
  approach (freeze only bottom N Llama layers) may be needed as a follow-up.
- Key metric: compare trainable param count printed at startup against CF1/CF2
  (which use full LoRA r=32 ~26M trainable params).

### Cost Estimate

Fewer trainable params → faster training. Estimate ~20-30 min on L40S. Approx
$0.25-0.40. The `--print-params` run takes ~2-3 min (model download + print).

### References

- Howard & Ruder 2018 "Universal Language Model Fine-Tuning" (ULMFiT), layer
  freezing strategy: https://arxiv.org/abs/1801.06146
- Sesame CSM architecture: https://github.com/SesameAILabs/csm
- Mimi codec: https://arxiv.org/abs/2410.00037

---

## TK1: Tokenizer-Expanded CSM, Stage 1 on Ewe

**Script**: `scripts/sagemaker_jobs/train_TK1_tokfix_ewe_stage1.py`

### Hypothesis

T1-csm-tokfix (job `69e857f3`, 2026-04-22) applied tokenizer expansion on a
cold-start (no Gbe prior) and produced "noise with speech fragments", refuting the
hypothesis that tokenizer fragmentation alone is the primary bottleneck.

However, this left open a compound question: does tokenizer expansion help when
combined with the Gbe-family cascade? If the Stage 1 Ewe training uses an expanded
tokenizer, the model learns consistent Adja/Ewe character → audio mappings from real
phonemic tokens (not byte fragments). Stage 2 would then start from a checkpoint
where both:
  (a) the Gbe-family acoustic prior is established (from Ewe Stage 1), AND
  (b) the text encoder already represents Adja characters natively (from expansion).

This tests whether tokenizer expansion has additive value on top of the Gbe cascade.

### Why Prior Approaches Did Not Answer This Question

- T1-csm-tokfix (job `69e857f3`): expanded tokenizer + direct Adja training. No Ewe
  prior. Result: noise. Conclusion: tokenizer alone is insufficient.
- T1-csm-ewe-stage1 (job `69e830e1`): standard tokenizer + Ewe training. No expansion.
  Result: intelligible Ewe. Conclusion: Ewe prior works, tokenizer not expanded.
- Missing cell: expanded tokenizer + Ewe training → the present experiment (TK1).
  If TK1 Ewe Stage 1 produces intelligible Ewe (similar to T1), then a TK2 Stage 2
  (TK1 checkpoint → Adja) can test the compound hypothesis.

### Technical Approach

Identical to T1_csm_ewe_stage1.py, with tokenizer surgery inserted before training.
The expansion logic is copied verbatim from T1_csm_tokfix.py:

1. `processor.tokenizer.add_tokens(ADJA_CHARS)` — adds 35 Adja/Gbe characters.
2. `model.resize_token_embeddings(vocab_after)` with fallback to
   `_manual_resize_token_embeddings()` — extends embedding matrix with N(0, 0.02)
   init for new rows.
3. `config.vocab_size` restore patch — prevents backbone_loss crash (CSM uses
   `config.vocab_size` for the Mimi audio codebook dimension=2051, not text vocab).

Diagnostic prints tokenize a sample Ewe sentence and a sample Adja sentence before
and after expansion. The before/after comparison confirms the expansion worked and
shows which characters changed from byte fragments to native tokens.

The merged model is pushed to `JosueG/adja-tts-checkpoints/TK1_csm_tokfix_ewe_stage1`
for a follow-on TK2 Stage 2 script.

Key code snippet (diagnostic print):
```python
tokens_ewe_before = processor.tokenizer.tokenize(sample_ewe)
tokens_adja_before = processor.tokenizer.tokenize(sample_adja)
# ... expand tokenizer ...
tokens_ewe_after = processor.tokenizer.tokenize(sample_ewe)
tokens_adja_after = processor.tokenizer.tokenize(sample_adja)
```

### Expected Outcomes

- Stage 1 Ewe result should match T1-csm-ewe-stage1 (job `69e830e1`):
  best_eval_loss ~4.3, intelligible Ewe audio.
- If Stage 1 Ewe quality degrades noticeably (best_eval_loss > 5.0): the expanded
  embedding rows for new tokens require more data than Ewe provides to learn, and
  the compound approach is not viable.
- If Stage 1 Ewe quality is preserved: proceed with TK2 Stage 2 (TK1 ckpt → Adja).
  The TK2 vs CF1/CF2/CF3 comparison will isolate the contribution of tokenizer
  expansion to catastrophic forgetting mitigation.

### Cost Estimate

Identical training setup to T1-csm-ewe-stage1 (job `69e830e1`, 17.4 min, 25.52 GB
VRAM). Expect ~17-22 min on L40S. Approx $0.20-0.28. The extra time for tokenizer
surgery and diagnostic prints is ~1 min.

### References

- T1_csm_tokfix.py (this repo): tokenizer surgery implementation
- T1-csm-tokfix registry entry: job `69e857f3`, 2026-04-22
- T1-csm-ewe-stage1 registry entry: job `69e830e1`, 2026-04-21
- Sesame CSM: https://github.com/SesameAILabs/csm
- WaxalNLP dataset: https://huggingface.co/datasets/google/WaxalNLP
- Mimi codec: https://arxiv.org/abs/2410.00037

---

## Submission Checklist

Before submitting any of these scripts to SageMaker (or HF Jobs):

1. Run `--smoke` locally to validate imports, dataset loading, and model forward pass.
2. For CF3 specifically: run `--print-params` and update `AUDIO_HEAD_SUBSTRINGS`.
3. Set `HF_TOKEN` in the job environment.
4. Verify Stage 1 checkpoint is accessible: `JosueG/adja-tts-checkpoints/T1_csm_ewe_stage1`.
5. After completion: update `experiments/registry.md` and `results/run-ledger.md`.
