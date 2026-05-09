# Training Hyperparameters for ASR: A Practical Guide

You have run ASR experiments. Some worked (C2 Whisper converged nicely over 50 epochs), some
collapsed (C3 MMS died at epoch 6). This guide explains *why* those things happened and how
to make better hyperparameter choices going forward.

Everything here uses real numbers from our Adja ASR experiments on ~1,277 training samples.

---

## 1. Epochs: How Many is Enough?

**What is an epoch?** One complete pass through ALL of your training data. With 1,277
training samples and batch_size=8, one epoch = ceil(1277/8) = 160 batches. Each batch
processes 8 audio clips, computes the loss, and updates the model weights.

### The lifecycle of training

Training follows a predictable pattern:

```
Epoch 1-5:     Loss drops fast, dev CER drops fast       (the model is learning basics)
Epoch 5-30:    Loss drops slower, dev CER still dropping  (the model is refining)
Epoch 30-50:   Loss still drops, dev CER flattens         (diminishing returns)
Epoch 50+:     Loss keeps dropping, dev CER goes UP       (OVERFITTING)
```

That last stage is the trap. The model memorizes the training data instead of learning
generalizable patterns. Training loss keeps going down because the model is getting better
at the training set, but it is getting *worse* on unseen data.

### How to tell if you need more epochs

Look at your dev CER curve:
- **Dev CER still improving at the last epoch?** You stopped too early. Increase max_epochs.
- **Dev CER flat for 10+ epochs?** You have enough epochs. The model is done learning.
- **Dev CER going up while train loss goes down?** Overfitting. You needed to stop earlier.

### What happened in our experiments

**C2 (Whisper fine-tune):** Loss went from ~8.5 to ~0.24 over 50 epochs. Dev CER improved
consistently. The model was still learning at epoch 50, which means 30 epochs (our original
config) was probably not enough. We should have set max_epochs higher and let early stopping
decide when to quit.

**E4 (Whisper-Ewe fine-tune):** Slow start for the first ~14 epochs (dev CER barely moved),
then rapid improvement kicked in. If we had set patience=10, early stopping would have killed
the run before the model even got going. This is why patience matters -- some models need a
warmup period before they start improving on the dev set.

**C3 (MMS fine-tune):** CER plateaued almost immediately and early stopping killed it at
epoch 6 (patience was only 5). But the problem was NOT epochs -- it was the learning rate and
architecture sensitivity. Giving it 100 more epochs would not have helped. When a model
flatlines from the very start, the issue is somewhere else.

### Comparison to NMT

In your NLLB fine-tuning for machine translation, you trained for ~10 epochs and that was
plenty. ASR needs more epochs because the mapping from audio to text is fundamentally harder
than text-to-text. Audio is continuous, noisy, and high-dimensional. Text-to-text starts with
discrete tokens that already carry meaning. More epochs compensates for the harder learning
problem.

### Rule of thumb

Set `max_epochs: 50-100` and rely on early stopping with patience 15-20 to decide when to
actually stop. Better to set it too high and let early stopping handle it than to cut
training short.


---

## 2. Batch Size: Memory vs Learning

**What is batch size?** The number of samples the model looks at before updating its weights.
With batch_size=8, the model processes 8 audio clips, computes the average loss across all 8,
and makes one weight update. Then it moves to the next 8 clips.

### Why batch size matters

Think of it like getting directions. If you ask 1 person, you get a specific answer that
might be wrong. If you ask 32 people and average their answers, you get a more reliable
direction but it takes longer to collect.

- **Bigger batch** (32-64): More stable gradient estimates. The model takes confident, steady
  steps. But each step requires more GPU memory because you are holding 32 audio clips and
  their intermediate computations in memory simultaneously.

- **Smaller batch** (4-8): Noisier gradient estimates. The model takes jittery steps that
  sometimes overshoot. But it fits in limited GPU memory, and the noise can actually help --
  it acts as a regularizer that sometimes leads to better generalization.

### Gradient accumulation: the best of both worlds

You do not need to fit everything in GPU memory at once. Gradient accumulation lets you
simulate a large batch using multiple small forward passes:

```
batch_size=8, gradient_accumulation_steps=4  -->  effective batch size = 32
```

How it works:
1. Process 8 samples, compute gradients, but DO NOT update weights. Store the gradients.
2. Process 8 more samples, compute gradients, ADD them to the stored gradients.
3. Repeat 2 more times (4 total mini-batches).
4. NOW update the weights using the accumulated gradients from all 32 samples.

The learning dynamics are mathematically the same as batch_size=32, but you only need
enough GPU memory for 8 samples at a time.

### What we used

All our experiments use batch_size=8 with gradient_accumulation_steps=4, giving an effective
batch of 32. This fits on an A100 40GB slice and provides stable enough gradients.

```yaml
# From C2 config:
batch_size: 8
gradient_accumulation_steps: 4
# Effective batch: 8 * 4 = 32
```

With 1,277 training samples and effective batch 32: one epoch = ceil(1277/32) = 40 optimizer
steps. Over 50 epochs, that is 2,000 total weight updates.

### Rule of thumb for ASR

Effective batch sizes of 16-64 are typical for ASR fine-tuning. If your GPU runs out of
memory, reduce batch_size and increase gradient_accumulation_steps proportionally. The model
does not care how you achieve the effective batch size.


---

## 3. Learning Rate: The Most Important Number

The learning rate controls how big a step the model takes when updating its weights. It is
the single most impactful hyperparameter.

### Intuition

Imagine you are blindfolded on a hilly landscape, trying to find the lowest valley. Each
step you take is the learning rate:

- **Too high (1e-2):** You take huge steps and overshoot the valley, bouncing back and forth
  over it, or flying off into the mountains. Loss explodes or oscillates wildly.
- **Too low (1e-7):** You take tiny steps. You might find a valley eventually, but you will
  run out of time (epochs) before you get there. Loss barely decreases.
- **Just right (1e-5 for fine-tuning):** You take measured steps that get you into the valley
  and let you settle at the bottom.

### The right learning rate depends on what you are doing

The key question is: **is the model starting from scratch or from a pre-trained checkpoint?**

| Scenario                        | Learning rate  | Why                                      |
|--------------------------------|----------------|------------------------------------------|
| Whisper fine-tuning (C2, E4)    | 1e-5           | Already pre-trained. Small adjustments.  |
| CTC fine-tuning (MMS, wav2vec2) | 1e-5 to 3e-5  | Pre-trained encoder, new CTC head.       |
| Training from scratch           | 1e-3 to 3e-4  | Learning everything from random init.    |

A pre-trained model already has good representations. You just need to nudge it toward Adja.
Big learning rates would destroy what it already knows (this is called "catastrophic
forgetting").

A from-scratch model starts with random weights. It needs bigger steps to make progress
before running out of epochs.

### Learning rate schedule: warmup + decay

You do not use a constant learning rate. Instead:

**1. Warmup phase (first 200-500 steps):**
Start with a tiny learning rate and linearly increase to the target LR. Why? At the very
beginning, the gradients are computed from random or poorly-calibrated activations. Taking
big steps based on these garbage gradients could push the model into a bad region of the
loss landscape that is hard to recover from.

```
Step 1:    LR = 0.0       (essentially frozen)
Step 250:  LR = 0.5e-5    (halfway to target)
Step 500:  LR = 1.0e-5    (full target LR)
```

**2. Decay phase (after warmup):**
Gradually decrease the learning rate toward 0. The idea: early in training, you want big
steps to make rapid progress. Later, you want small steps for fine-grained refinement.

Common schedules:
- **Linear decay:** LR decreases linearly from target to 0 over remaining steps.
- **Cosine decay:** LR follows a cosine curve (slower decay at start, faster at end).

Our experiments use linear warmup + linear decay:

```python
# From C2/E4 train.py:
def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

### How to diagnose learning rate problems

| Symptom                                      | Likely cause        | Fix                       |
|----------------------------------------------|---------------------|---------------------------|
| Loss explodes (goes to inf or NaN)            | LR too high         | Reduce LR by 5-10x        |
| Loss oscillates wildly, never settles         | LR too high         | Reduce LR by 2-5x         |
| Loss decreases but CER does not improve       | LR maybe OK         | Check other issues first   |
| Loss stuck at the same value for many epochs  | LR too low          | Increase LR by 2-5x       |
| Loss stuck at one specific value from epoch 1 | Model capacity issue | Check architecture/config  |

### What happened to C3

C3 (MMS) used lr=3e-5. For a CTC model, this is on the aggressive side. Combined with the
CTC loss function's sensitivity (more on this in section 5), this likely contributed to the
model never learning. A lower LR (1e-5) with longer patience might have saved it.


---

## 4. Early Stopping: When to Stop Training

Training until max_epochs wastes compute if the model has stopped learning. Early stopping
monitors a dev metric and stops training when improvements stall.

### How it works

```
Epoch 1:   Dev CER = 85.2%    Best! Save checkpoint. patience_counter = 0
Epoch 2:   Dev CER = 72.1%    Best! Save checkpoint. patience_counter = 0
Epoch 3:   Dev CER = 73.5%    Worse. patience_counter = 1
Epoch 4:   Dev CER = 71.8%    Best! Save checkpoint. patience_counter = 0
Epoch 5:   Dev CER = 72.0%    Worse. patience_counter = 1
Epoch 6:   Dev CER = 72.3%    Worse. patience_counter = 2
...
Epoch N:   patience_counter hits patience limit --> STOP
```

### Which metric to monitor

**Use CER (Character Error Rate), NOT training loss.**

Why? Loss and CER can diverge. Training loss measures how well the model predicts the next
token given the correct previous tokens (teacher forcing). CER measures how well the model
generates the entire transcription on its own (autoregressive decoding). A model can get
better at the first task while getting worse at the second -- this is overfitting.

In our NMT experiments, we learned this same lesson: early stop on chrF, not on loss. CER
is the ASR equivalent of chrF.

### Patience: the critical setting

Patience is how many epochs without improvement you tolerate before stopping.

**C3 (MMS) had patience=5.** It stopped at epoch 6. But as we saw with E4, some models have
a slow start and need 10+ epochs before improvement kicks in. Patience=5 was too aggressive
-- it killed training before the model had a chance.

**E4 (Whisper-Ewe) was slow for ~14 epochs.** If patience had been set to 10, it would have
been killed before the rapid improvement phase even began.

Setting patience:
- **Too small (5):** Risk killing training during a normal plateau or slow-start phase.
- **Too large (50):** Waste compute running for 50 epochs after the model has peaked.
- **Sweet spot (15-20):** Tolerates slow starts and temporary plateaus, but does not waste
  too much compute on a truly stuck model.

### Always save the BEST checkpoint

Early stopping means the last epoch is NOT the best epoch. Your training loop must save the
model checkpoint whenever dev CER hits a new minimum, and load that checkpoint for final
evaluation.

From our train.py:
```python
if dev_cer < best_cer:
    best_cer = dev_cer
    patience_counter = 0
    model.save_pretrained(str(ckpt_dir))  # save the BEST model
else:
    patience_counter += 1
```

After training, we load the best checkpoint (not the final one) for evaluation:
```python
model = WhisperForConditionalGeneration.from_pretrained(str(ckpt_dir))
```


---

## 5. Why CTC Models Collapsed But Whisper Didn't

This is the most confusing thing in our experiments so far. C2 (Whisper) converged smoothly.
C3 (MMS with CTC) flatlined and died. Why?

### CTC collapse explained

CTC (Connectionist Temporal Classification) has a special "blank" token. At each audio frame,
the model predicts either a character or blank. Blanks are collapsed during decoding:

```
Model output: [blank, blank, h, h, blank, e, blank, blank, l, l, l, blank, o]
After collapse: "helo"  (wait, that's wrong -- CTC removes CONSECUTIVE duplicates)
Actually:       "helo"  -->  this is why repeated chars need special handling
```

**CTC collapse** is when the model learns to predict blank for EVERY frame. This is a valid
output that produces zero CTC loss (the loss function allows any alignment). Once the model
falls into this trap, every prediction is blank, the output is empty string, and CER = 100%.
The model cannot recover because the gradients from an all-blank prediction do not push it
toward producing real characters.

### Why Whisper does not have this problem

Whisper is an encoder-decoder model. The decoder must produce actual text tokens
autoregressively (one at a time, conditioned on previous tokens). There is no "blank" escape
hatch. The decoder must commit to real characters, so it cannot collapse into the trivial
solution.

### What makes CTC fragile

CTC is more sensitive to several factors that encoder-decoder models handle gracefully:

1. **Learning rate:** CTC's loss landscape has sharp cliffs near the blank-collapse region.
   A learning rate that is slightly too high can push the model over the cliff.

2. **Attention mask correctness:** If the attention mask is wrong (e.g., marking padded
   frames as real), the model sees garbage frames and learns to output blank for them. This
   habit can spread to real frames.

3. **fp16 precision:** CTC loss involves log-sum-exp computations that can overflow or
   underflow in half precision. Always compute CTC loss in fp32, even if the forward pass
   uses fp16.

4. **Input normalization:** CTC models are sensitive to the scale of input features. The
   wav2vec2 feature encoder expects normalized input, and feeding it unnormalized audio
   can cause instability.

### How to prevent CTC collapse

- Use a lower learning rate for CTC models: 1e-5 instead of 3e-5
- Freeze the feature encoder (CNN layers) -- this is already set in our configs
- Compute CTC loss in fp32: `loss = ctc_loss_fn(log_probs.float(), ...)`
- Set `ctc_zero_infinity=True` in the model config (clips infinite loss values)
- Use longer warmup (500+ steps) to let the model ease into training
- Set patience to at least 15 -- CTC models often have a slow start

### Our actual bugs

Several issues in our CTC experiments compounded the problem:

- **Pre-padded attention masks:** The collate function was generating masks that covered
  padding, causing the model to process silence as if it were speech.
- **fp16 CTC loss:** Running the CTC loss computation in half precision caused numerical
  instability. The fix: always cast log_probs to float32 before the loss.
- **Patience too low:** With patience=5, the CTC model was killed before it had a chance to
  escape the initial plateau.


---

## 6. Practical Checklist: Before Running an Experiment

Copy this checklist and go through it before every HPC submission. A crash on line 1 after
waiting hours in the SLURM queue is an expensive mistake.

### Pre-flight checks

```
[ ] Dry-run passes locally? (--dry-run flag, 2 steps on 5 samples)
[ ] All imports resolve? (transformers, torch, soundfile, etc.)
[ ] Data paths exist and manifests are not empty?
[ ] Model weights are pre-downloaded to cluster storage?
    (Never pull from HuggingFace Hub during a SLURM job)
```

### Hyperparameter sanity

```
[ ] Learning rate reasonable for this model type?
    - Fine-tuning pre-trained: 1e-5
    - CTC fine-tuning: 1e-5 to 3e-5
    - From scratch: 1e-3 to 3e-4

[ ] Batch size fits in GPU memory?
    - Check with a dry-run first
    - If OOM: reduce batch_size, increase gradient_accumulation_steps

[ ] Effective batch size in the 16-64 range?
    - effective = batch_size * gradient_accumulation_steps

[ ] Warmup steps set?
    - 500 for Whisper fine-tuning
    - 200 for CTC fine-tuning
    - ~10% of total steps as a general heuristic

[ ] Max epochs high enough? (50-100, not 10-20)

[ ] Early stopping patience set to 15-20? (not 5)
```

### Monitoring and saving

```
[ ] Dev metric is CER (not loss)?
    - eval_metric: cer in config.yaml

[ ] Logging shows BOTH loss AND CER each epoch?
    - You need both to diagnose problems

[ ] Model checkpoint saved on best dev CER (not last epoch)?

[ ] Output directory has enough disk space for checkpoints?

[ ] SLURM job requests enough wall time?
    - 50 epochs * ~2 min/epoch = ~2 hours + buffer = request 4h
```

### CTC-specific checks

```
[ ] CTC loss computed in fp32?
[ ] ctc_zero_infinity=True?
[ ] Feature encoder frozen? (freeze_feature_encoder: true)
[ ] Attention masks correctly computed from actual audio lengths?
```


---

## 7. Quick Reference: Our Experiment Configs

For copy-paste when setting up new experiments:

| Parameter                   | Whisper FT (C2/E4) | CTC FT (C3/C4/C5) | Notes                    |
|----------------------------|--------------------|--------------------|--------------------------|
| learning_rate               | 1e-5               | 3e-5               | Lower is safer for CTC   |
| warmup_steps                | 500                | 200                | Longer warmup for Whisper |
| max_epochs                  | 30 (increase to 50+) | 30 (increase to 50+) | Always pair with early stopping |
| batch_size                  | 8                  | 8                  | Limited by GPU memory     |
| gradient_accumulation_steps | 4                  | 4                  | Effective batch = 32      |
| early_stopping_patience     | 5 (increase to 15-20) | 5 (increase to 15-20) | 5 was too aggressive     |
| max_grad_norm               | 1.0                | 1.0                | Standard gradient clipping |
| weight_decay                | 0.01               | 0.01               | Standard regularization   |
| freeze_encoder              | false              | N/A                | For Whisper only          |
| freeze_feature_encoder      | N/A                | true               | For CTC models only       |


---

## 8. Links and Resources

### Foundational understanding
- "But what is a Neural Network?" -- 3Blue1Brown (YouTube). Visual intuition for how neural
  networks learn. Start here if gradients and loss functions feel abstract.
- "Training Neural Networks" -- Andrej Karpathy's Stanford CS231n lectures (YouTube).
  Practical walkthrough of training dynamics, learning rates, and debugging.

### The best single blog post on this topic
- "A Recipe for Training Neural Networks" by Andrej Karpathy:
  https://karpathy.github.io/2019/04/25/recipe/
  Covers the entire workflow from data inspection to hyperparameter tuning. Read this.

### ASR-specific fine-tuning guides
- HuggingFace blog: "Fine-Tune Wav2Vec2 for English ASR with Transformers"
  https://huggingface.co/blog/fine-tune-wav2vec2-english
  Shows learning rate scheduling, CTC training, and eval setup.
- HuggingFace blog: "Fine-Tune Whisper For Multilingual ASR with Transformers"
  https://huggingface.co/blog/fine-tune-whisper
  Our C2 experiment closely follows this pattern.

### Hyperparameter tuning techniques
- The "learning rate finder" technique (Leslie Smith, 2017): Train for a few hundred steps
  with LR increasing exponentially from 1e-7 to 1e-1. Plot loss vs LR. The optimal LR is
  roughly where the loss is decreasing fastest (steepest part of the curve), typically one
  order of magnitude below where loss explodes.
- Weights & Biases guide to hyperparameter tuning:
  https://docs.wandb.ai/guides/sweeps
  Automated hyperparameter search. Useful when you have the compute budget for it.

### Papers
- wav2vec 2.0: https://arxiv.org/abs/2006.11477 (the architecture behind MMS, XLS-R)
- Whisper: https://cdn.openai.com/papers/whisper.pdf (the architecture behind C2/E4)
- CTC original paper: https://www.cs.toronto.edu/~graves/icml_2006.pdf (Alex Graves, 2006)
- MMS: https://jmlr.org/papers/v25/23-1318.html (1100+ language ASR)
