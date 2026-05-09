# /// script
# dependencies = ["transformers==4.48.3", "torch==2.5.1", "torchaudio==2.5.1", "datasets", "soundfile", "librosa", "numpy", "huggingface-hub"]
# ///
# Note: transformers pinned to 4.48.3 for determinism, but this script is
# now version-independent with respect to the Whisper 3000-mel-frames
# requirement: the collate function uses padding=True and then pads the mel
# tensor to exactly 3000 frames via torch.nn.functional.pad. Do NOT switch
# to padding="max_length" — that pads raw audio with silence, which teaches
# the decoder cross-attention a hallucination-loop failure mode (verified
# empirically in E4v2: loss→0.001 while CER→150-250%). The original working
# E4 (CER=24.9%) was trained with padding=True + auto-pad-to-3000, which
# this script now reproduces explicitly. See docs/whisper-training-gotchas.md.
"""
Whisper fine-tuning for Adja ASR on HuggingFace Jobs.
Self-contained. Supports vanilla Whisper (C2) and Whisper-Ewe (E4).

Env vars:
  EXP_ID      = C2 | E4
  MODEL_NAME  = openai/whisper-small | dodziraynard/whisper-small-ee
"""
from __future__ import annotations
import functools, json, os, sys, time, random, unicodedata, math
import numpy as np
import torch
sys.stdout.reconfigure(line_buffering=True)

# ---- Config ----
EXP_ID = os.environ.get("EXP_ID", "C2")
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/whisper-small")
DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
SEED = 42
LR = float(os.environ.get("LR", "1e-5"))
WARMUP_STEPS = int(os.environ.get("WARMUP_STEPS", "500"))
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "30"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
PATIENCE = int(os.environ.get("PATIENCE", "10"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

token = os.environ.get("HF_TOKEN")
if not DRY_RUN:
    assert token, "HF_TOKEN not set!"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{EXP_ID}: {MODEL_NAME}")
print(f"Device: {device}")
if DRY_RUN:
    print("*** DRY_RUN=1: 4 samples, 2 train steps, 1 eval — local verification only ***")
    MAX_EPOCHS = 1

# ---- Load data ----
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=SEED)
splits = {"train": split2["train"], "dev": split2["test"], "test": split1["test"]}
if DRY_RUN:
    splits = {k: v.select(range(min(4, len(v)))) for k, v in splits.items()}
print(f"Train: {len(splits['train'])}, Dev: {len(splits['dev'])}, Test: {len(splits['test'])}")

def normalize_text(t): return " ".join(unicodedata.normalize("NFC", t.strip()).split())

# ---- Metrics ----
def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(dp[i-1][j-1],dp[i][j-1],dp[i-1][j])
    return dp[n][m]

def compute_cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r,h in zip(refs,hyps))
    total = sum(len(r) for r in refs)
    return round(edits/max(total,1)*100, 2)

def compute_wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r,h in zip(refs,hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits/max(total,1)*100, 2)

# ---- Load model ----
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import librosa

processor = WhisperProcessor.from_pretrained(MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# Configure for Adja
model.config.forced_decoder_ids = None
model.generation_config.forced_decoder_ids = None
model.generation_config.task = "transcribe"
model.generation_config.language = None
model.to(device)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Params: {trainable:,} trainable / {total_params:,} total")

# ---- Dataset ----
class AdjaWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split):
        self.data = hf_split
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = normalize_text(s["text"])
        return {"audio": audio, "text": text}

def collate_whisper(batch, processor):
    audios = [b["audio"] for b in batch]
    texts = [b["text"] for b in batch]

    # Whisper encoder requires exactly 3000 mel frames (30s at 16kHz, 10ms hop).
    # Use padding=True (pads to longest in batch) then manually pad/truncate
    # mel to exactly 3000 frames. Version-independent: works whether the
    # processor auto-pads to 3000 (older transformers) or not (>=4.49).
    #
    # This preserves the short-batch label distribution that the original
    # E4 (CER=24.9%) was trained with. Do NOT use padding="max_length" —
    # it pads raw AUDIO to 30s with silence, producing silence-log-mel values
    # across most of the tensor. Cross-attention then learns to lean on
    # silence encoder states for token emission, which at inference (no
    # teacher forcing) degenerates into repetitive n-gram loops.
    # See docs/whisper-training-gotchas.md section 1.
    input_features = processor(
        audios, sampling_rate=16000, return_tensors="pt", padding=True
    ).input_features

    if input_features.shape[-1] < 3000:
        pad_width = 3000 - input_features.shape[-1]
        input_features = torch.nn.functional.pad(
            input_features, (0, pad_width), value=0.0
        )
    elif input_features.shape[-1] > 3000:
        input_features = input_features[..., :3000]

    labels = processor.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    )
    label_ids = labels.input_ids
    attn_mask = labels.attention_mask

    # Strip BOS token if present (model adds it internally as decoder_start)
    bos_id = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    if bos_id is not None and label_ids.size(1) > 0 and (label_ids[:, 0] == bos_id).all():
        label_ids = label_ids[:, 1:]
        attn_mask = attn_mask[:, 1:]

    # Mask padding positions with -100 using attention_mask.
    # CRITICAL: do NOT mask by `label_ids == pad_token_id` equality — in the
    # Whisper tokenizer, pad_token == eos_token == "<|endoftext|>" (id 50257).
    # Masking every 50257 erases the REAL EOS from training targets, so the
    # model never learns to stop → infinite hallucination loops at inference.
    # Verified empirically: that bug killed both E4v2 (CER=76-90%) and the
    # first E4v3 attempt (loss→0.01, CER=400%+, HYPs showed correct prefix
    # followed by long repetitive token loops). Use attention_mask — it is
    # 1 for real content tokens (including the trailing EOS) and 0 for the
    # post-EOS padding positions.
    label_ids = label_ids.masked_fill(attn_mask == 0, -100)

    return {"input_features": input_features, "labels": label_ids, "texts": texts}

collate = functools.partial(collate_whisper, processor=processor)
train_loader = torch.utils.data.DataLoader(
    AdjaWhisperDataset(splits["train"]), batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, collate_fn=collate)
dev_loader = torch.utils.data.DataLoader(
    AdjaWhisperDataset(splits["dev"]), batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, collate_fn=collate)

# ---- LR schedule ----
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS

def get_lr(step):
    # Linear warmup then linear decay to 0 — matches original E4 schedule
    # (experiments/asr/E4_whisper_ewe_finetune/train.py get_linear_warmup_scheduler).
    if step < WARMUP_STEPS:
        return step / max(WARMUP_STEPS, 1)
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return max(0.0, 1.0 - progress)

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

# ---- Training ----
best_cer = float("inf")
best_epoch = 0
no_improve = 0
history = []
global_step = 0
t_start = time.time()

print(f"\nTraining: {MAX_EPOCHS} epochs, patience={PATIENCE}, lr={LR}")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0; n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        if DRY_RUN and step >= 2:
            break
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        if DRY_RUN and step == 0:
            print(f"  [DRY_RUN] input_features.shape={tuple(input_features.shape)} "
                  f"(expected last dim = 3000)")
            print(f"  [DRY_RUN] labels.shape={tuple(labels.shape)}")

        outputs = model(input_features=input_features, labels=labels)
        loss = outputs.loss

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            continue

        if DRY_RUN:
            print(f"  [DRY_RUN] step={step} loss={loss.item():.4f}")

        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        epoch_loss += loss.item() * GRAD_ACCUM
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)

    # ---- Evaluate ----
    model.eval()
    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch in dev_loader:
            input_features = batch["input_features"].to(device)
            # Match original E4 inference config exactly: greedy, no n-gram
            # suppression. The old no_repeat_ngram_size=3 was a band-aid for
            # the padding="max_length" hallucination bug; with the padding
            # fix in the collate function, it's unneeded and diverges from
            # the reference run.
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=225,
                do_sample=False,
                num_beams=1,
            )
            hyps = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            all_refs.extend(batch["texts"])
            all_hyps.extend([h.strip() for h in hyps])

    dev_cer = compute_cer(all_refs, all_hyps)
    dev_wer = compute_wer(all_refs, all_hyps)

    print(f"Epoch {epoch}/{MAX_EPOCHS} | loss={avg_loss:.4f} | dev_CER={dev_cer}% | dev_WER={dev_wer}%")
    for i in range(min(3, len(all_refs))):
        print(f"  REF: {all_refs[i]}")
        print(f"  HYP: {all_hyps[i]}")

    history.append({"epoch": epoch, "loss": round(avg_loss, 4), "dev_cer": dev_cer, "dev_wer": dev_wer})

    if dev_cer < best_cer:
        best_cer = dev_cer
        best_epoch = epoch
        no_improve = 0
        model.save_pretrained(f"/tmp/best_{EXP_ID}")
        processor.save_pretrained(f"/tmp/best_{EXP_ID}")
        print(f"  ** New best CER: {best_cer}%")
    else:
        no_improve += 1
        print(f"  No improve {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print("Early stopping!")
            break

elapsed = time.time() - t_start
print(f"\nDone! Best CER={best_cer}% at epoch {best_epoch}, time={elapsed/60:.1f}min")

if DRY_RUN:
    print("\n*** DRY_RUN complete — skipping Hub upload ***")
    print("Checklist for DRY_RUN pass:")
    print("  [ ] input_features.shape last dim == 3000")
    print("  [ ] loss is finite (not NaN/Inf)")
    print("  [ ] HYP strings are non-empty and not infinite loops of same n-gram")
    sys.exit(0)

# ---- Push results ----
from huggingface_hub import HfApi
api = HfApi(token=token)
try: api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except: pass

results = {
    "experiment": EXP_ID, "model": MODEL_NAME,
    "best_cer": best_cer, "best_wer": dev_wer, "best_epoch": best_epoch,
    "total_time_min": round(elapsed/60, 1), "history": history,
}
api.upload_file(
    path_or_fileobj=json.dumps(results, indent=2).encode(),
    path_in_repo=f"{EXP_ID}/metrics.json",
    repo_id=RESULTS_REPO, token=token,
)

try:
    api.upload_folder(
        folder_path=f"/tmp/best_{EXP_ID}",
        path_in_repo=f"{EXP_ID}/best_model",
        repo_id=RESULTS_REPO, token=token,
    )
except Exception as e:
    print(f"Model upload: {e}")

print(f"Results pushed to {RESULTS_REPO}/{EXP_ID}/")
