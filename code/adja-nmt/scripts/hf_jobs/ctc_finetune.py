# /// script
# dependencies = ["transformers<4.50", "torch==2.5.1", "torchaudio==2.5.1", "datasets", "soundfile", "numpy", "huggingface-hub"]
# ///
"""
Unified CTC fine-tuning for Adja ASR.
Self-contained for HuggingFace Jobs.

Supports: MMS (C3/E1), XLS-R (C4), wav2vec2 (C5)
Controlled by environment variables:
  EXP_ID      = C3 | E1 | C4 | C5
  MODEL_NAME  = facebook/mms-1b-all | facebook/wav2vec2-xls-r-300m | facebook/wav2vec2-large-960h
  TARGET_LANG = fra | ewe | (empty)
"""
from __future__ import annotations
import json, os, sys, time, random, unicodedata
import numpy as np
import torch
import torch.nn.functional as F
sys.stdout.reconfigure(line_buffering=True)

# ---- Config from env ----
EXP_ID = os.environ.get("EXP_ID", "C3")
MODEL_NAME = os.environ.get("MODEL_NAME", "facebook/mms-1b-all")
TARGET_LANG = os.environ.get("TARGET_LANG", "fra")
DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
SEED = 42
LR = 3e-5
WARMUP_STEPS = 200
MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
PATIENCE = int(os.environ.get("PATIENCE", "20"))
MAX_AUDIO_SEC = 30.0

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{EXP_ID}: {MODEL_NAME} (adapter={TARGET_LANG})")
print(f"Device: {device}")

# ---- Load data ----
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=SEED)
splits = {"train": split2["train"], "dev": split2["test"], "test": split1["test"]}
print(f"Train: {len(splits['train'])}, Dev: {len(splits['dev'])}, Test: {len(splits['test'])}")

# ---- Build CTC vocabulary ----
def normalize_text(t): return " ".join(unicodedata.normalize("NFC", t.strip()).split())

all_texts = [normalize_text(s["text"]) for s in ds]
chars = set()
for t in all_texts: chars.update(t)
vocab = sorted(chars)
char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
for i, c in enumerate(vocab, start=len(char2idx)): char2idx[c] = i
idx2char = {v: k for k, v in char2idx.items()}
vocab_size = len(char2idx)
print(f"Vocab: {vocab_size} tokens")

# ---- Load model ----
from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

load_kwargs = dict(
    vocab_size=vocab_size,
    ignore_mismatched_sizes=True,
    ctc_loss_reduction="mean",
    pad_token_id=char2idx["<pad>"],
    ctc_zero_infinity=True,
)
if TARGET_LANG:
    load_kwargs["target_lang"] = TARGET_LANG
    print(f"Loading adapter: {TARGET_LANG}")

model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME, **load_kwargs)
# Ensure config.vocab_size matches our actual vocab (fixes Ewe adapter
# which has vocab_size=55, smaller than our Adja vocab of 115)
model.config.vocab_size = vocab_size
model.freeze_feature_encoder()
# Gradient checkpointing: trades compute for VRAM (critical for 1B models)
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")
model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Params: {trainable:,} trainable / {total_params:,} total")

# ---- Dataset & DataLoader ----
import soundfile as sf
import torchaudio

class AdjaDataset(torch.utils.data.Dataset):
    def __init__(self, hf_split, max_samples=None):
        self.data = hf_split
        if max_samples: self.data = self.data.select(range(min(max_samples, len(self.data))))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        s = self.data[idx]
        audio = np.array(s["audio"]["array"], dtype=np.float32)
        sr = s["audio"]["sampling_rate"]
        if sr != 16000:
            audio = torchaudio.functional.resample(torch.tensor(audio), sr, 16000).numpy()
        text = normalize_text(s["text"])
        tokens = [char2idx.get(c, char2idx["<unk>"]) for c in text]
        return {"audio": audio, "tokens": tokens, "text": text}

def collate(batch):
    max_tlen = max(len(b["tokens"]) for b in batch)
    # Pass variable-length audio arrays (NOT pre-padded) to feature_extractor
    # so it computes correct attention masks
    raw_audios = [b["audio"] for b in batch]
    targets = torch.full((len(batch), max_tlen), -100, dtype=torch.long)
    target_lens = []
    texts = []
    for i, b in enumerate(batch):
        t = b["tokens"]
        targets[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        target_lens.append(len(t))
        texts.append(b["text"])
    inputs = feature_extractor(raw_audios, sampling_rate=16000, return_tensors="pt", padding=True)
    return {
        "input_values": inputs.input_values,
        "attention_mask": inputs.get("attention_mask"),
        "targets": targets,
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
        "texts": texts,
    }

train_loader = torch.utils.data.DataLoader(AdjaDataset(splits["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
dev_loader = torch.utils.data.DataLoader(AdjaDataset(splits["dev"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)

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

def cer(refs, hyps):
    edits = sum(edit_distance(list(r), list(h)) for r,h in zip(refs,hyps))
    total = sum(len(r) for r in refs)
    return round(edits/max(total,1)*100, 2)

def wer(refs, hyps):
    edits = sum(edit_distance(r.split(), h.split()) for r,h in zip(refs,hyps))
    total = sum(len(r.split()) for r in refs)
    return round(edits/max(total,1)*100, 2)

def greedy_decode(logits, lengths):
    preds = torch.argmax(logits, dim=-1)
    decoded = []
    for i in range(preds.size(0)):
        seq = preds[i, :lengths[i]].tolist()
        collapsed = []
        prev = None
        for t in seq:
            if t != prev:
                if t != 0:  # skip blank
                    collapsed.append(t)
                prev = t
        text = "".join(idx2char.get(t, "") for t in collapsed if idx2char.get(t, "") not in ("<blank>","<pad>","<unk>"))
        decoded.append(text)
    return decoded

# ---- Training ----
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
total_steps = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS
best_cer = float("inf")
best_epoch = 0
no_improve = 0
history = []
t_start = time.time()

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0; n_batches = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        iv = batch["input_values"].to(device)
        am = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None
        targets = batch["targets"].to(device)

        # Use model's built-in CTC loss — pass labels directly.
        # The model handles: log-softmax, input length computation,
        # label padding (-100 → ignore), and CTC alignment.
        # This is the approach from the official HF fine-tuning guide.
        outputs = model(input_values=iv, attention_mask=am, labels=targets)
        loss = outputs.loss

        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            continue

        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item() * GRAD_ACCUM
        n_batches += 1

    avg_loss = epoch_loss / max(n_batches, 1)

    # ---- Evaluate ----
    model.eval()
    all_refs, all_hyps = [], []
    with torch.no_grad():
        for batch in dev_loader:
            iv = batch["input_values"].to(device)
            am = batch["attention_mask"].to(device) if batch["attention_mask"] is not None else None
            outputs = model(input_values=iv, attention_mask=am)
            if hasattr(model, "_get_feat_extract_output_lengths"):
                input_lens = am.sum(dim=-1).long() if am is not None else torch.full((iv.shape[0],), iv.shape[1], dtype=torch.long)
                out_lens = model._get_feat_extract_output_lengths(input_lens).long().clamp(max=outputs.logits.shape[1])
            else:
                out_lens = torch.full((outputs.logits.shape[0],), outputs.logits.shape[1], dtype=torch.long)
            hyps = greedy_decode(outputs.logits.cpu(), out_lens.cpu())
            all_refs.extend(batch["texts"])
            all_hyps.extend(hyps)

    dev_cer = cer(all_refs, all_hyps)
    dev_wer = wer(all_refs, all_hyps)

    print(f"Epoch {epoch}/{MAX_EPOCHS} | loss={avg_loss:.4f} | dev_CER={dev_cer}% | dev_WER={dev_wer}%")

    # Samples
    for i in range(min(3, len(all_refs))):
        print(f"  REF: {all_refs[i]}")
        print(f"  HYP: {all_hyps[i]}")

    history.append({"epoch": epoch, "loss": round(avg_loss, 4), "dev_cer": dev_cer, "dev_wer": dev_wer})

    if dev_cer < best_cer:
        best_cer = dev_cer
        best_epoch = epoch
        no_improve = 0
        # Save best model + feature extractor + CTC vocab (needed for LM decoding)
        save_dir = f"/tmp/best_{EXP_ID}"
        model.save_pretrained(save_dir)
        feature_extractor.save_pretrained(save_dir)
        with open(f"{save_dir}/ctc_vocab.json", "w", encoding="utf-8") as vf:
            json.dump(char2idx, vf, ensure_ascii=False)
        print(f"  ** New best CER: {best_cer}%")
    else:
        no_improve += 1
        print(f"  No improve {no_improve}/{PATIENCE}")
        if no_improve >= PATIENCE:
            print("Early stopping!")
            break

elapsed = time.time() - t_start
print(f"\nDone! Best CER={best_cer}% at epoch {best_epoch}, time={elapsed/60:.1f}min")

# ---- Push results ----
from huggingface_hub import HfApi
api = HfApi(token=token)
try: api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except: pass

results = {
    "experiment": EXP_ID, "model": MODEL_NAME, "adapter": TARGET_LANG,
    "best_cer": best_cer, "best_epoch": best_epoch,
    "total_time_min": round(elapsed/60, 1), "history": history,
    "train_samples": len(splits["train"]), "dev_samples": len(splits["dev"]),
}
api.upload_file(
    path_or_fileobj=json.dumps(results, indent=2).encode(),
    path_in_repo=f"{EXP_ID}/metrics.json",
    repo_id=RESULTS_REPO, token=token,
)

# Push best model
try:
    api.upload_folder(
        folder_path=f"/tmp/best_{EXP_ID}",
        path_in_repo=f"{EXP_ID}/best_model",
        repo_id=RESULTS_REPO, token=token,
    )
except Exception as e:
    print(f"Model upload: {e}")

print(f"Results pushed to {RESULTS_REPO}/{EXP_ID}/")
