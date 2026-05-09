# /// script
# dependencies = ["transformers==4.49.0", "torch==2.5.1", "torchaudio==2.5.1", "datasets", "soundfile", "librosa", "numpy", "huggingface-hub", "optuna"]
# ///
"""
LM-decoding with Optuna hyperparameter search (smarter than grid search).

Same shallow-fusion formula as decode_with_lm.py:
    score = AM_logprob + alpha * LM_logprob + beta * word_count

Difference: use Optuna's Tree-structured Parzen Estimator (TPE) to
explore (alpha, beta) space more efficiently. Typically reaches the
optimum in ~20-30 trials where grid search might need 50+ configs
for equivalent coverage.

Inspired by:
- hitz-zentroa/whisper-lm (uses Optuna)
- Reddit post by MarkoMarjamaa: grid search found WER=6.62,
  Optuna sweep on same LM found WER=3.34 (Finnish CV23)

Env vars:
  EXP_ID, MODEL_REPO, MODEL_PATH, DECODER, LM_REPO, LM_PATH
  BASE_MODEL_NAME   fallback when processor/feature_extractor missing
  N_TRIALS          (default 30) number of Optuna trials
  ALPHA_MIN/MAX     (default 0.0/3.0)
  BETA_MIN/MAX      (default -2.0/4.0)
  NUM_BEAMS         (default 10)
"""
from __future__ import annotations
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict

import numpy as np
sys.stdout.reconfigure(line_buffering=True)

import torch
import optuna

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
EXP_ID = os.environ.get("EXP_ID", "D4_unknown_lm_optuna")
MODEL_REPO = os.environ.get("MODEL_REPO", "JosueG/adja-asr-results")
MODEL_PATH = os.environ.get("MODEL_PATH", "C4v2/best_model")
DECODER = os.environ.get("DECODER", "ctc").lower()
LM_REPO = os.environ.get("LM_REPO", "JosueG/adja-asr-results")
LM_PATH = os.environ.get("LM_PATH", "lm/char_5gram.arpa")
N_TRIALS = int(os.environ.get("N_TRIALS", "30"))
ALPHA_MIN = float(os.environ.get("ALPHA_MIN", "0.0"))
ALPHA_MAX = float(os.environ.get("ALPHA_MAX", "3.0"))
BETA_MIN = float(os.environ.get("BETA_MIN", "-2.0"))
BETA_MAX = float(os.environ.get("BETA_MAX", "4.0"))
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "10"))
DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
SEED = 42

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== {EXP_ID} ===")
print(f"Model: {MODEL_REPO}/{MODEL_PATH} | Decoder: {DECODER} | Device: {device}")
print(f"Optuna: n_trials={N_TRIALS}, α∈[{ALPHA_MIN},{ALPHA_MAX}], β∈[{BETA_MIN},{BETA_MAX}]")

# ------------------------------------------------------------
# Shared utils (metrics, LM scoring) — copied from decode_with_lm.py
# ------------------------------------------------------------
_PUNCT = r"""[.,!?;:()\[\]"'«»“”‘’]"""

def normalize_for_wer(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(_PUNCT, "", text)
    return " ".join(text.split())

def _edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): dp[i][0] = i
    for j in range(m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i-1][j-1] if ref[i-1]==hyp[j-1] else 1+min(dp[i-1][j-1], dp[i][j-1], dp[i-1][j])
    return dp[n][m]

def wer(refs, hyps, normalize=False):
    edits = total = 0
    for r, h in zip(refs, hyps):
        if normalize: r = normalize_for_wer(r); h = normalize_for_wer(h)
        rw, hw = r.split(), h.split()
        edits += _edit_distance(rw, hw); total += len(rw)
    return round(edits / max(total, 1) * 100, 2)

def cer(refs, hyps, normalize=False):
    edits = total = 0
    for r, h in zip(refs, hyps):
        if normalize: r = normalize_for_wer(r); h = normalize_for_wer(h)
        edits += _edit_distance(list(r), list(h)); total += len(r)
    return round(edits / max(total, 1) * 100, 2)

# ------------------------------------------------------------
# Pure-Python ARPA scorer
# ------------------------------------------------------------
class ArpaLM:
    def __init__(self, path):
        self.order = 0
        self.ngrams = defaultdict(dict)
        self._parse(path)

    def _parse(self, path):
        print(f"Parsing ARPA: {path}")
        with open(path, "r", encoding="utf-8") as f:
            current_n = 0
            for line in f:
                line = line.strip()
                if not line: continue
                m = re.match(r"\\(\d+)-grams:", line)
                if m:
                    current_n = int(m.group(1))
                    self.order = max(self.order, current_n); continue
                if line.startswith("\\") or line.startswith("ngram "): continue
                if current_n > 0:
                    parts = line.split("\t")
                    if len(parts) < 2: continue
                    try: lp = float(parts[0])
                    except: continue
                    toks = tuple(parts[1].split())
                    bo = float(parts[2]) if len(parts) >= 3 else 0.0
                    self.ngrams[current_n][toks] = (lp, bo)
        print(f"  order: {self.order}, {sum(len(v) for v in self.ngrams.values())} total n-grams")

    def score_ngram(self, ngram):
        n = len(ngram)
        if n == 0: return -99.0
        if ngram in self.ngrams[n]: return self.ngrams[n][ngram][0]
        if n > 1:
            prefix = ngram[:-1]
            bo = self.ngrams[n-1][prefix][1] if prefix in self.ngrams[n-1] else 0.0
            return bo + self.score_ngram(ngram[1:])
        if ("<unk>",) in self.ngrams[1]: return self.ngrams[1][("<unk>",)][0]
        return -99.0

    def score_chars(self, text):
        if not text: return -99.0
        toks = ["<s>"] + ["|" if c == " " else c for c in text] + ["</s>"]
        total = 0.0
        for i in range(1, len(toks)):
            ngram = tuple(toks[max(0, i - self.order + 1): i + 1])
            total += self.score_ngram(ngram)
        return total

# ------------------------------------------------------------
# Load data + downloads
# ------------------------------------------------------------
print("Loading dataset...")
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1/0.9, seed=SEED)
splits = {"dev": split2["test"], "test": split1["test"]}
print(f"Dev: {len(splits['dev'])}, Test: {len(splits['test'])}")

def norm_ref(t):
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())

from huggingface_hub import hf_hub_download, snapshot_download
print("Downloading LM ARPA...")
lm_arpa_path = hf_hub_download(repo_id=LM_REPO, filename=LM_PATH, token=token)
lm = ArpaLM(lm_arpa_path)

print(f"Downloading model from {MODEL_REPO}/{MODEL_PATH}...")
model_local_dir = snapshot_download(
    repo_id=MODEL_REPO, allow_patterns=[f"{MODEL_PATH}/*"], token=token,
)
model_dir = os.path.join(model_local_dir, MODEL_PATH)

import librosa

def lm_score_text(text: str) -> float:
    return lm.score_chars(text.strip())

# ------------------------------------------------------------
# CTC branch
# ------------------------------------------------------------
if DECODER == "ctc":
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor
    base = os.environ.get("BASE_MODEL_NAME", "facebook/wav2vec2-xls-r-300m")
    try: feat = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
    except OSError: feat = Wav2Vec2FeatureExtractor.from_pretrained(base)
    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()

    vocab_json = os.path.join(model_dir, "ctc_vocab.json")
    if os.path.exists(vocab_json):
        with open(vocab_json) as f: char2idx = json.load(f)
    else:
        print("Rebuilding ctc_vocab from dataset")
        chars = set()
        for s in ds:
            chars.update(" ".join(unicodedata.normalize("NFC", s["text"].strip()).split()))
        vl = sorted(chars)
        char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
        for i, c in enumerate(vl, start=len(char2idx)): char2idx[c] = i
    idx2char = {int(v): k for k, v in char2idx.items()}
    labels = [idx2char.get(i, "") for i in range(len(idx2char))]
    special = {"<blank>", "<pad>", "<unk>"}

    # Precompute logits per dev utterance ONCE (Optuna evaluates same audio many times)
    print("Precomputing dev logits...")
    dev_cache = []
    for sample in splits["dev"]:
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = norm_ref(sample["text"])
        inputs = feat(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        am = inputs.get("attention_mask")
        am = am.to(device) if am is not None else None
        with torch.no_grad():
            logits = model(input_values=iv, attention_mask=am).logits[0].cpu().numpy()
        dev_cache.append((text, logits))
    print(f"  {len(dev_cache)} dev utterances cached")

    def ctc_greedy(logits):
        pred = logits.argmax(-1); out = []; prev = None
        for p in pred:
            if p != prev:
                ch = labels[p]
                if ch and ch not in special: out.append(ch)
                prev = p
        return "".join(out)

    def ctc_nbest(logits, k):
        T, V = logits.shape
        lp = logits - np.log(np.exp(logits).sum(-1, keepdims=True) + 1e-12)
        beam = [(0.0, -1, "")]
        for t in range(T):
            top = np.argsort(-lp[t])[:k]
            nb = []
            for cl, prev, txt in beam:
                for vi in top:
                    new_txt = txt
                    ch = labels[vi]
                    if vi != prev:
                        if ch and ch not in special: new_txt = txt + ch
                    nb.append((cl + float(lp[t][vi]), int(vi), new_txt))
            nb.sort(key=lambda x: -x[0])
            beam = nb[:k]
        seen = {}
        for cl, _, tx in beam:
            if tx not in seen or seen[tx] < cl: seen[tx] = cl
        return sorted([(s, t) for t, s in seen.items()], key=lambda x: -x[0])

    def decode_dev_with_lm(alpha, beta):
        hyps = []; refs = []
        for text, logits in dev_cache:
            cands = ctc_nbest(logits, NUM_BEAMS)
            if not cands: hyps.append(""); refs.append(text); continue
            best = max(cands, key=lambda x: x[0] + alpha * lm_score_text(x[1]) + beta * len(x[1].split()))
            hyps.append(best[1]); refs.append(text)
        return refs, hyps

    def decode_test_with_lm(alpha, beta):
        refs = []; hyps = []
        for sample in splits["test"]:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            text = norm_ref(sample["text"])
            inputs = feat(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inputs.input_values.to(device)
            am = inputs.get("attention_mask")
            am = am.to(device) if am is not None else None
            with torch.no_grad():
                logits = model(input_values=iv, attention_mask=am).logits[0].cpu().numpy()
            cands = ctc_nbest(logits, NUM_BEAMS)
            if not cands: hyps.append(""); refs.append(text); continue
            best = max(cands, key=lambda x: x[0] + alpha * lm_score_text(x[1]) + beta * len(x[1].split()))
            hyps.append(best[1]); refs.append(text)
        return refs, hyps

    # Greedy baseline on dev + test
    print("\n=== Greedy baseline ===")
    t0 = time.time()
    dev_refs_g = [t for t, _ in dev_cache]
    dev_hyps_g = [ctc_greedy(lg) for _, lg in dev_cache]
    test_refs_g = []; test_hyps_g = []
    for sample in splits["test"]:
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        if sr != 16000: audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        text = norm_ref(sample["text"])
        inputs = feat(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        iv = inputs.input_values.to(device)
        am = inputs.get("attention_mask")
        am = am.to(device) if am is not None else None
        with torch.no_grad():
            logits = model(input_values=iv, attention_mask=am).logits[0].cpu().numpy()
        test_refs_g.append(text); test_hyps_g.append(ctc_greedy(logits))
    greedy_time = time.time() - t0
    greedy_metrics = {
        "dev_cer": cer(dev_refs_g, dev_hyps_g),
        "dev_wer": wer(dev_refs_g, dev_hyps_g),
        "dev_cer_norm": cer(dev_refs_g, dev_hyps_g, normalize=True),
        "dev_wer_norm": wer(dev_refs_g, dev_hyps_g, normalize=True),
        "test_cer": cer(test_refs_g, test_hyps_g),
        "test_wer": wer(test_refs_g, test_hyps_g),
        "test_cer_norm": cer(test_refs_g, test_hyps_g, normalize=True),
        "test_wer_norm": wer(test_refs_g, test_hyps_g, normalize=True),
        "inference_time_sec": round(greedy_time, 1),
    }
    print(f"  Greedy dev CER={greedy_metrics['dev_cer']}%, WER={greedy_metrics['dev_wer']}%")
    print(f"  Greedy test CER={greedy_metrics['test_cer']}%, WER={greedy_metrics['test_wer']}%")

    # Optuna objective: minimize dev CER
    def objective(trial):
        alpha = trial.suggest_float("alpha", ALPHA_MIN, ALPHA_MAX)
        beta = trial.suggest_float("beta", BETA_MIN, BETA_MAX)
        refs, hyps = decode_dev_with_lm(alpha, beta)
        return cer(refs, hyps)

    print(f"\n=== Optuna sweep ({N_TRIALS} trials) ===")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"{EXP_ID}_lm_tune",
    )
    t0 = time.time()
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    sweep_time = time.time() - t0
    best_alpha = study.best_params["alpha"]
    best_beta = study.best_params["beta"]
    best_dev_cer = study.best_value
    print(f"\nBest: α={best_alpha:.3f} β={best_beta:.3f} dev_CER={best_dev_cer}%")
    print(f"Sweep took {sweep_time:.0f}s")

    print("\nTop 10 trials:")
    trials_sorted = sorted(study.trials, key=lambda t: t.value or 1e9)[:10]
    for i, t in enumerate(trials_sorted):
        if t.value is not None:
            print(f"  #{i+1}: α={t.params['alpha']:.3f} β={t.params['beta']:.3f} dev_CER={t.value}%")

    # Run best on test
    test_refs_lm, test_hyps_lm = decode_test_with_lm(best_alpha, best_beta)
    lm_test = {
        "alpha": best_alpha, "beta": best_beta,
        "test_cer": cer(test_refs_lm, test_hyps_lm),
        "test_wer": wer(test_refs_lm, test_hyps_lm),
        "test_cer_norm": cer(test_refs_lm, test_hyps_lm, normalize=True),
        "test_wer_norm": wer(test_refs_lm, test_hyps_lm, normalize=True),
    }
    print(f"\n=== LM test: CER={lm_test['test_cer']}%, WER={lm_test['test_wer']}% ===")
    print(f"  Normalized: CER={lm_test['test_cer_norm']}%, WER={lm_test['test_wer_norm']}%")

    all_trials = [
        {"alpha": t.params.get("alpha"), "beta": t.params.get("beta"), "dev_cer": t.value}
        for t in study.trials if t.value is not None
    ]
    samples = [{"ref": test_refs_lm[i], "hyp_greedy": test_hyps_g[i], "hyp_lm": test_hyps_lm[i]}
               for i in range(min(20, len(test_refs_lm)))]

else:
    raise NotImplementedError("Whisper Optuna branch not yet implemented; use decode_with_lm.py grid for Whisper")

# ------------------------------------------------------------
# Push results
# ------------------------------------------------------------
results = {
    "experiment": EXP_ID,
    "model_repo": MODEL_REPO,
    "model_path": MODEL_PATH,
    "lm_path": LM_PATH,
    "decoder": DECODER,
    "search": "optuna-tpe",
    "n_trials": N_TRIALS,
    "num_beams": NUM_BEAMS,
    "greedy": greedy_metrics,
    "lm_best_on_test": lm_test,
    "all_trials": all_trials,
    "samples": samples,
}

from huggingface_hub import HfApi
api = HfApi(token=token)
try: api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except Exception as e: print(f"Repo creation: {e}")
api.upload_file(
    path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
    path_in_repo=f"{EXP_ID}/metrics.json",
    repo_id=RESULTS_REPO, token=token,
)
print(f"\nResults pushed to {RESULTS_REPO}/{EXP_ID}/metrics.json")
