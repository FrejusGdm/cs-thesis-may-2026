# /// script
# dependencies = ["transformers==4.49.0", "torch==2.5.1", "torchaudio==2.5.1", "datasets", "soundfile", "librosa", "numpy", "huggingface-hub"]
# ///
"""
LM-decoding comparison (D4) for trained Adja ASR models.

Applies char n-gram LM via shallow fusion (score = AM + α·LM + β·word_count).
Pure-Python ARPA parser and LM scoring — no kenlm/pyctcdecode dependency
(those require build tools not available on HF Jobs default runner).

Shallow fusion formula matches Whisper-LM paper
(https://arxiv.org/abs/2503.23542):
    combined_score = AM_logprob + alpha * LM_logprob + beta * word_count

Two DECODER modes:
- ctc     : Wav2Vec2ForCTC + N-best CTC paths + LM rescoring
- whisper : Whisper beam search + LM rescoring

Env vars (see plan for details):
  EXP_ID, MODEL_REPO, MODEL_PATH, DECODER, LM_REPO, LM_PATH,
  ALPHA_GRID, BETA_GRID, NUM_BEAMS
"""
from __future__ import annotations
import itertools
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

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
EXP_ID = os.environ.get("EXP_ID", "D4_unknown_lm")
MODEL_REPO = os.environ.get("MODEL_REPO", "JosueG/adja-asr-results")
MODEL_PATH = os.environ.get("MODEL_PATH", "C4v2/best_model")
DECODER = os.environ.get("DECODER", "ctc").lower()
LM_REPO = os.environ.get("LM_REPO", "JosueG/adja-asr-results")
LM_PATH = os.environ.get("LM_PATH", "lm/char_5gram.arpa")
ALPHA_GRID = [float(x) for x in os.environ.get("ALPHA_GRID", "0.3,0.5,0.7,1.0").split(",")]
BETA_GRID = [float(x) for x in os.environ.get("BETA_GRID", "0.0,0.5,1.0,2.0").split(",")]
NUM_BEAMS = int(os.environ.get("NUM_BEAMS", "10"))
DATASET_ID = "JosueG/adja-tts-orpheus"
RESULTS_REPO = "JosueG/adja-asr-results"
SEED = 42

token = os.environ.get("HF_TOKEN")
assert token, "HF_TOKEN not set!"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== {EXP_ID} ===")
print(f"Model: {MODEL_REPO}/{MODEL_PATH}")
print(f"LM:    {LM_REPO}/{LM_PATH}")
print(f"Decoder: {DECODER} | Device: {device}")
print(f"α grid: {ALPHA_GRID}")
print(f"β grid: {BETA_GRID}")

# ------------------------------------------------------------
# Metrics (self-contained; same as metrics.py)
# ------------------------------------------------------------
_PUNCT_PATTERN = r"""[.,!?;:()\[\]"'«»“”‘’]"""

def normalize_for_wer(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(_PUNCT_PATTERN, "", text)
    return " ".join(text.split())


def _edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = dp[i - 1][j - 1] if ref[i - 1] == hyp[j - 1] else (
                1 + min(dp[i - 1][j - 1], dp[i][j - 1], dp[i - 1][j])
            )
    return dp[n][m]


def wer(refs, hyps, normalize=False):
    edits = total = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r = normalize_for_wer(r)
            h = normalize_for_wer(h)
        rw = r.split()
        hw = h.split()
        edits += _edit_distance(rw, hw)
        total += len(rw)
    return round(edits / max(total, 1) * 100, 2)


def cer(refs, hyps, normalize=False):
    edits = total = 0
    for r, h in zip(refs, hyps):
        if normalize:
            r = normalize_for_wer(r)
            h = normalize_for_wer(h)
        edits += _edit_distance(list(r), list(h))
        total += len(r)
    return round(edits / max(total, 1) * 100, 2)

# ------------------------------------------------------------
# Pure-Python ARPA parser + scorer (replaces kenlm)
# ------------------------------------------------------------
class ArpaLM:
    """Minimal ARPA n-gram LM scorer in pure Python.

    Handles standard ARPA format:
        \\N-grams:
        <logprob>\t<token1 token2 ... tokenN>[\t<backoff_logprob>]
    Supports scoring a full sequence with Katz backoff-style logic
    (simplified; good enough for rescoring).
    """
    def __init__(self, arpa_path: str):
        self.order = 0
        # ngrams[n] = {tuple of tokens: (logprob, backoff)}
        self.ngrams = defaultdict(dict)
        self._parse(arpa_path)

    def _parse(self, path):
        print(f"Parsing ARPA: {path}")
        with open(path, "r", encoding="utf-8") as f:
            section = None
            current_n = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Section markers like "\data\", "\1-grams:", "\end\"
                m = re.match(r"\\(\d+)-grams:", line)
                if m:
                    current_n = int(m.group(1))
                    self.order = max(self.order, current_n)
                    continue
                if line.startswith("\\") or line.startswith("ngram "):
                    continue
                # n-gram line
                if current_n > 0:
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    try:
                        logprob = float(parts[0])
                    except ValueError:
                        continue
                    tokens = tuple(parts[1].split())
                    backoff = 0.0
                    if len(parts) >= 3:
                        try:
                            backoff = float(parts[2])
                        except ValueError:
                            backoff = 0.0
                    self.ngrams[current_n][tokens] = (logprob, backoff)
        print(f"  order: {self.order}")
        for n in range(1, self.order + 1):
            print(f"  {n}-grams: {len(self.ngrams[n])}")

    def score_ngram(self, ngram: tuple) -> float:
        """Score a single n-gram with backoff. Returns log10 probability."""
        n = len(ngram)
        if n == 0:
            return -99.0
        # Try to find the exact n-gram
        if ngram in self.ngrams[n]:
            return self.ngrams[n][ngram][0]
        # Backoff: shorter context
        if n > 1:
            # Get backoff weight from (n-1)-gram prefix
            prefix = ngram[:-1]
            bo = 0.0
            if prefix in self.ngrams[n - 1]:
                bo = self.ngrams[n - 1][prefix][1]
            # Recurse with shorter suffix
            return bo + self.score_ngram(ngram[1:])
        # Unknown unigram
        if ("<unk>",) in self.ngrams[1]:
            return self.ngrams[1][("<unk>",)][0]
        return -99.0

    def score_sequence(self, tokens: list[str]) -> float:
        """Score a full sequence of tokens. Returns sum of log10 probs."""
        if not tokens:
            return -99.0
        # Prepend <s> and append </s> for context
        full = ["<s>"] + list(tokens) + ["</s>"]
        total = 0.0
        for i in range(1, len(full)):
            ngram = tuple(full[max(0, i - self.order + 1): i + 1])
            total += self.score_ngram(ngram)
        return total

    def score_chars(self, text: str) -> float:
        """Score a char-level string. Converts spaces to | (word boundary)."""
        if not text:
            return -99.0
        tokens = []
        for c in text:
            tokens.append("|" if c == " " else c)
        return self.score_sequence(tokens)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
print("\nLoading dataset...")
from datasets import load_dataset
ds = load_dataset(DATASET_ID, token=token, split="train")
split1 = ds.train_test_split(test_size=0.1, seed=SEED)
split2 = split1["train"].train_test_split(test_size=0.1 / 0.9, seed=SEED)
splits = {"dev": split2["test"], "test": split1["test"]}
print(f"Dev: {len(splits['dev'])}, Test: {len(splits['test'])}")

def norm_ref(t):
    return " ".join(unicodedata.normalize("NFC", t.strip()).split())

# ------------------------------------------------------------
# Download LM and trained model
# ------------------------------------------------------------
from huggingface_hub import hf_hub_download, snapshot_download

print("\nDownloading LM ARPA...")
lm_arpa_path = hf_hub_download(repo_id=LM_REPO, filename=LM_PATH, token=token)
lm = ArpaLM(lm_arpa_path)

print(f"\nDownloading model snapshot from {MODEL_REPO}/{MODEL_PATH}...")
model_local_dir = snapshot_download(
    repo_id=MODEL_REPO,
    allow_patterns=[f"{MODEL_PATH}/*"],
    token=token,
)
model_dir = os.path.join(model_local_dir, MODEL_PATH)
print(f"  model dir: {model_dir}")

import librosa

# ------------------------------------------------------------
# Shared: score candidate text with LM (shallow fusion helper)
# ------------------------------------------------------------
def lm_score_text(text: str) -> float:
    """Log10 probability of a hypothesis under the char LM."""
    return lm.score_chars(text.strip())

# ============================================================
# CTC branch — greedy baseline + LM-rescore N-best from top-K paths
# ============================================================
if DECODER == "ctc":
    from transformers import Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor

    # Fallback: the original training scripts saved model.safetensors + config.json
    # but NOT the feature extractor or ctc_vocab.json. Load the feature extractor
    # from the base pre-trained model (same architecture, same audio preprocessing).
    base_model = os.environ.get("BASE_MODEL_NAME", "facebook/wav2vec2-xls-r-300m")
    try:
        feat = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        print(f"Loaded feature extractor from model dir")
    except OSError:
        print(f"No preprocessor_config.json in model dir; falling back to {base_model}")
        feat = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

    model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device).eval()

    # Load CTC vocab. If ctc_vocab.json missing from model dir, rebuild from the
    # dataset (same logic as ctc_finetune.py).
    vocab_json = os.path.join(model_dir, "ctc_vocab.json")
    if os.path.exists(vocab_json):
        with open(vocab_json, "r", encoding="utf-8") as f:
            char2idx = json.load(f)
    else:
        print(f"No ctc_vocab.json; rebuilding from dataset (same logic as training)")
        all_chars = set()
        for s in ds:
            all_chars.update(" ".join(unicodedata.normalize("NFC", s["text"].strip()).split()))
        vocab_list = sorted(all_chars)
        char2idx = {"<blank>": 0, "<pad>": 1, "<unk>": 2}
        for i, c in enumerate(vocab_list, start=len(char2idx)):
            char2idx[c] = i
    idx2char = {int(v): k for k, v in char2idx.items()}
    labels = [idx2char.get(i, "") for i in range(len(idx2char))]
    special = {"<blank>", "<pad>", "<unk>"}
    print(f"CTC vocab: {len(labels)} tokens")

    def ctc_greedy(logits: np.ndarray) -> str:
        pred = logits.argmax(axis=-1)
        out, prev = [], None
        for p in pred:
            if p != prev:
                ch = labels[p]
                if ch and ch not in special:
                    out.append(ch)
                prev = p
        return "".join(out)

    def ctc_nbest(logits: np.ndarray, k: int) -> list[tuple[float, str]]:
        """Simple N-best: for each frame take top-k probs, expand prefix beam.
        Not a true CTC beam search (ignores blank collapsing subtleties) but
        gives meaningful diversity for rescoring on short utterances."""
        T, V = logits.shape
        # Log-softmax over vocab per frame
        logprobs = logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True) + 1e-12)
        # Beam = list of (cumlogp, prev_idx, collapsed_text)
        beam = [(0.0, -1, "")]
        for t in range(T):
            lp_t = logprobs[t]
            top_k_idx = np.argsort(-lp_t)[:k]
            new_beam = []
            for cumlp, prev, txt in beam:
                for vi in top_k_idx:
                    ch = labels[vi]
                    new_text = txt
                    if vi != prev:
                        if ch and ch not in special:
                            new_text = txt + ch
                    new_beam.append((cumlp + float(lp_t[vi]), int(vi), new_text))
            # Keep top k by cumulative score
            new_beam.sort(key=lambda x: -x[0])
            beam = new_beam[:k]
        # Dedupe identical hypotheses, keep highest AM score
        seen = {}
        for cumlp, _, txt in beam:
            if txt not in seen or seen[txt] < cumlp:
                seen[txt] = cumlp
        return sorted([(sc, tx) for tx, sc in seen.items()], key=lambda x: -x[0])

    def decode_ctc(split_data, alpha, beta, use_lm):
        refs, hyps = [], []
        for sample in split_data:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            text = norm_ref(sample["text"])

            inputs = feat(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            iv = inputs.input_values.to(device)
            am = inputs.get("attention_mask")
            am = am.to(device) if am is not None else None
            with torch.no_grad():
                logits = model(input_values=iv, attention_mask=am).logits[0].cpu().numpy()

            if not use_lm:
                hyp = ctc_greedy(logits)
            else:
                candidates = ctc_nbest(logits, k=NUM_BEAMS)
                if not candidates:
                    hyp = ""
                else:
                    best = max(
                        candidates,
                        key=lambda x: x[0] + alpha * lm_score_text(x[1]) + beta * len(x[1].split()),
                    )
                    hyp = best[1]
            refs.append(text)
            hyps.append(hyp)
        return refs, hyps

    decode_split = decode_ctc

# ============================================================
# Whisper branch — beam search + LM rescoring
# ============================================================
elif DECODER == "whisper":
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    # If the fine-tuned dir doesn't have the processor files, fall back to the
    # base pre-trained Whisper (tokenizer/feature-extractor aren't fine-tuned).
    base_model = os.environ.get("BASE_MODEL_NAME", "openai/whisper-small")
    try:
        processor = WhisperProcessor.from_pretrained(model_dir)
        print(f"Loaded Whisper processor from model dir")
    except (OSError, Exception) as e:
        print(f"Processor not in model dir ({type(e).__name__}); falling back to {base_model}")
        processor = WhisperProcessor.from_pretrained(base_model)

    model = WhisperForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.task = "transcribe"
    model.generation_config.language = None

    def decode_whisper(split_data, alpha, beta, use_lm):
        refs, hyps = [], []
        for sample in split_data:
            audio = np.array(sample["audio"]["array"], dtype=np.float32)
            sr = sample["audio"]["sampling_rate"]
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            text = norm_ref(sample["text"])

            iv = processor(
                audio, sampling_rate=16000, return_tensors="pt",
                padding="max_length", truncation=True,
            ).input_features.to(device)

            if not use_lm:
                with torch.no_grad():
                    out = model.generate(iv, max_new_tokens=128, no_repeat_ngram_size=3, num_beams=1)
                hyp = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
            else:
                with torch.no_grad():
                    out = model.generate(
                        iv, max_new_tokens=128, no_repeat_ngram_size=3,
                        num_beams=NUM_BEAMS, num_return_sequences=NUM_BEAMS,
                        return_dict_in_generate=True, output_scores=True,
                    )
                seqs = out.sequences
                am_scores = out.sequences_scores.tolist() if hasattr(out, "sequences_scores") else [0.0] * len(seqs)
                texts = processor.batch_decode(seqs, skip_special_tokens=True)
                best_txt, best_score = "", -1e18
                for txt, am_s in zip(texts, am_scores):
                    txt = txt.strip()
                    lms = lm_score_text(txt)
                    wc = len(txt.split())
                    s = am_s + alpha * lms + beta * wc
                    if s > best_score:
                        best_score = s
                        best_txt = txt
                hyp = best_txt
            refs.append(text)
            hyps.append(hyp)
        return refs, hyps

    decode_split = decode_whisper

else:
    raise ValueError(f"Unknown DECODER: {DECODER}")

# ------------------------------------------------------------
# Greedy baseline
# ------------------------------------------------------------
print("\n=== Greedy baseline ===")
t0 = time.time()
dev_refs, dev_hyps_greedy = decode_split(splits["dev"], 0, 0, use_lm=False)
test_refs, test_hyps_greedy = decode_split(splits["test"], 0, 0, use_lm=False)
greedy_time = time.time() - t0
greedy_metrics = {
    "dev_cer": cer(dev_refs, dev_hyps_greedy),
    "dev_wer": wer(dev_refs, dev_hyps_greedy),
    "dev_cer_norm": cer(dev_refs, dev_hyps_greedy, normalize=True),
    "dev_wer_norm": wer(dev_refs, dev_hyps_greedy, normalize=True),
    "test_cer": cer(test_refs, test_hyps_greedy),
    "test_wer": wer(test_refs, test_hyps_greedy),
    "test_cer_norm": cer(test_refs, test_hyps_greedy, normalize=True),
    "test_wer_norm": wer(test_refs, test_hyps_greedy, normalize=True),
    "inference_time_sec": round(greedy_time, 1),
}
print(f"  Greedy dev  CER={greedy_metrics['dev_cer']}%, WER={greedy_metrics['dev_wer']}% | normalized: CER={greedy_metrics['dev_cer_norm']}%, WER={greedy_metrics['dev_wer_norm']}%")
print(f"  Greedy test CER={greedy_metrics['test_cer']}%, WER={greedy_metrics['test_wer']}% | normalized: CER={greedy_metrics['test_cer_norm']}%, WER={greedy_metrics['test_wer_norm']}%")

# ------------------------------------------------------------
# LM α/β sweep on dev
# ------------------------------------------------------------
print("\n=== LM α/β sweep (dev) ===")
sweep = []
for alpha, beta in itertools.product(ALPHA_GRID, BETA_GRID):
    t0 = time.time()
    _, dev_hyps_lm = decode_split(splits["dev"], alpha, beta, use_lm=True)
    dt = time.time() - t0
    c = cer(dev_refs, dev_hyps_lm)
    w = wer(dev_refs, dev_hyps_lm)
    cn = cer(dev_refs, dev_hyps_lm, normalize=True)
    wn = wer(dev_refs, dev_hyps_lm, normalize=True)
    entry = {
        "alpha": alpha, "beta": beta,
        "dev_cer": c, "dev_wer": w,
        "dev_cer_norm": cn, "dev_wer_norm": wn,
        "time_sec": round(dt, 1),
    }
    sweep.append(entry)
    print(f"  α={alpha} β={beta} → CER={c}% WER={w}% (norm: {cn}%/{wn}%) in {dt:.0f}s")

# Best by dev CER → run on test
best = min(sweep, key=lambda e: e["dev_cer"])
print(f"\nBest config: α={best['alpha']} β={best['beta']} (dev CER={best['dev_cer']}%)")
_, test_hyps_lm = decode_split(splits["test"], best["alpha"], best["beta"], use_lm=True)
lm_test_metrics = {
    "alpha": best["alpha"], "beta": best["beta"],
    "test_cer": cer(test_refs, test_hyps_lm),
    "test_wer": wer(test_refs, test_hyps_lm),
    "test_cer_norm": cer(test_refs, test_hyps_lm, normalize=True),
    "test_wer_norm": wer(test_refs, test_hyps_lm, normalize=True),
}
print(f"  LM test CER={lm_test_metrics['test_cer']}%, WER={lm_test_metrics['test_wer']}% | normalized: CER={lm_test_metrics['test_cer_norm']}%, WER={lm_test_metrics['test_wer_norm']}%")

# Sample pairs for qualitative inspection
sample_n = min(20, len(test_refs))
samples = [
    {"ref": test_refs[i], "hyp_greedy": test_hyps_greedy[i], "hyp_lm": test_hyps_lm[i]}
    for i in range(sample_n)
]

# ------------------------------------------------------------
# Push results
# ------------------------------------------------------------
results = {
    "experiment": EXP_ID,
    "model_repo": MODEL_REPO,
    "model_path": MODEL_PATH,
    "lm_path": LM_PATH,
    "decoder": DECODER,
    "num_beams": NUM_BEAMS,
    "greedy": greedy_metrics,
    "lm_sweep": sweep,
    "lm_best_on_test": lm_test_metrics,
    "samples": samples,
}

from huggingface_hub import HfApi
api = HfApi(token=token)
try:
    api.create_repo(RESULTS_REPO, private=True, exist_ok=True)
except Exception as e:
    print(f"Repo creation: {e}")
api.upload_file(
    path_or_fileobj=json.dumps(results, indent=2, ensure_ascii=False).encode(),
    path_in_repo=f"{EXP_ID}/metrics.json",
    repo_id=RESULTS_REPO,
    token=token,
)
print(f"\nResults pushed to {RESULTS_REPO}/{EXP_ID}/metrics.json")
