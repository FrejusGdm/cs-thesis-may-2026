#!/usr/bin/env python3
"""
Regenerate the qualitative ASR-decode table on real Adja test audio.

Why this exists
---------------
The thesis (Chapter 4) and `results/comparison.md` cite hand-picked decode
samples from C4v2 (XLS-R + CTC) and E4v4 (Whisper-Ewe-Adja) on real Adja
test-set audio. Those samples were exported one-off during training runs
that no longer exist locally. To put a sample table in the paper that the
review committee can reproduce, we need a *deterministic* re-run from the
two deployed ASRs against the held-out test split.

This driver:

    1. Loads `JosueG/adja-tts-orpheus` (the only Adja ASR dataset).
    2. Reproduces the canonical 80/10/10 split with `seed=42` (matches
       `experiments/asr/shared/data_prep.py` and Whisper FT scripts).
    3. Picks N samples from the *test* split, deterministic by index.
    4. Transcribes each sample with each ASR (HF Inference Endpoint URL or
       local `transformers.pipeline` fallback).
    5. Scores each (ref, hyp) with the project's NFC-safe WER/CER and
       writes both a structured JSON and a paper-ready Markdown table to
       `results/qualitative/`.

CLAUDE.md alignment
-------------------
- Local `--dry-run` mode (1 sample, 1 ASR) before any batch usage.
- NFC normalization on every reference and every hypothesis.
- Reuses `experiments/asr/shared/metrics.py` (no copy-paste of WER logic).
- Caches endpoint responses keyed by sha256(audio) so re-runs are free.
- Output goes to `results/qualitative/`; nothing here writes to the Hub.

Usage
-----
    # Smoke test: 1 sample, 1 ASR (C4v2 endpoint).
    python scripts/qualitative/regen_asr_decodes.py --dry-run \\
        --asrs '<C4V2_ENDPOINT_URL>:c4v2'

    # Canonical 20-sample paper table against both deployed endpoints.
    python scripts/qualitative/regen_asr_decodes.py \\
        --n 20 \\
        --asrs '<C4V2_ENDPOINT_URL>:c4v2,<E4V4_ENDPOINT_URL>:e4v4' \\
        --tag 2026-05-03

    # Local fallback (no endpoints up — uses transformers pipeline).
    # Slow on CPU; use only for ad-hoc inspection.
    python scripts/qualitative/regen_asr_decodes.py \\
        --n 5 --backend local \\
        --asrs JosueG/wav2vec2-xlsr-adja-c4v2:c4v2,JosueG/whisper-ewe-adja-e4v4:e4v4

Outputs
-------
    results/qualitative/asr_decodes_<tag>.json     # full structured payload
    results/qualitative/asr_decodes_<tag>.md       # paper-ready table
    results/qualitative/.cache/<asr_label>/*.json  # per-audio response cache
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)

# Reuse shared metrics — keep one definition of WER/CER for the project.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "asr" / "shared"))

from metrics import compute_cer, compute_wer  # noqa: E402

# --- Constants ---------------------------------------------------------------

ADJA_DATASET = "JosueG/adja-tts-orpheus"   # private; ~1.6k Adja utterances
TEST_SEED = 42                             # matches data_prep.py
TEST_SPLIT_FRAC = 0.10                     # 80/10/10 → 10% test

# Default deployed ASRs. The endpoint URLs are filled in via --asrs at call
# time; the registry below is the local-pipeline fallback (much slower).
DEFAULT_LOCAL_ASRS = [
    ("JosueG/wav2vec2-xlsr-adja-c4v2", "c4v2"),
    ("JosueG/whisper-ewe-adja-e4v4", "e4v4"),
]

# CTC special-token markers some Wav2Vec2 endpoints emit literally because
# their tokenizer didn't register them as `additional_special_tokens`.
_CTC_LITERAL_TOKENS = ("<pad>", "<unk>", "<blank>", "<s>", "</s>")

RETRY_BACKOFF_SECONDS = (5, 15, 30)  # for endpoint cold-start + transient 5xx
COLD_START_BACKOFF = 65               # explicit wait if endpoint says "loading"


# --- Data structures ---------------------------------------------------------


@dataclass
class TestSample:
    index: int           # row index in the test split (post-seeded shuffle)
    wav_path: str        # tmp-dir WAV materialized from the HF row
    ref: str             # NFC-normalized ground-truth transcript


@dataclass
class Decode:
    asr_label: str
    asr_target: str      # repo id OR endpoint URL
    hyp: str             # raw ASR output (post _clean_hyp())
    latency_sec: float = 0.0
    cached: bool = False
    error: str | None = None


@dataclass
class Row:
    sample: TestSample
    by_asr: dict[str, Decode] = field(default_factory=dict)


# --- Text helpers -----------------------------------------------------------


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", (text or "").strip())


def _clean_hyp(text: str) -> str:
    """Strip literal CTC marker tokens and collapse whitespace.

    Mirrors `experiments/tts/eval/reverse_wer.py:_clean_hyp` so qualitative
    decodes line up with the reverse-WER aggregate scoring on the same audio.
    """
    if not text:
        return ""
    out = text
    for tok in _CTC_LITERAL_TOKENS:
        out = out.replace(tok, "")
    return " ".join(out.split())


# --- Dataset loading --------------------------------------------------------


def load_test_samples(n: int, tmp_dir: Path) -> list[TestSample]:
    """Reproduce the canonical 80/10/10 (seed=42) split and materialize the
    first `n` test rows as WAV files."""
    from datasets import load_dataset
    import soundfile as sf

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(ADJA_DATASET, token=token, split="train")
    # Match data_prep.py: train_test_split → 90/10 first (test slice), then
    # the train-side gets a second 80/10 split. We only need the first 10%
    # (test) here; that's stable across both code paths.
    split = ds.train_test_split(test_size=TEST_SPLIT_FRAC, seed=TEST_SEED)
    test = split["test"]
    n = min(n, len(test))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for i in range(n):
        row = test[i]
        wav_path = tmp_dir / f"test_{i:03d}.wav"
        if not wav_path.exists():
            sf.write(wav_path, row["audio"]["array"], row["audio"]["sampling_rate"])
        samples.append(TestSample(
            index=i,
            wav_path=str(wav_path),
            ref=_nfc(row["text"]),
        ))
    return samples


# --- ASR clients ------------------------------------------------------------


def _audio_sha256(wav_path: str) -> str:
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class EndpointASR:
    """POST raw WAV bytes to a dedicated HF Inference Endpoint URL.

    We do NOT use `huggingface_hub.InferenceClient` against a raw URL — it
    omits the `Content-Type` header and the endpoint rejects with
    `Content type "None" not supported` (verified 2026-04-28). Same pattern
    as `experiments/tts/eval/reverse_wer.py:CachingASRClient.transcribe`.
    """

    def __init__(self, url: str, label: str, cache_dir: Path):
        import requests
        from huggingface_hub import get_token
        self.url = url.rstrip("/")
        self.label = label
        self.cache_dir = cache_dir / label
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._requests = requests
        self.token = os.environ.get("HF_TOKEN") or get_token()

    def transcribe(self, wav_path: str) -> Decode:
        h = _audio_sha256(wav_path)
        cached_path = self.cache_dir / f"{h}.json"
        if cached_path.exists():
            with open(cached_path, encoding="utf-8") as f:
                payload = json.load(f)
            return Decode(
                asr_label=self.label, asr_target=self.url,
                hyp=_clean_hyp(payload["text"]),
                latency_sec=payload.get("latency_sec", 0.0),
                cached=True,
            )

        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        headers = {
            "Content-Type": "audio/wav",
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_err: Exception | None = None
        t0 = time.time()
        for attempt, backoff in enumerate([0, *RETRY_BACKOFF_SECONDS]):
            if backoff:
                time.sleep(backoff)
            try:
                r = self._requests.post(
                    self.url, data=audio_bytes, headers=headers, timeout=180,
                )
                if r.status_code == 503 or "loading" in r.text.lower():
                    raise RuntimeError(f"503 cold-start: {r.text[:200]}")
                r.raise_for_status()
                body = r.json()
                if isinstance(body, dict):
                    hyp = body.get("text", "")
                elif isinstance(body, list) and body:
                    first = body[0]
                    hyp = first.get("text", "") if isinstance(first, dict) else str(first)
                else:
                    hyp = str(body)
                latency = time.time() - t0
                with open(cached_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "asr_target": self.url,
                        "wav_path": wav_path,
                        "text": hyp,
                        "latency_sec": round(latency, 2),
                    }, f, ensure_ascii=False, indent=2)
                return Decode(
                    asr_label=self.label, asr_target=self.url,
                    hyp=_clean_hyp(hyp), latency_sec=latency,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                if "loading" in msg or "503" in msg or "cold-start" in msg:
                    print(f"  {self.label}: cold start (attempt {attempt + 1}), waiting {COLD_START_BACKOFF}s")
                    time.sleep(COLD_START_BACKOFF)
                else:
                    print(f"  {self.label}: error (attempt {attempt + 1}): {str(e)[:200]}")

        return Decode(
            asr_label=self.label, asr_target=self.url, hyp="",
            latency_sec=time.time() - t0, error=str(last_err) if last_err else "unknown",
        )


class LocalPipelineASR:
    """Fallback path: load the model locally with `transformers.pipeline`.

    Slow on CPU. Use only when the deployed endpoints are down or when
    you want a hermetic re-run with no network calls. Loads the model on
    first `transcribe()` to keep `--help` and dry-run paths fast.
    """

    def __init__(self, repo_id: str, label: str, cache_dir: Path):
        self.repo_id = repo_id
        self.label = label
        self.cache_dir = cache_dir / label
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None  # lazy

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        # `automatic-speech-recognition` works for both Whisper and
        # Wav2Vec2-CTC architectures; the pipeline auto-detects.
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=self.repo_id,
            device=device,
            chunk_length_s=30,    # safe for both archs
        )

    def transcribe(self, wav_path: str) -> Decode:
        h = _audio_sha256(wav_path)
        cached_path = self.cache_dir / f"{h}.json"
        if cached_path.exists():
            with open(cached_path, encoding="utf-8") as f:
                payload = json.load(f)
            return Decode(
                asr_label=self.label, asr_target=self.repo_id,
                hyp=_clean_hyp(payload["text"]),
                latency_sec=payload.get("latency_sec", 0.0),
                cached=True,
            )
        self._ensure_loaded()
        t0 = time.time()
        try:
            result = self._pipe(wav_path)
            hyp = result["text"] if isinstance(result, dict) else str(result)
        except Exception as e:  # noqa: BLE001
            return Decode(
                asr_label=self.label, asr_target=self.repo_id,
                hyp="", latency_sec=time.time() - t0, error=str(e),
            )
        latency = time.time() - t0
        with open(cached_path, "w", encoding="utf-8") as f:
            json.dump({
                "asr_target": self.repo_id,
                "wav_path": wav_path,
                "text": hyp,
                "latency_sec": round(latency, 2),
            }, f, ensure_ascii=False, indent=2)
        return Decode(
            asr_label=self.label, asr_target=self.repo_id,
            hyp=_clean_hyp(hyp), latency_sec=latency,
        )


# --- Spec parsing -----------------------------------------------------------


def parse_asr_specs(spec: str) -> list[tuple[str, str]]:
    """Parse `<repo-or-url>[:<label>],...` into [(target, label), ...].

    URLs already contain `:` (e.g. `https:`) — split on the LAST colon for
    the label. No colon at all → label = repo's last path component.
    Mirrors `experiments/tts/eval/reverse_wer.py:parse_asr_specs`.
    """
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "://" in chunk:
            target, _, label = chunk.rpartition(":")
            if not label or "/" in label:
                target, label = chunk, chunk.rstrip("/").split("/")[-1]
        elif ":" in chunk:
            target, label = chunk.split(":", 1)
        else:
            target, label = chunk, chunk.split("/")[-1]
        out.append((target.strip(), label.strip()))
    return out


def build_clients(
    asr_specs: list[tuple[str, str]],
    backend: str,
    cache_dir: Path,
) -> list[tuple[str, Any]]:
    """Return [(label, client), ...]."""
    clients = []
    for target, label in asr_specs:
        if backend == "endpoint" or target.startswith("http"):
            clients.append((label, EndpointASR(target, label, cache_dir)))
        elif backend == "local":
            clients.append((label, LocalPipelineASR(target, label, cache_dir)))
        else:
            raise ValueError(f"backend={backend!r} not in {{endpoint, local}}")
    return clients


# --- Scoring + emission ------------------------------------------------------


def score_pair(ref: str, hyp: str) -> dict[str, float]:
    """Per-utterance metrics. NFC + lowercase + ASCII-punct normalization."""
    return {
        "wer_raw": compute_wer([ref], [hyp], normalize=False)["wer"],
        "cer_raw": compute_cer([ref], [hyp], normalize=False)["cer"],
        "wer_norm": compute_wer([ref], [hyp], normalize=True)["wer"],
        "cer_norm": compute_cer([ref], [hyp], normalize=True)["cer"],
    }


def aggregate(rows: list[Row], asr_label: str) -> dict[str, float]:
    refs = [r.sample.ref for r in rows if asr_label in r.by_asr]
    hyps = [r.by_asr[asr_label].hyp for r in rows if asr_label in r.by_asr]
    if not refs:
        return {}
    return {
        "n": len(refs),
        "wer_raw": compute_wer(refs, hyps, normalize=False)["wer"],
        "cer_raw": compute_cer(refs, hyps, normalize=False)["cer"],
        "wer_norm": compute_wer(refs, hyps, normalize=True)["wer"],
        "cer_norm": compute_cer(refs, hyps, normalize=True)["cer"],
        "errors": sum(
            1 for r in rows
            if asr_label in r.by_asr and r.by_asr[asr_label].error
        ),
    }


def emit_json(rows: list[Row], asr_labels: list[str], out_path: Path) -> None:
    payload = {
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": ADJA_DATASET,
        "split": "test (10% of seed=42 train_test_split)",
        "asr_labels": asr_labels,
        "samples": [
            {
                "index": r.sample.index,
                "ref": r.sample.ref,
                "decodes": {
                    label: {
                        "hyp": d.hyp,
                        "metrics": score_pair(r.sample.ref, d.hyp),
                        "asr_target": d.asr_target,
                        "latency_sec": round(d.latency_sec, 2),
                        "cached": d.cached,
                        "error": d.error,
                    }
                    for label, d in r.by_asr.items()
                },
            }
            for r in rows
        ],
        "aggregate": {label: aggregate(rows, label) for label in asr_labels},
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def emit_markdown(rows: list[Row], asr_labels: list[str], out_path: Path) -> None:
    """Paper-ready table.

    Columns: index | reference | <label-1 hyp + CER> | <label-2 hyp + CER> ...
    """
    lines = [
        f"# Qualitative ASR decodes — regenerated {_dt.date.today().isoformat()}",
        "",
        f"Source: `{ADJA_DATASET}` test split (seed={TEST_SEED}, "
        f"{int(TEST_SPLIT_FRAC * 100)}% holdout). "
        f"Driver: `scripts/qualitative/regen_asr_decodes.py`.",
        "",
        "## Aggregate (all reported samples)",
        "",
        "| ASR | n | WER (norm) | CER (norm) | errors |",
        "|---|---|---|---|---|",
    ]
    for label in asr_labels:
        agg = aggregate(rows, label)
        if not agg:
            continue
        lines.append(
            f"| `{label}` | {agg['n']} | {agg['wer_norm']:.2f} | "
            f"{agg['cer_norm']:.2f} | {agg['errors']} |"
        )

    lines += [
        "",
        "## Per-sample decodes",
        "",
        "| # | Reference | "
        + " | ".join(f"`{label}` hyp (CER)" for label in asr_labels)
        + " |",
        "|---|---|" + "|".join(["---"] * len(asr_labels)) + "|",
    ]
    for r in rows:
        cells = [str(r.sample.index), _md_escape(r.sample.ref)]
        for label in asr_labels:
            d = r.by_asr.get(label)
            if d is None:
                cells.append("—")
                continue
            if d.error:
                cells.append(f"_error: {_md_escape(d.error[:60])}_")
                continue
            cer = score_pair(r.sample.ref, d.hyp)["cer_norm"]
            cells.append(f"{_md_escape(d.hyp)} ({cer:.2f})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    out_path.write_text("\n".join(lines))


def _md_escape(text: str) -> str:
    """Just enough escaping to keep table cells readable."""
    if not text:
        return "_(empty)_"
    return text.replace("|", "\\|").replace("\n", " ")


# --- Driver -----------------------------------------------------------------


def run(
    asr_specs: list[tuple[str, str]],
    backend: str,
    n: int,
    out_dir: Path,
    tag: str,
    dry_run: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"
    tmp_dir = Path("/tmp/adja-qualitative-decodes")

    if dry_run:
        n = 1
        asr_specs = asr_specs[:1]
        print(f"[dry-run] n=1, asr={asr_specs[0][1]}")

    print(f"Loading {n} test samples from {ADJA_DATASET} (seed={TEST_SEED})...")
    samples = load_test_samples(n, tmp_dir)
    print(f"Loaded {len(samples)} samples.")

    clients = build_clients(asr_specs, backend, cache_dir)
    asr_labels = [label for label, _ in clients]

    rows: list[Row] = []
    for s in samples:
        row = Row(sample=s)
        print(f"\n[{s.index:>3}] REF: {s.ref}")
        for label, client in clients:
            d = client.transcribe(s.wav_path)
            row.by_asr[label] = d
            tag_str = "[cache]" if d.cached else "[live] "
            metrics = score_pair(s.ref, d.hyp)
            err = f"  ERR={d.error[:60]}" if d.error else ""
            print(
                f"  {tag_str} {label:>6}: CER_norm={metrics['cer_norm']:6.2f} "
                f"| {d.hyp[:80]}{err}"
            )
        rows.append(row)

    json_path = out_dir / f"asr_decodes_{tag}.json"
    md_path = out_dir / f"asr_decodes_{tag}.md"
    emit_json(rows, asr_labels, json_path)
    emit_markdown(rows, asr_labels, md_path)

    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    for label in asr_labels:
        agg = aggregate(rows, label)
        if agg:
            print(
                f"  {label}: n={agg['n']} "
                f"WER_norm={agg['wer_norm']:.2f} CER_norm={agg['cer_norm']:.2f} "
                f"errors={agg['errors']}"
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--asrs", required=True,
        help="Comma-separated `<repo-or-url>[:<label>]`. Example: "
             "'https://abc.endpoints.huggingface.cloud:c4v2,JosueG/whisper-ewe-adja-e4v4:e4v4'",
    )
    p.add_argument(
        "--backend", choices=("endpoint", "local"), default="endpoint",
        help="ASR backend. Default 'endpoint' for HF Inference Endpoints. "
             "'local' loads the model with transformers.pipeline (CPU-slow).",
    )
    p.add_argument("--n", type=int, default=20, help="Number of test samples.")
    p.add_argument(
        "--out", default="results/qualitative",
        help="Output directory (default: results/qualitative/).",
    )
    p.add_argument(
        "--tag",
        default=_dt.date.today().isoformat(),
        help="Filename suffix for the JSON/MD pair. Default: today's date.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Single-sample plumbing check (1 sample, 1 ASR). No batch usage.",
    )
    args = p.parse_args()

    asr_specs = parse_asr_specs(args.asrs)
    if not asr_specs:
        print("ERROR: --asrs produced 0 specs", file=sys.stderr)
        sys.exit(2)

    run(
        asr_specs=asr_specs,
        backend=args.backend,
        n=args.n,
        out_dir=Path(args.out),
        tag=args.tag,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
