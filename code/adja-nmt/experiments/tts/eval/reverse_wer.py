#!/usr/bin/env python3
"""
Reverse-WER auto-evaluation for Adja TTS.

We rank TTS quality by transcribing every synthesized WAV with an ASR model
for the matching language, then comparing the transcription to the original
prompt. Lower reverse-WER -> more intelligible synthesized speech.

Pipeline:
    JosueG/adja-tts-results/<run>/
        generated_audio/*.wav
        metrics.json   (contains the prompt-text per WAV)
                |
                v
    ASR backend (default: local, --backend=serverless to opt in)
        local      : transformers.from_pretrained() once, then inference per WAV
        serverless : HF Serverless Inference API (NOT supported for our custom
                     ASR models — HF only auto-deploys popular base models.
                     Errors with "Model not supported by provider hf-inference"
                     verified 2026-04-28 on the freshly published E4v4 repo.)
                |
                v
    experiments/asr/shared/metrics.py: evaluate(refs, hyps)
                |
                v
    results/reverse-wer/<run>__<asr_label>.json
    results/reverse-wer/summary.json

Each (run, asr) pair is scored TWICE:
  - raw      : exact-string WER/CER (conservative)
  - normalized: NFC + lowercase + strip ASCII punctuation, preserves Adja
                special chars and tone marks. Use for cross-run ranking.

Cache:
  Inference responses are cached to results/reverse-wer/.cache/<asr-label>/
  keyed by sha256(audio_bytes). Re-runs are free as long as the WAV bytes
  and ASR repo don't change.

Usage:
    # Dry-run: validates auth + plumbing on a single sentence/sample.
    python experiments/tts/eval/reverse_wer.py --dry-run

    # Full eval over the surviving Adja TTS runs against E4v4 + C4v2.
    python experiments/tts/eval/reverse_wer.py \
        --runs CF1_ewc_stage2,CF2_curriculum_stage2,CF3_frozen_backbone_stage2,T3A_spark_ewe_adja_direct \
        --adja-asrs JosueG/whisper-ewe-adja-e4v4,JosueG/wav2vec2-xlsr-adja-c4v2

    # Sanity check on real Adja audio (test-set slice). Each ASR's reverse-CER
    # should land near its published test-set CER (E4v4: ~37%, C4v2: ~25%).
    python experiments/tts/eval/reverse_wer.py --sanity --n 20

CLAUDE.md alignment:
  - Local --dry-run mode (1 sample, 1 ASR call) before any batch action.
  - Reuses experiments/asr/shared/metrics.py (NFC + Adja-safe normalization).
  - All inputs come from `JosueG/adja-tts-results` (Hub), no local data deps.
  - Updates results/reverse-wer/ with raw JSON per (run, ASR).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(line_buffering=True)

# Local utilities — must import via path because experiments/asr/shared isn't
# a package. Done before any heavy imports for fast --help.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "asr" / "shared"))

from metrics import compute_cer, compute_wer  # noqa: E402

# --- Constants ---------------------------------------------------------------

# `JosueG/adja-tts-results` exists as BOTH a dataset and a model repo (HF
# allows the same id under different types). The model-side has all the
# 2026-04-18 → 2026-04-22 runs (T1, T2, T3, T6, T7, T10, T11 — 31 scoreable
# runs, ~150 wavs); the dataset-side has the newer 2026-04-26 runs (CF1,
# CF2, CF3, T3A — 4 scoreable runs, 20 wavs). We try both.
TTS_RESULTS_REPO = "JosueG/adja-tts-results"
TTS_RESULTS_TYPES = ("dataset", "model")

ADJA_TEST_DATASET = "JosueG/adja-tts-orpheus"  # for --sanity mode

# Default ASR registry. Each entry = (repo_id, label, language).
# E4_v5 will appear here once the resubmit job (id 69f0f1c8d2c8bd8662bd235c)
# completes and we publish it as a model repo.
DEFAULT_ADJA_ASRS = [
    ("JosueG/whisper-ewe-adja-e4v4", "e4v4", "aj"),
    ("JosueG/wav2vec2-xlsr-adja-c4v2", "c4v2", "aj"),
]

# The runs surviving in JosueG/adja-tts-results with both audio + metrics.json
# (verified 2026-04-28). Older runs referenced in results/tts-comparison.md
# (T1, T2, original T3-120step, etc.) are NOT publicly archived on the Hub.
# Default scope = every run on either side of JosueG/adja-tts-results that
# has both audio + metrics.json. Discovery routine below auto-fills this
# when --runs is omitted; the explicit list here is a fallback that pins
# the canonical 35-run audit (verified 2026-04-29).
DEFAULT_ADJA_RUNS = [
    # dataset side (4 runs — 2026-04-26 Gbe-cascade)
    "CF1_ewc_stage2",
    "CF2_curriculum_stage2",
    "CF3_frozen_backbone_stage2",
    "T3A_spark_ewe_adja_direct",
    # model side (31 runs — 2026-04-18 → 2026-04-22 breadth sweep)
    "T1",
    "T1_csm_ewe_adja_stage2",
    "T1_csm_ewe_stage1",
    "T1_csm_tokfix",
    "T1_diagnostic",
    "T1_full_ft_20ep_earlystop_2026-04-18",
    "T1_long_20ep_earlystop_2026-04-18",
    "T2_diagnostic_2026-04-18_v2",
    "T2_orpheus_en_ewe_adja_stage2",
    "T2_orpheus_en_ewe_adja_stage2_mixed",
    "T2_orpheus_en_ewe_stage1_inference",
    "T2_orpheus_en_fullft_10ep_2026-04-18",
    "T2_orpheus_en_lora_r128_20ep_2026-04-18",
    "T2_orpheus_en_lora_r32_20ep_2026-04-18",
    "T2_orpheus_en_lora_r64_20ep_2026-04-18",
    "T2_orpheus_fr_ewe_adja_stage2",
    "T2_orpheus_fr_lora_r64_20ep_2026-04-18",
    "T2_orpheus_zh_ewe_stage1_inference",
    "T3",
    "T3_spark_120steps_2026-04-18",
    "T3_spark_20ep_earlystop_2026-04-18",
    "T3_spark_ewe_stage1_inference",
    "T3_spark_ewe_stage1_inference_l40s",
    "T6_mms_ewe_20ep_2026-04-18",
    "T7_voxcpm_smoke_audio_retry2_2026-04-19-011425",
    "T10_f5_2026-04-19",
    "T10_f5_full_2026-04-21_l40s",
    "T10_f5_smoke_2026-04-19",
    "T11_e2_2026-04-19",
    "T11_e2_full_2026-04-21_l40s",
    "T11_e2_smoke_2026-04-19",
]

# HF serverless API: 503 on cold start, 429 on rate-limit. Retry policy.
RETRY_BACKOFF_SECONDS = (5, 15, 30, 60)  # ~110s total worst case
COLD_START_BACKOFF = 65  # explicit wait if HF returns "model is loading"


# --- Data structures ---------------------------------------------------------


@dataclass
class TTSExample:
    """One TTS-generated sample with its ground-truth prompt text."""
    run: str
    file: str            # filename inside generated_audio/
    text: str            # the prompt that was synthesized
    duration_sec: float | None = None


@dataclass
class TranscriptionResult:
    asr_label: str
    asr_repo: str
    text: str            # ASR hypothesis
    latency_sec: float = 0.0
    cached: bool = False
    error: str | None = None


@dataclass
class ScoredPair:
    example: TTSExample
    transcription: TranscriptionResult
    wer_raw: float
    cer_raw: float
    wer_norm: float
    cer_norm: float


# --- Text normalization (re-imported here to keep this driver self-contained) -


_PUNCT_TO_STRIP = r"""[.,!?;:()\[\]"'«»“”‘’]"""


def normalize_for_wer(text: str) -> str:
    """NFC + lowercase + strip ASCII punctuation + collapse whitespace.

    Preserves Adja-meaningful characters (ɛ, ɔ, ŋ, ɖ, tone marks, hyphens).
    Mirrors `experiments/asr/shared/metrics.py:normalize_for_wer`.
    """
    import re
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(_PUNCT_TO_STRIP, "", text)
    return " ".join(text.split())


# --- Hub IO -----------------------------------------------------------------


def _try_hub_download(run: str, sub_path: str) -> tuple[str, str] | None:
    """Try fetching `<run>/<sub_path>` across both repo types. Returns
    `(local_path, repo_type)` or None if not found in either."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
    for rt in TTS_RESULTS_TYPES:
        try:
            return hf_hub_download(TTS_RESULTS_REPO, f"{run}/{sub_path}", repo_type=rt), rt
        except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
            continue
        except Exception as e:
            # Treat 404-from-API as not-found, anything else re-raises
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            raise
    return None


def fetch_run_examples(run: str) -> list[TTSExample]:
    """Download metrics.json + audio for one TTS run from the Hub.

    Tries both `dataset` and `model` repo types under `JosueG/adja-tts-results`.
    Newer Stage-2 runs (CF1-3, T3A) live in the dataset repo; older
    2026-04 runs (T1, T2, T3, T6, T7, T10, T11) live in the model repo.

    Returns a list of TTSExample with text + on-disk WAV path.
    """
    metrics_pull = _try_hub_download(run, "metrics.json")
    if metrics_pull is None:
        print(f"  SKIP {run}: no metrics.json on Hub")
        return []
    metrics_path, repo_type = metrics_pull
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    if "generated" not in metrics or not isinstance(metrics["generated"], list):
        # Older runs (e.g. T6 MMS-Ewe) only saved training config, not the
        # synthesized text↔audio map. Without prompts we can't reverse-WER.
        print(f"  SKIP {run}: metrics.json has no 'generated' list")
        return []

    examples = []
    for entry in metrics["generated"]:
        if not isinstance(entry, dict) or "file" not in entry:
            print(f"  WARN {run}: malformed generated entry, skipping: {entry!r}")
            continue
        # Schema variants seen across runs:
        #   {"text": ..., "file": ...}                       (default)
        #   {"target_text": ..., "reference_text": ...}      (CSM speaker-conditioned)
        #   {"prompt": ..., "file": ...}                     (some Spark runs)
        ref_text = (
            entry.get("text")
            or entry.get("target_text")
            or entry.get("prompt")
        )
        if not ref_text:
            print(f"  WARN {run}: no reference text in entry {entry!r}, skipping")
            continue
        wav_pull = _try_hub_download(run, f"generated_audio/{entry['file']}")
        if wav_pull is None:
            print(f"  WARN {run}/{entry['file']}: not found in either repo type, skipping")
            continue
        local_path, _ = wav_pull
        examples.append(TTSExample(
            run=run,
            file=local_path,
            text=unicodedata.normalize("NFC", ref_text.strip()),
            duration_sec=entry.get("duration_sec"),
        ))
    return examples


def fetch_real_adja_test_slice(n: int) -> list[TTSExample]:
    """Pull `n` real Adja-speech test samples for sanity-check mode.

    Uses the SAME 80/10/10 split (seed=42) as training so we hit the held-out
    test set. Mirrors scripts/hf_jobs/whisper_finetune.py:48-50.
    """
    from datasets import load_dataset

    token = os.environ.get("HF_TOKEN")
    ds = load_dataset(ADJA_TEST_DATASET, token=token, split="train")
    split1 = ds.train_test_split(test_size=0.1, seed=42)
    test = split1["test"].select(range(min(n, len(split1["test"]))))

    # Materialize each row's audio to a tmp WAV so the InferenceClient can
    # pass it to the API. soundfile handles the array → bytes conversion.
    import soundfile as sf
    tmp_dir = Path("/tmp/adja-reverse-wer-sanity")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    for i, row in enumerate(test):
        audio = row["audio"]
        wav_path = tmp_dir / f"sanity_{i:03d}.wav"
        sf.write(wav_path, audio["array"], audio["sampling_rate"])
        examples.append(TTSExample(
            run="__sanity__real_adja__",
            file=str(wav_path),
            text=unicodedata.normalize("NFC", row["text"].strip()),
        ))
    return examples


# --- Inference client wrapper -----------------------------------------------


class CachingASRClient:
    """Calls HF Serverless Inference API with on-disk cache + retry/backoff.

    Cache layout:
      <cache_dir>/<asr_label>/<sha256-of-audio-bytes>.json
        { "asr_repo": ..., "wav_path": ..., "text": ..., "latency_sec": ... }
    """

    def __init__(self, asr_repo: str, asr_label: str, cache_dir: Path):
        """`asr_repo` may be either a HF model repo id (`org/name`) or a
        dedicated Inference Endpoint URL (`https://...endpoints.huggingface.cloud`).

        We default to dedicated endpoints because HF's free Serverless
        Inference API does NOT auto-deploy our custom Adja ASR repos
        (verified 2026-04-28: returned "Model not supported by provider
        hf-inference"). Spin endpoints up via `scripts/hub/manage_endpoints.py up`.

        For dedicated endpoints we use `requests.post` directly so the
        `Content-Type: audio/wav` header is set — `InferenceClient` against
        a raw URL omits the header and the endpoint rejects with
        `Content type "None" not supported`.
        """
        from huggingface_hub import InferenceClient, get_token
        self.asr_repo = asr_repo
        self.asr_label = asr_label
        self.cache_dir = cache_dir / asr_label
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or get_token()
        self.token = token
        self.is_endpoint = asr_repo.startswith("http")
        if self.is_endpoint:
            import requests as _requests
            self._requests = _requests
            self.endpoint_url = asr_repo.rstrip("/")
            self.client = None
        else:
            self.client = InferenceClient(
                model=asr_repo, token=token, timeout=180,
                provider="hf-inference",
            )

    @staticmethod
    def _audio_hash(wav_path: str) -> str:
        h = hashlib.sha256()
        with open(wav_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def transcribe(self, wav_path: str) -> TranscriptionResult:
        h = self._audio_hash(wav_path)
        cached_path = self.cache_dir / f"{h}.json"
        if cached_path.exists():
            with open(cached_path, encoding="utf-8") as f:
                payload = json.load(f)
            return TranscriptionResult(
                asr_label=self.asr_label,
                asr_repo=self.asr_repo,
                text=payload["text"],
                latency_sec=payload.get("latency_sec", 0.0),
                cached=True,
            )

        # Live call, with retry on 503/429.
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()

        last_err: Exception | None = None
        t0 = time.time()
        for attempt, backoff in enumerate([0, *RETRY_BACKOFF_SECONDS]):
            if backoff:
                time.sleep(backoff)
            try:
                if self.is_endpoint:
                    # Dedicated endpoint — POST raw WAV bytes with explicit Content-Type.
                    r = self._requests.post(
                        self.endpoint_url,
                        data=audio_bytes,
                        headers={
                            "Authorization": f"Bearer {self.token}" if self.token else "",
                            "Content-Type": "audio/wav",
                            "Accept": "application/json",
                        },
                        timeout=180,
                    )
                    if r.status_code == 503 or "loading" in r.text.lower():
                        raise RuntimeError(f"503 cold-start: {r.text[:200]}")
                    r.raise_for_status()
                    payload_raw = r.json()
                    if isinstance(payload_raw, dict):
                        hyp = payload_raw.get("text", "")
                    elif isinstance(payload_raw, list) and payload_raw:
                        # Some pipelines return [{"text": ...}, ...]
                        hyp = payload_raw[0].get("text", "") if isinstance(payload_raw[0], dict) else str(payload_raw[0])
                    else:
                        hyp = str(payload_raw)
                else:
                    resp = self.client.automatic_speech_recognition(audio_bytes)
                    hyp = (
                        resp.text
                        if hasattr(resp, "text")
                        else resp.get("text", "") if isinstance(resp, dict)
                        else str(resp)
                    )
                latency = time.time() - t0
                payload = {
                    "asr_repo": self.asr_repo,
                    "wav_path": wav_path,
                    "text": hyp,
                    "latency_sec": round(latency, 2),
                }
                with open(cached_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                return TranscriptionResult(
                    asr_label=self.asr_label,
                    asr_repo=self.asr_repo,
                    text=hyp,
                    latency_sec=latency,
                )
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                last_err = e
                # Cold-start: the API tells us "model is loading, ETA Xs"
                if "loading" in msg or "503" in msg or "cold-start" in msg:
                    print(f"  {self.asr_label}: cold start (attempt {attempt + 1}), waiting {COLD_START_BACKOFF}s")
                    time.sleep(COLD_START_BACKOFF)
                else:
                    print(f"  {self.asr_label}: error (attempt {attempt + 1}): {str(e)[:200]}")

        return TranscriptionResult(
            asr_label=self.asr_label,
            asr_repo=self.asr_repo,
            text="",
            latency_sec=time.time() - t0,
            error=str(last_err) if last_err else "unknown",
        )


# --- Scoring ----------------------------------------------------------------


def _clean_hyp(text: str) -> str:
    """Strip literal CTC special-token markers that some Wav2Vec2 endpoints
    emit as text (e.g. "<pad>", "<unk>", "<blank>") because their tokenizer
    didn't register those as `additional_special_tokens`. Also collapses
    repeated whitespace introduced by removing those markers.
    """
    import re
    if not text:
        return ""
    text = re.sub(r"<(pad|unk|blank)>", "", text)
    return " ".join(text.split())


def score_example(example: TTSExample, tr: TranscriptionResult) -> ScoredPair:
    refs = [example.text]
    hyps = [_clean_hyp(tr.text)]
    wer_raw = compute_wer(refs, hyps, normalize=False)["wer"]
    cer_raw = compute_cer(refs, hyps, normalize=False)["cer"]
    wer_norm = compute_wer(refs, hyps, normalize=True)["wer"]
    cer_norm = compute_cer(refs, hyps, normalize=True)["cer"]
    return ScoredPair(
        example=example,
        transcription=tr,
        wer_raw=wer_raw,
        cer_raw=cer_raw,
        wer_norm=wer_norm,
        cer_norm=cer_norm,
    )


def aggregate(pairs: list[ScoredPair]) -> dict[str, Any]:
    """Compute corpus-level WER/CER (sum-of-edits / sum-of-refs).

    Uses _clean_hyp() so the corpus aggregate matches the per-utterance
    scoring shape (otherwise literal "<pad>" tokens emitted by the C4v2
    endpoint inflate the aggregate to >300%).
    """
    if not pairs:
        return {}
    refs = [p.example.text for p in pairs]
    hyps = [_clean_hyp(p.transcription.text) for p in pairs]
    wer_raw = compute_wer(refs, hyps, normalize=False)["wer"]
    cer_raw = compute_cer(refs, hyps, normalize=False)["cer"]
    wer_norm = compute_wer(refs, hyps, normalize=True)["wer"]
    cer_norm = compute_cer(refs, hyps, normalize=True)["cer"]
    # Median is robust to Whisper hallucination loops (single bad utterance
    # can push corpus WER >500% via repetitive infinite-token outputs).
    sorted_wers = sorted(p.wer_norm for p in pairs)
    sorted_cers = sorted(p.cer_norm for p in pairs)
    mid = len(pairs) // 2
    median_wer = (
        sorted_wers[mid] if len(pairs) % 2 == 1
        else (sorted_wers[mid - 1] + sorted_wers[mid]) / 2
    )
    median_cer = (
        sorted_cers[mid] if len(pairs) % 2 == 1
        else (sorted_cers[mid - 1] + sorted_cers[mid]) / 2
    )

    return {
        "n": len(pairs),
        "wer_raw": wer_raw,
        "cer_raw": cer_raw,
        "wer_norm": wer_norm,
        "cer_norm": cer_norm,
        "median_utt_wer_norm": round(median_wer, 2),
        "median_utt_cer_norm": round(median_cer, 2),
        "mean_utt_wer_norm": round(
            sum(p.wer_norm for p in pairs) / len(pairs), 2,
        ),
        "errors": sum(1 for p in pairs if p.transcription.error),
    }


def serialize_pair(p: ScoredPair) -> dict[str, Any]:
    return {
        "file": Path(p.example.file).name,
        "text": p.example.text,
        "hyp": p.transcription.text,
        "asr_label": p.transcription.asr_label,
        "asr_repo": p.transcription.asr_repo,
        "wer_raw": p.wer_raw,
        "wer_norm": p.wer_norm,
        "cer_raw": p.cer_raw,
        "cer_norm": p.cer_norm,
        "cached": p.transcription.cached,
        "error": p.transcription.error,
        "duration_sec": p.example.duration_sec,
    }


# --- CLI driver --------------------------------------------------------------


def parse_asr_specs(spec: str) -> list[tuple[str, str]]:
    """Parse '<repo-or-url>[:<label>],...' into [(target, label), ...].

    Handles two input formats per item:
      * ``org/repo:label``               → HF repo id (serverless)
      * ``https://...cloud:label``       → dedicated Inference Endpoint URL

    URLs already contain colons (`https:`) — the rule is: use the LAST `:`
    for label-splitting, treating the rest as the target. No colon at all =
    derive label from the repo's last path component.
    """
    out = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # If the chunk contains '://' it's a URL — split on the last ':' so
        # 'https://foo.bar:lbl' → target='https://foo.bar', label='lbl'.
        if "://" in chunk:
            target, _, label = chunk.rpartition(":")
            if not label or "/" in label:  # no explicit label, take whole url
                target, label = chunk, chunk.rstrip("/").split("/")[-1]
        elif ":" in chunk:
            target, label = chunk.split(":", 1)
        else:
            target = chunk
            label = chunk.split("/")[-1]
        out.append((target.strip(), label.strip()))
    return out


def run_eval(
    runs: list[str],
    asr_pairs: list[tuple[str, str]],
    out_dir: Path,
    sanity: bool = False,
    sanity_n: int = 20,
    dry_run: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / ".cache"

    if sanity:
        examples_by_run = {"__sanity__real_adja__": fetch_real_adja_test_slice(sanity_n)}
    elif dry_run:
        # First sentence of the first run, against the first ASR only.
        first_run = runs[0]
        examples = fetch_run_examples(first_run)[:1]
        examples_by_run = {first_run: examples}
        asr_pairs = asr_pairs[:1]
    else:
        examples_by_run = {}
        for r in runs:
            try:
                got = fetch_run_examples(r)
                if got:
                    examples_by_run[r] = got
            except Exception as e:
                print(f"  ERROR {r}: {e!s}; continuing")

    summary: dict[str, dict[str, Any]] = {}

    for run, examples in examples_by_run.items():
        print(f"\n=== Run: {run} ({len(examples)} samples) ===")
        for asr_repo, asr_label in asr_pairs:
            print(f"  ASR: {asr_label}  ({asr_repo})")
            client = CachingASRClient(asr_repo, asr_label, cache_dir)
            pairs: list[ScoredPair] = []
            for ex in examples:
                tr = client.transcribe(ex.file)
                pair = score_example(ex, tr)
                pairs.append(pair)
                cached_marker = "[cache]" if tr.cached else "[live] "
                print(
                    f"    {cached_marker} {Path(ex.file).name:>30} | "
                    f"WER_raw={pair.wer_raw:6.2f} | WER_norm={pair.wer_norm:6.2f}"
                    f" | CER_norm={pair.cer_norm:6.2f}"
                    + (f"  err={tr.error[:60]}" if tr.error else "")
                )
            agg = aggregate(pairs)
            run_payload = {
                "run": run,
                "asr_repo": asr_repo,
                "asr_label": asr_label,
                "aggregate": agg,
                "per_utterance": [serialize_pair(p) for p in pairs],
            }
            out_file = out_dir / f"{run}__{asr_label}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(run_payload, f, ensure_ascii=False, indent=2)
            summary.setdefault(run, {})[asr_label] = agg
            print(
                f"    -> WER_raw={agg.get('wer_raw'):6.2f}  WER_norm={agg.get('wer_norm'):6.2f}"
                f"  CER_norm={agg.get('cer_norm'):6.2f}  ({out_file.name})"
            )

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary -> {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runs",
        default=",".join(DEFAULT_ADJA_RUNS),
        help="Comma-separated TTS run folder names under JosueG/adja-tts-results/. "
             f"Default: {','.join(DEFAULT_ADJA_RUNS)}",
    )
    p.add_argument(
        "--adja-asrs",
        default=",".join(f"{r}:{lbl}" for r, lbl, _ in DEFAULT_ADJA_ASRS),
        help="Comma-separated <repo>[:<label>] pairs for Adja ASR(s) used to "
             "transcribe TTS audio. Each repo MUST be a public HF model repo "
             "compatible with the serverless automatic-speech-recognition pipeline.",
    )
    p.add_argument(
        "--out", default="results/reverse-wer",
        help="Output directory (default: results/reverse-wer/)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Single-sample plumbing check: validates auth + JSON shape against "
             "1 ASR on 1 sentence. No batch usage.",
    )
    p.add_argument(
        "--sanity", action="store_true",
        help="Sanity check on REAL Adja test audio (--n samples). Each ASR's "
             "reverse-CER should land near its published test CER.",
    )
    p.add_argument("--n", type=int, default=20, help="Sanity-mode sample count.")
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    asr_pairs = parse_asr_specs(args.adja_asrs)
    out_dir = Path(args.out)

    run_eval(
        runs=runs,
        asr_pairs=asr_pairs,
        out_dir=out_dir,
        sanity=args.sanity,
        sanity_n=args.n,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
