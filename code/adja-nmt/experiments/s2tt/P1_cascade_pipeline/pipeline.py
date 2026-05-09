#!/usr/bin/env python3
from __future__ import annotations
"""
P1 Cascade Pipeline — Adja speech Q&A via French LLM bridge.
2026-05-02  Added Mode B (--input-text) and OpenRouter response backend.
2026-05-07  Added Mode C (--input-lang fr) — French audio in.

Full pipeline:
  Adja speech → ASR → MT(→FR) → LLM(FR) → MT(→Adja) → TTS → Adja speech

Three input modes
-----------------
  Mode A  --input audio.wav             Adja speech in   [canonical demo]
  Mode B  --input-text "question FR"    French text in   [easiest to test]
  Mode C  --input audio.wav             French speech in [--input-lang fr]

Mode B rationale (2026-05-02):
  Finding a fluent Adja speaker to record test questions is non-trivial.
  Mode B bypasses the ASR + MT→FR legs entirely: you supply the French question
  directly and the pipeline handles LLM → MT→Adja → TTS. This lets you verify
  the MT + TTS legs are working before you ever touch a microphone.

Mode choices
------------
  --mode roundtrip  No LLM. Measures ASR+MT+TTS error accumulation cleanly.
                    With --input-text: FR text → MT→Adja → TTS (no LLM).
  --mode qa         LLM responds in French (via OpenRouter by default).
                    With --input-text: FR text → LLM(FR) → MT→Adja → TTS.

Usage
-----
  # Dry run (stubs, no GPU, no API key):
  python pipeline.py --dry-run --input samples/test.wav

  # Mode B — French question → Adja speech (test without a microphone):
  OPENROUTER_API_KEY=sk-or-... python pipeline.py \\
    --mode qa \\
    --input-text "Quel est le plat traditionnel adja ?" \\
    --output-dir runs/test_mode_b/

  # Mode B roundtrip — test MT+TTS from a known French sentence:
  python pipeline.py \\
    --mode roundtrip \\
    --input-text "Le dolo est une boisson fermentée traditionnelle." \\
    --output-dir runs/test_mt_tts/

  # Mode A — Adja speech → Adja speech (canonical):
  OPENROUTER_API_KEY=sk-or-... python pipeline.py \\
    --mode qa \\
    --input path/to/adja_question.wav \\
    --output-dir runs/qa_mode_a/

  # Batch:
  python pipeline.py --mode roundtrip --input-dir samples/ --output-dir runs/batch01/

  # Single stage only:
  python pipeline.py --stages asr --input audio.wav --ref-transcript "ref"
"""
import argparse
import csv
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import yaml

sys.stdout.reconfigure(line_buffering=True)

_HERE = Path(__file__).resolve().parent
# Add pipeline dir to sys.path so 'stages' and 'eval' packages are importable
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
# Add shared ASR metrics (experiments/asr/shared/)
_SHARED = _HERE.parent.parent / "asr" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))


# ---------------------------------------------------------------------------
# Config + CLI
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P1 Cascade Speech Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input — three mutually exclusive sources
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--input",
                     help="Single Adja WAV file (Mode A)")
    inp.add_argument("--input-dir",
                     help="Directory of Adja WAV files — batch mode (Mode A)")
    inp.add_argument("--input-manifest",
                     help="TSV manifest with id, audio_path, text columns — batch mode with per-sample references")
    inp.add_argument("--input-text",
                     metavar="FRENCH_TEXT",
                     help="French question/sentence — skip ASR + MT→FR (Mode B). "
                          "Useful for testing MT+TTS without an Adja speaker.")

    # Language of the audio input (only relevant for --input / --input-dir)
    p.add_argument("--input-lang", choices=["adja", "fr"], default="adja",
                   help="Language of the audio input. "
                        "'adja' (default): Adja ASR → MT→FR → ... (Mode A). "
                        "'fr': French ASR → ... skip MT→FR (Mode C, 2026-05-07).")

    # Optional references for metric computation
    p.add_argument("--ref-transcript",
                   help="Reference Adja transcript (ASR CER). Only used with --input/--input-dir.")
    p.add_argument("--ref-fr",
                   help="Reference French translation (MT→FR chrF).")
    p.add_argument("--ref-adja-answer",
                   help="Reference Adja back-translation (MT→Adja chrF).")

    # Pipeline mode
    p.add_argument("--mode", choices=["roundtrip", "qa"], default="roundtrip",
                   help="roundtrip: skip LLM, measure error accumulation; "
                        "qa: LLM (OpenRouter) generates French answer (default: roundtrip)")

    # Stage selection
    p.add_argument("--stages", default="all",
                   help="Comma-separated subset: asr,mt_fwd,response,mt_back,tts (default: all). "
                        "asr and mt_fwd are skipped automatically in Mode B.")

    # Paths + flags
    p.add_argument("--config",
                   default=str(_HERE / "configs" / "p1_config.yaml"),
                   help="Config YAML path")
    p.add_argument("--output-dir",
                   default=str(_HERE / "runs" / "latest"),
                   help="Directory for outputs and per-stage cache files")
    p.add_argument("--dry-run", action="store_true",
                   help="Use stubs for all stages — no GPU, no API key needed")
    p.add_argument("--force", action="store_true",
                   help="Ignore cached stage outputs and re-run everything")
    p.add_argument("--fail-fast", action="store_true",
                   help="Stop batch mode on the first sample error. Default: record the error and continue.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip())


def _text_to_sample_id(text: str) -> str:
    """Derive a filesystem-safe, human-readable ID from French input text.

    Used as the sample identifier when running in Mode B (--input-text).
    Format: fr_<first-30-chars-slugified>_<HHMMSS>
    The timestamp suffix prevents cache collisions when running the same
    question multiple times with different configs.
    """
    slug = re.sub(r"[^\w\s-]", "", _nfc(text).lower())
    slug = re.sub(r"\s+", "_", slug.strip())[:30].rstrip("_")
    ts = time.strftime("%H%M%S")
    return f"fr_{slug}_{ts}" if slug else f"fr_input_{ts}"


def _save_cache(cache: dict, path: Path) -> None:
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from the nearest .env without printing secrets."""
    candidates = [
        Path.cwd() / ".env",
        _HERE / ".env",
        _HERE.parent / ".env",
        _HERE.parent.parent / ".env",
        _HERE.parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        print(f"Loaded environment variables from {env_path}")
        return


def _load_manifest(path: Path) -> list[dict]:
    """Load the ASR TSV manifest produced by experiments/asr/shared/data_prep.py."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    required = {"id", "audio_path", "text"}
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Manifest {path} missing required columns: {sorted(missing)}")
    return rows


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 2) if values else None


def _write_aggregate_summary(all_results: list[dict], output_dir: Path) -> None:
    """Write a compact aggregate JSON for reports and thesis plotting."""
    from eval.stage_metrics import asr_metrics, rtt_metrics

    asr_cer: list[float] = []
    asr_wer: list[float] = []
    rtt_primary_wer: list[float] = []
    rtt_primary_cer: list[float] = []
    rtt_xlsr_wer: list[float] = []
    rtt_xlsr_cer: list[float] = []
    rtt_whisper_wer: list[float] = []
    rtt_whisper_cer: list[float] = []
    stage_times: dict[str, list[float]] = {}

    def add_time(stage: str, payload: dict) -> None:
        elapsed = payload.get("elapsed_sec")
        if isinstance(elapsed, (int, float)):
            stage_times.setdefault(stage, []).append(float(elapsed))

    for result in all_results:
        ref = result.get("references", {}).get("transcript")
        asr_out = result.get("asr", {})
        if ref and asr_out.get("primary"):
            m = asr_metrics(ref, asr_out["primary"])
            if m.get("cer") is not None:
                asr_cer.append(float(m["cer"]))
            if m.get("wer") is not None:
                asr_wer.append(float(m["wer"]))

        tts_text = result.get("tts", {}).get("text") or result.get("mt_back", {}).get("text")
        rtt_out = result.get("rtt", {})
        if tts_text and rtt_out.get("primary"):
            m = rtt_metrics(tts_text, rtt_out["primary"])
            if m.get("wer") is not None:
                rtt_primary_wer.append(float(m["wer"]))
            if m.get("cer") is not None:
                rtt_primary_cer.append(float(m["cer"]))
        if tts_text and rtt_out.get("xlsr"):
            m = rtt_metrics(tts_text, rtt_out["xlsr"])
            if m.get("wer") is not None:
                rtt_xlsr_wer.append(float(m["wer"]))
            if m.get("cer") is not None:
                rtt_xlsr_cer.append(float(m["cer"]))
        if tts_text and rtt_out.get("whisper"):
            m = rtt_metrics(tts_text, rtt_out["whisper"])
            if m.get("wer") is not None:
                rtt_whisper_wer.append(float(m["wer"]))
            if m.get("cer") is not None:
                rtt_whisper_cer.append(float(m["cer"]))

        for stage in ("asr", "mt_fwd", "response", "mt_back", "tts"):
            add_time(stage, result.get(stage, {}))

    summary = {
        "n_total": len(all_results),
        "n_failed": sum(1 for r in all_results if r.get("error")),
        "n_generated_audio": sum(1 for r in all_results if r.get("tts", {}).get("file")),
        "asr_primary": {
            "cer_mean": _mean(asr_cer),
            "wer_mean": _mean(asr_wer),
            "n": len(asr_cer),
        },
        "rtt_primary": {
            "wer_mean": _mean(rtt_primary_wer),
            "cer_mean": _mean(rtt_primary_cer),
            "n": len(rtt_primary_wer),
        },
        "rtt_xlsr": {
            "wer_mean": _mean(rtt_xlsr_wer),
            "cer_mean": _mean(rtt_xlsr_cer),
            "n": len(rtt_xlsr_wer),
        },
        "rtt_whisper": {
            "wer_mean": _mean(rtt_whisper_wer),
            "cer_mean": _mean(rtt_whisper_cer),
            "n": len(rtt_whisper_wer),
        },
        "stage_elapsed_sec_mean": {
            stage: _mean(vals) for stage, vals in sorted(stage_times.items())
        },
    }
    summary_path = output_dir / "aggregate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Aggregate summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# Build stage objects once (models load lazily on first use)
# ---------------------------------------------------------------------------

def build_stages(config: dict, dry_run: bool, input_lang: str = "adja") -> dict:
    from stages.asr import ASRStage, FrenchASR, StubASR
    from stages.mt import MTStage
    from stages.response import ResponseStage
    from stages.tts import TTSStage
    stages = {
        "asr":      ASRStage(config["asr"],           dry_run=dry_run),
        "mt":       MTStage(config["mt"],              dry_run=dry_run),
        "response": ResponseStage(config["response"],  dry_run=dry_run),
        "tts":      TTSStage(config["tts"],            dry_run=dry_run),
    }
    # Mode C: replace the Adja ASR with a French Whisper model so the
    # transcription output is French, not Adja, and MT→FR can be skipped.
    if input_lang == "fr":
        fr_model = config.get("asr_fr", {}).get("model_id", "openai/whisper-small")
        stages["asr_fr"] = StubASR("fr") if dry_run else FrenchASR(fr_model)
    return stages


# ---------------------------------------------------------------------------
# Mode A pipeline — audio input (Adja speech → ... → Adja speech)
# ---------------------------------------------------------------------------

def run_audio_sample(
    audio_path: Path,
    stages: dict,
    config: dict,
    args: argparse.Namespace,
    output_dir: Path,
    refs: dict,
    generated_audio_dir: Path,
    sample_meta: Optional[dict] = None,
) -> dict:
    """Full pipeline from an Adja WAV file.

    Stages: ASR → MT→FR → [response] → MT→Adja → TTS → RTT-ASR
    Each stage output is cached so re-runs are cheap (--force to override).
    """
    from eval.stage_metrics import asr_metrics, mt_metrics, rtt_metrics
    from eval.accumulation import AccumulationReport, StageResult

    sample_id = (sample_meta or {}).get("id") or audio_path.stem
    cache_path = output_dir / f"{sample_id}_cache.json"
    report = AccumulationReport(sample_id)

    cache: dict = {}
    if cache_path.exists() and not args.force:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    requested: Optional[set[str]] = (
        set(args.stages.split(",")) if args.stages != "all" else None
    )

    def should_run(stage: str) -> bool:
        return requested is None or stage in requested

    result: dict = {
        "sample_id": sample_id,
        "audio": str(audio_path),
        "mode": "A",
        "references": {k: v for k, v in refs.items() if v},
    }
    if sample_meta:
        result["manifest"] = sample_meta

    # ── Stage 1: ASR ──────────────────────────────────────────────────────
    if should_run("asr"):
        if "asr" not in cache or args.force:
            t0 = time.time()
            asr_out = stages["asr"].run(str(audio_path))
            cache["asr"] = {**asr_out, "elapsed_sec": round(time.time() - t0, 2)}
            _save_cache(cache, cache_path)

        asr_result = cache["asr"]
        m = asr_metrics(refs.get("transcript"), asr_result["primary"])
        report.add(StageResult(
            name=f"ASR ({config['asr']['primary']})",
            input=str(audio_path),
            output=asr_result["primary"],
            metric_name="CER",
            metric_value=m.get("cer"),
            extra={"whisper": asr_result.get("whisper"),
                   "xlsr": asr_result.get("xlsr"),
                   "asr_wer": m.get("wer")},
        ))
        result["asr"] = asr_result

    adja_text: str = cache.get("asr", {}).get("primary", "")

    # ── Stage 2: MT Adja → French ─────────────────────────────────────────
    if should_run("mt_fwd") and adja_text:
        if "mt_fwd" not in cache or args.force:
            t0 = time.time()
            fr_text = stages["mt"].adja_to_fr(adja_text)
            cache["mt_fwd"] = {"text": fr_text, "elapsed_sec": round(time.time() - t0, 2)}
            _save_cache(cache, cache_path)

        fr_text = cache["mt_fwd"]["text"]
        m = mt_metrics(refs.get("fr"), fr_text)
        report.add(StageResult(
            name="MT Adja→FR",
            input=adja_text, output=fr_text,
            metric_name="chrF", metric_value=m.get("chrf"),
        ))
        result["mt_fwd"] = cache["mt_fwd"]

    fr_text: str = cache.get("mt_fwd", {}).get("text", "")

    # Shared tail (response → MT→Adja → TTS) reused by both mode A and B
    return _run_fr_to_speech_tail(
        fr_text, stages, args, output_dir, refs, generated_audio_dir,
        cache, cache_path, report, result,
    )


# ---------------------------------------------------------------------------
# Mode B pipeline — French text input (skip ASR + MT→FR)
# ---------------------------------------------------------------------------

def run_text_input(
    french_text: str,
    stages: dict,
    config: dict,
    args: argparse.Namespace,
    output_dir: Path,
    refs: dict,
    generated_audio_dir: Path,
) -> dict:
    """Pipeline from a French text question (Mode B).

    Mode B rationale: testing without an Adja speaker.
    Skips ASR and MT→FR entirely. Starts directly at the response stage
    (or MT→Adja in roundtrip mode). The sample_id is derived from the
    text so cache files are human-readable.

    Stages run: [response] → MT→Adja → TTS → RTT-ASR
    """
    from eval.accumulation import AccumulationReport, StageResult

    sample_id = _text_to_sample_id(french_text)
    cache_path = output_dir / f"{sample_id}_cache.json"
    report = AccumulationReport(sample_id)

    cache: dict = {}
    if cache_path.exists() and not args.force:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    # Seed cache with the input text so the tail function can read it
    if "fr_input" not in cache:
        cache["fr_input"] = {"text": _nfc(french_text)}
        _save_cache(cache, cache_path)

    report.add(StageResult(
        name="Input (FR text)",
        input=None, output=_nfc(french_text),
        metric_name=None, metric_value=None,
    ))

    result: dict = {
        "sample_id": sample_id,
        "input_text": french_text,
        "mode": "B",
        "fr_input": cache["fr_input"],
    }

    return _run_fr_to_speech_tail(
        _nfc(french_text), stages, args, output_dir, refs, generated_audio_dir,
        cache, cache_path, report, result,
    )


# ---------------------------------------------------------------------------
# Mode C pipeline — French audio input (--input-lang fr)
# ---------------------------------------------------------------------------

def run_french_audio_sample(
    audio_path: Path,
    stages: dict,
    config: dict,
    args: argparse.Namespace,
    output_dir: Path,
    refs: dict,
    generated_audio_dir: Path,
    sample_meta: Optional[dict] = None,
) -> dict:
    """Pipeline from a French WAV file (Mode C, 2026-05-07).

    French speech → French ASR (Whisper-small, language=fr)
                  → [response (LLM)]
                  → MT→Adja
                  → TTS
                  → RTT-ASR

    MT→FR is SKIPPED: the ASR output is already French.
    Everything else reuses the shared tail identical to Mode B.
    """
    from eval.stage_metrics import rtt_metrics
    from eval.accumulation import AccumulationReport, StageResult

    sample_id = (sample_meta or {}).get("id") or audio_path.stem
    cache_path = output_dir / f"{sample_id}_cache.json"
    report = AccumulationReport(sample_id)

    cache: dict = {}
    if cache_path.exists() and not args.force:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)

    result: dict = {
        "sample_id": sample_id,
        "audio": str(audio_path),
        "mode": "C",
        "references": {k: v for k, v in refs.items() if v},
    }
    if sample_meta:
        result["manifest"] = sample_meta

    # ── French ASR ────────────────────────────────────────────────────────
    if "asr_fr" not in cache or args.force:
        t0 = time.time()
        fr_text = stages["asr_fr"].transcribe(str(audio_path))
        cache["asr_fr"] = {"text": fr_text, "elapsed_sec": round(time.time() - t0, 2)}
        _save_cache(cache, cache_path)

    fr_text = cache["asr_fr"]["text"]
    report.add(StageResult(
        name="ASR FR (Whisper)",
        input=str(audio_path), output=fr_text,
        metric_name=None, metric_value=None,
    ))
    result["asr_fr"] = cache["asr_fr"]

    # Shared tail: [response] → MT→Adja → TTS → RTT-ASR
    return _run_fr_to_speech_tail(
        fr_text, stages, args, output_dir, refs, generated_audio_dir,
        cache, cache_path, report, result,
    )


# ---------------------------------------------------------------------------
# Shared tail: FR text → [response] → MT→Adja → TTS → RTT-ASR
# ---------------------------------------------------------------------------

def _run_fr_to_speech_tail(
    fr_text: str,
    stages: dict,
    args: argparse.Namespace,
    output_dir: Path,
    refs: dict,
    generated_audio_dir: Path,
    cache: dict,
    cache_path: Path,
    report,
    result: dict,
) -> dict:
    """Shared pipeline tail used by both Mode A and Mode B.

    Takes a French text string and produces:
      [response] → MT→Adja → TTS → RTT-ASR

    The response stage is only executed when --mode qa is set. In roundtrip
    mode the French text is passed directly to MT→Adja, giving a clean
    measurement of MT + TTS error without LLM variance.
    """
    from eval.stage_metrics import mt_metrics, rtt_metrics
    from eval.accumulation import StageResult

    requested: Optional[set[str]] = (
        set(args.stages.split(",")) if args.stages != "all" else None
    )

    def should_run(stage: str) -> bool:
        return requested is None or stage in requested

    # ── Stage 3: Response (Q&A mode only) ─────────────────────────────────
    if args.mode == "qa" and should_run("response") and fr_text:
        if "response" not in cache or args.force:
            t0 = time.time()
            fr_answer = stages["response"].run(fr_text)
            cache["response"] = {"text": fr_answer, "elapsed_sec": round(time.time() - t0, 2)}
            _save_cache(cache, cache_path)

        fr_answer = cache["response"]["text"]
        report.add(StageResult(
            name="Response (FR)", input=fr_text, output=fr_answer,
            metric_name=None, metric_value=None,
        ))
        result["response"] = cache["response"]
    else:
        # Roundtrip: pass French text directly to back-translation
        fr_answer = fr_text

    # ── Stage 4: MT French → Adja ─────────────────────────────────────────
    if should_run("mt_back") and fr_answer:
        if "mt_back" not in cache or args.force:
            t0 = time.time()
            adja_answer = stages["mt"].fr_to_adja(fr_answer)
            cache["mt_back"] = {"text": adja_answer, "elapsed_sec": round(time.time() - t0, 2)}
            _save_cache(cache, cache_path)

        adja_answer = cache["mt_back"]["text"]
        m = mt_metrics(refs.get("adja_answer"), adja_answer)
        report.add(StageResult(
            name="MT FR→Adja", input=fr_answer, output=adja_answer,
            metric_name="chrF", metric_value=m.get("chrf"),
        ))
        result["mt_back"] = cache["mt_back"]

    adja_answer: str = cache.get("mt_back", {}).get("text", "")

    # ── Stage 5: TTS + RTT ────────────────────────────────────────────────
    if should_run("tts") and adja_answer:
        # Write to generated_audio/ — reverse_wer.py expects this layout:
        #   <run>/generated_audio/*.wav  +  <run>/metrics.json
        sample_id = result["sample_id"]
        tts_wav = generated_audio_dir / f"{sample_id}.wav"

        if "tts" not in cache or args.force or not tts_wav.exists():
            t0 = time.time()
            stages["tts"].run(adja_answer, str(tts_wav))
            cache["tts"] = {
                "wav": str(tts_wav),
                "file": tts_wav.name,   # filename inside generated_audio/
                "text": adja_answer,
                "elapsed_sec": round(time.time() - t0, 2),
            }
            _save_cache(cache, cache_path)

        report.add(StageResult(
            name="TTS", input=adja_answer, output=str(tts_wav),
            metric_name=None, metric_value=None,
        ))

        # RTT: transcribe TTS output with ASR to measure round-trip degradation.
        # Floor: ASR's own published CER (~25% for C4v2) is the minimum
        # achievable RTT-WER even on perfect TTS. Use as a relative ranking
        # signal, not absolute intelligibility. See reverse_wer.py docs.
        if tts_wav.exists() and ("rtt" not in cache or args.force):
            try:
                rtt_out = stages["asr"].run(str(tts_wav))
                cache["rtt"] = {**rtt_out, "text": rtt_out["primary"]}
            except Exception as exc:
                cache["rtt"] = {"text": "", "error": str(exc)}
            _save_cache(cache, cache_path)

        rtt_text = cache.get("rtt", {}).get("text", "")
        rtt_m = rtt_metrics(adja_answer, rtt_text)
        report.add(StageResult(
            name="TTS RTT-WER", input=adja_answer, output=rtt_text,
            metric_name="WER", metric_value=rtt_m.get("wer"),
            error=cache.get("rtt", {}).get("error"),
        ))
        result["tts"] = cache.get("tts", {})
        result["rtt"] = cache.get("rtt", {})

    report.print_table()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_dotenv()
    args = parse_args()
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("=== DRY RUN MODE — all stages use stubs (no GPU, no API key) ===\n")

    refs = {
        "transcript":   args.ref_transcript,
        "fr":           args.ref_fr,
        "adja_answer":  args.ref_adja_answer,
    }

    # generated_audio/ — reverse_wer.py expects:
    #   <run>/generated_audio/*.wav  +  <run>/metrics.json
    generated_audio_dir = output_dir / "generated_audio"
    generated_audio_dir.mkdir(parents=True, exist_ok=True)

    # Build stage objects once — models load lazily on first use
    input_lang = getattr(args, "input_lang", "adja")
    stage_objects = build_stages(config, dry_run=args.dry_run, input_lang=input_lang)

    all_results: list[dict] = []

    # ── Dispatch based on input mode ──────────────────────────────────────
    if args.input_text:
        # Mode B: single French text input
        mode_label = "B (French text → Adja speech)"
        print(f"Mode {mode_label}")
        print(f"Input: {args.input_text[:80]}\n")
        result = run_text_input(
            args.input_text, stage_objects, config, args, output_dir, refs,
            generated_audio_dir=generated_audio_dir,
        )
        all_results.append(result)

    else:
        # Mode A (Adja audio) or Mode C (French audio, --input-lang fr)
        mode_char = "C" if input_lang == "fr" else "A"
        if args.input:
            audio_files = [Path(args.input)]
            manifest_rows: Optional[list[dict]] = None
        elif args.input_manifest:
            manifest_rows = _load_manifest(Path(args.input_manifest))
            audio_files = [Path(row["audio_path"]) for row in manifest_rows]
            print(f"Manifest batch mode: {len(audio_files)} files from {args.input_manifest}\n")
        else:
            manifest_rows = None
            audio_files = sorted(Path(args.input_dir).glob("*.wav"))
            if not audio_files:
                sys.exit(f"No .wav files found in {args.input_dir}")
            print(f"Batch mode: {len(audio_files)} files in {args.input_dir}\n")

        for idx, audio_path in enumerate(audio_files):
            print(f"\n{'=' * 60}")
            print(f"Processing: {audio_path.name}")
            sample_refs = refs
            sample_meta = None
            if manifest_rows is not None:
                row = manifest_rows[idx]
                sample_refs = {
                    "transcript": row.get("text") or args.ref_transcript,
                    "fr": args.ref_fr,
                    "adja_answer": args.ref_adja_answer,
                }
                sample_meta = {
                    "id": row.get("id") or audio_path.stem,
                    "audio_path": row.get("audio_path") or str(audio_path),
                    "duration_sec": row.get("duration_sec"),
                    "sampling_rate": row.get("sampling_rate"),
                }
            try:
                if input_lang == "fr":
                    # Mode C: French audio in → French ASR → shared tail
                    result = run_french_audio_sample(
                        audio_path, stage_objects, config, args, output_dir, sample_refs,
                        generated_audio_dir=generated_audio_dir,
                        sample_meta=sample_meta,
                    )
                else:
                    # Mode A: Adja audio in → Adja ASR → MT→FR → shared tail
                    result = run_audio_sample(
                        audio_path, stage_objects, config, args, output_dir, sample_refs,
                        generated_audio_dir=generated_audio_dir,
                        sample_meta=sample_meta,
                    )
            except Exception as exc:
                if args.fail_fast:
                    raise
                result = {
                    "sample_id": (sample_meta or {}).get("id") or audio_path.stem,
                    "audio": str(audio_path),
                    "mode": mode_char,
                    "references": {k: v for k, v in sample_refs.items() if v},
                    "manifest": sample_meta,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[ERROR] {result['sample_id']}: {result['error']}")
            all_results.append(result)

    if len(all_results) > 1:
        print(f"\nProcessed {len(all_results)} samples. See individual reports above.")

    # ── Detailed results JSON ──────────────────────────────────────────────
    results_path = output_dir / "results.json"
    results_path.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nFull results saved to {results_path}")
    _write_aggregate_summary(all_results, output_dir)

    # ── reverse_wer.py-compatible metrics.json ────────────────────────────
    # Format: {"generated": [{"file": "<wav>", "text": "<prompt>", ...}, ...]}
    # Push <output_dir>/ to JosueG/adja-tts-results/<run-name> on Hub, then:
    #   python experiments/tts/eval/reverse_wer.py \
    #       --runs <run-name> \
    #       --adja-asrs '<C4V2_URL>:c4v2,<E4V4_URL>:e4v4'
    generated_entries = []
    for r in all_results:
        tts_info = r.get("tts", {})
        if tts_info.get("file") and tts_info.get("text"):
            wav_path = generated_audio_dir / tts_info["file"]
            duration: Optional[float] = None
            if wav_path.exists():
                try:
                    import soundfile as sf
                    duration = round(sf.info(str(wav_path)).duration, 2)
                except Exception:
                    pass
            generated_entries.append({
                "file": tts_info["file"],
                "text": tts_info["text"],
                **({"duration_sec": duration} if duration is not None else {}),
            })

    if generated_entries:
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps({"generated": generated_entries}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"reverse_wer-compatible metrics.json → {metrics_path}")
        print("Score TTS quality after pushing to Hub:")
        print("  python experiments/tts/eval/reverse_wer.py \\")
        print("      --runs <run-name> \\")
        print("      --adja-asrs '<C4V2_URL>:c4v2,<E4V4_URL>:e4v4'")


if __name__ == "__main__":
    main()
