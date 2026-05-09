#!/usr/bin/env python3
from __future__ import annotations
"""
Per-stage error metrics for the P1 cascade pipeline.

Wraps shared ASR metrics (WER/CER) and adds chrF for MT stages.
chrF is standard for low-resource MT evaluation (Popovic, 2015).
Ref: https://aclanthology.org/W15-3049/
"""
import sys
import unicodedata
from pathlib import Path
from typing import Optional

# Reach the shared ASR metrics module at experiments/asr/shared/
_HERE = Path(__file__).resolve().parent
_SHARED = _HERE.parent.parent.parent / "asr" / "shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from metrics import compute_wer, compute_cer  # type: ignore  # noqa: E402


def asr_metrics(ref: Optional[str], hyp: str) -> dict:
    """WER and CER for an ASR output against a reference transcript."""
    if not ref:
        return {"wer": None, "cer": None, "note": "no reference"}
    wer = compute_wer([ref], [hyp], normalize=True)
    cer = compute_cer([ref], [hyp], normalize=True)
    return {
        "wer": wer["wer"],
        "cer": cer["cer"],
        "ref": ref,
        "hyp": hyp,
    }


def chrf_score(ref: str, hyp: str) -> Optional[float]:
    """ChrF between hypothesis and reference.

    Returns None if reference is empty. Falls back to simple char Jaccard
    if sacrebleu is not installed — install it for proper chrF.
    """
    if not ref or not hyp:
        return None
    try:
        from sacrebleu.metrics import CHRF
        return round(CHRF().sentence_score(hyp, [ref]).score, 2)
    except ImportError:
        # Rough fallback: character Jaccard on NFC-lowercased strings
        ref_n = unicodedata.normalize("NFC", ref.lower())
        hyp_n = unicodedata.normalize("NFC", hyp.lower())
        ref_c, hyp_c = set(ref_n), set(hyp_n)
        union = ref_c | hyp_c
        if not union:
            return None
        return round(len(ref_c & hyp_c) / len(union) * 100, 2)


def mt_metrics(ref: Optional[str], hyp: str) -> dict:
    """ChrF for a machine translation output against an optional reference."""
    score = chrf_score(ref, hyp) if ref else None
    return {
        "chrf": score,
        "ref": ref,
        "hyp": hyp,
        "note": "no reference" if ref is None else None,
    }


def rtt_metrics(original_text: str, rtt_transcript: str) -> dict:
    """Round-trip TTS metrics: ASR(TTS(text)) vs. original text.

    Measures how much the TTS+ASR round-trip degrades the text signal.
    """
    return asr_metrics(original_text, rtt_transcript)
