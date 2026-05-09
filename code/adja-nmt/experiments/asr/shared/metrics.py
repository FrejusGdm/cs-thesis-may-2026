#!/usr/bin/env python3
"""
ASR evaluation metrics for Adja experiments.

Computes WER, CER, and related metrics.
Used by all experiments for consistent evaluation.

Usage:
    python metrics.py --ref references.txt --hyp hypotheses.txt
    python metrics.py --ref-tsv manifests/test.tsv --hyp-tsv results/decode.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)


# --- Text normalization for robust WER/CER ---

# Punctuation characters to strip for normalized metrics.
# CRITICAL: Do NOT include ɛ, ɔ, ŋ, ɖ or tone marks here — those change Adja meaning.
_PUNCT_TO_STRIP = r"""[.,!?;:()\[\]"'«»“”‘’]"""


def normalize_for_wer(text: str) -> str:
    """Normalize a string for WER/CER computation robustness.

    The comparison-agnostic normalizations applied:
    - Unicode NFC (so "ɔ̀" composed == "ɔ̀" decomposed)
    - Lowercase (so "Enu" == "enu")
    - Strip sentence-ending punctuation (. , ! ? ; : parens quotes)
    - Collapse whitespace (tabs, multiple spaces)

    PRESERVED (character-level Adja meaning):
    - ɛ, ɔ, ŋ, ɖ (special Gbe characters)
    - Combining tone marks (é, è, ẽ, ɔ̀, etc.)
    - Hyphens inside words

    This function mirrors Whisper's BasicTextNormalizer approach but is tailored
    to keep Adja's phonologically meaningful features intact. Paper ref:
    https://github.com/openai/whisper/blob/main/whisper/normalizers/basic.py

    Args:
        text: raw reference or hypothesis string.

    Returns:
        Normalized string ready for edit-distance computation.
    """
    if not text or not isinstance(text, str):
        return ""
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)
    # 2. Lowercase
    text = text.lower()
    # 3. Strip punctuation (keep tone marks / special chars)
    text = re.sub(_PUNCT_TO_STRIP, "", text)
    # 4. Collapse whitespace
    text = " ".join(text.split())
    return text


def edit_distance(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """Compute edit distance (Levenshtein) between two sequences.

    Returns: (substitutions, insertions, deletions, total_edits)
    """
    n, m = len(ref), len(hyp)

    # dp[i][j] = (subs, ins, dels) to convert ref[:i] to hyp[:j]
    dp = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (0, 0, i)  # delete all ref tokens
    for j in range(1, m + 1):
        dp[0][j] = (0, j, 0)  # insert all hyp tokens

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Substitution
                s, ins, d = dp[i - 1][j - 1]
                sub_cost = (s + 1, ins, d)

                # Insertion (extra token in hyp)
                s, ins, d = dp[i][j - 1]
                ins_cost = (s, ins + 1, d)

                # Deletion (missing token in hyp)
                s, ins, d = dp[i - 1][j]
                del_cost = (s, ins, d + 1)

                # Pick minimum total
                best = min([sub_cost, ins_cost, del_cost], key=lambda x: sum(x))
                dp[i][j] = best

    s, i, d = dp[n][m]
    return s, i, d, s + i + d


def compute_wer(references: list[str], hypotheses: list[str], normalize: bool = False) -> dict:
    """Compute Word Error Rate.

    WER = (S + I + D) / N where N = total reference words

    Args:
        references: list of reference strings.
        hypotheses: list of hypothesis strings.
        normalize: if True, apply normalize_for_wer() to BOTH refs and hyps
                   before comparison (strips punct, lowercases, NFC). Default
                   False to preserve existing caller behavior.
    """
    total_edits = 0
    total_ref_words = 0
    total_sub = 0
    total_ins = 0
    total_del = 0

    per_utt_wer = []

    for ref_str, hyp_str in zip(references, hypotheses):
        if normalize:
            ref_str = normalize_for_wer(ref_str)
            hyp_str = normalize_for_wer(hyp_str)
        ref_words = ref_str.strip().split()
        hyp_words = hyp_str.strip().split()

        s, i, d, edits = edit_distance(ref_words, hyp_words)
        n = len(ref_words)

        total_sub += s
        total_ins += i
        total_del += d
        total_edits += edits
        total_ref_words += n

        utt_wer = edits / n if n > 0 else 0.0
        per_utt_wer.append(utt_wer)

    wer = total_edits / total_ref_words if total_ref_words > 0 else 0.0

    return {
        "wer": round(wer * 100, 2),
        "total_edits": total_edits,
        "total_ref_words": total_ref_words,
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
        "num_utterances": len(references),
        "mean_utt_wer": round(sum(per_utt_wer) / len(per_utt_wer) * 100, 2) if per_utt_wer else 0.0,
    }


def compute_cer(references: list[str], hypotheses: list[str], normalize: bool = False) -> dict:
    """Compute Character Error Rate.

    CER = (S + I + D) / N where N = total reference characters

    Args:
        references: list of reference strings.
        hypotheses: list of hypothesis strings.
        normalize: if True, apply normalize_for_wer() (NFC + lowercase +
                   strip punctuation + whitespace collapse) to both refs and
                   hyps before comparison. Adja special chars and tone marks
                   are preserved. Default False for caller compatibility.
    """
    total_edits = 0
    total_ref_chars = 0
    total_sub = 0
    total_ins = 0
    total_del = 0

    per_utt_cer = []

    for ref_str, hyp_str in zip(references, hypotheses):
        if normalize:
            ref_str = normalize_for_wer(ref_str)
            hyp_str = normalize_for_wer(hyp_str)
        ref_chars = list(ref_str.strip())
        hyp_chars = list(hyp_str.strip())

        s, i, d, edits = edit_distance(ref_chars, hyp_chars)
        n = len(ref_chars)

        total_sub += s
        total_ins += i
        total_del += d
        total_edits += edits
        total_ref_chars += n

        utt_cer = edits / n if n > 0 else 0.0
        per_utt_cer.append(utt_cer)

    cer = total_edits / total_ref_chars if total_ref_chars > 0 else 0.0

    return {
        "cer": round(cer * 100, 2),
        "total_edits": total_edits,
        "total_ref_chars": total_ref_chars,
        "substitutions": total_sub,
        "insertions": total_ins,
        "deletions": total_del,
        "num_utterances": len(references),
        "mean_utt_cer": round(sum(per_utt_cer) / len(per_utt_cer) * 100, 2) if per_utt_cer else 0.0,
    }


def evaluate(references: list[str], hypotheses: list[str], output_path: str | None = None) -> dict:
    """Run full evaluation and optionally save results."""
    wer_results = compute_wer(references, hypotheses)
    cer_results = compute_cer(references, hypotheses)

    results = {
        "wer": wer_results["wer"],
        "cer": cer_results["cer"],
        "wer_details": wer_results,
        "cer_details": cer_results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    return results


def load_text_file(path: str) -> list[str]:
    """Load lines from a text file."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_tsv_column(path: str, col: str = "text") -> list[str]:
    """Load a column from a TSV manifest."""
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        col_idx = header.index(col)
        return [line.strip().split("\t")[col_idx] for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="ASR Evaluation Metrics")
    parser.add_argument("--ref", type=str, help="Reference text file (one per line)")
    parser.add_argument("--hyp", type=str, help="Hypothesis text file (one per line)")
    parser.add_argument("--ref-tsv", type=str, help="Reference TSV manifest")
    parser.add_argument("--hyp-tsv", type=str, help="Hypothesis TSV file")
    parser.add_argument("--output", type=str, help="Output JSON path")
    args = parser.parse_args()

    if args.ref and args.hyp:
        references = load_text_file(args.ref)
        hypotheses = load_text_file(args.hyp)
    elif args.ref_tsv and args.hyp_tsv:
        references = load_tsv_column(args.ref_tsv, "text")
        hypotheses = load_tsv_column(args.hyp_tsv, "text")
    else:
        parser.error("Provide either --ref/--hyp or --ref-tsv/--hyp-tsv")

    assert len(references) == len(hypotheses), \
        f"Mismatch: {len(references)} refs vs {len(hypotheses)} hyps"

    results = evaluate(references, hypotheses, args.output)

    print(f"\n=== ASR Evaluation Results ===")
    print(f"WER: {results['wer']}%")
    print(f"CER: {results['cer']}%")
    print(f"Utterances: {results['wer_details']['num_utterances']}")
    print(f"WER breakdown: S={results['wer_details']['substitutions']} "
          f"I={results['wer_details']['insertions']} "
          f"D={results['wer_details']['deletions']}")


if __name__ == "__main__":
    main()
