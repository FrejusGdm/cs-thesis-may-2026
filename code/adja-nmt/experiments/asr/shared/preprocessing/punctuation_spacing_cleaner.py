#!/usr/bin/env python3
"""
Punctuation and spacing cleaner for Adja text.

Source: adapted from
  <LOCAL_PATH>
  experiments/preprocessing/punctuation_spacing_cleaner.py
Original author: Josue Godeme (ACL paper dataset prep).

Adaptations:
- Operates on a single string (not bilingual French+Adja CSV rows).
- Dropped French-specific question-word detection — the original script
  was fixing mismatched punctuation *between* French and its Adja translation.
  For LM text prep we don't have French context; we just normalize Adja.
- Kept: exclamation mark standardization (ǃ → !), multi-space collapse,
  whitespace around punctuation.
"""
from __future__ import annotations

import re


# Replacement rules for non-standard punctuation characters
# (U+01C3 "LATIN LETTER RETROFLEX CLICK" and U+01C3 variants get confused with !)
_EXCLAMATION_VARIANTS = ["ǃ", "\u01c3"]


def standardize_exclamation(text: str) -> str:
    """Replace non-standard exclamation marks with regular '!'."""
    for variant in _EXCLAMATION_VARIANTS:
        text = text.replace(variant, "!")
    return text


def collapse_multi_spaces(text: str) -> str:
    """Collapse multiple consecutive spaces/whitespace into a single space."""
    return re.sub(r"\s+", " ", text)


def normalize_punctuation_spacing(text: str) -> str:
    """
    Remove space BEFORE punctuation (. , ! ? ; :)
    Ensure single space AFTER punctuation (when followed by word char).
    """
    # Remove space before punctuation
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    # Ensure a space after punctuation if followed by a non-space, non-punct char
    text = re.sub(
        r"([.,!?;:])([^\s.,!?;:\"'»“”‘’])",
        r"\1 \2",
        text,
    )

    # Collapse any multiple spaces introduced by the rewrites
    text = re.sub(r" {2,}", " ", text)

    return text


def clean_punctuation_spacing(text: str) -> str:
    """Full single-string punctuation/spacing pipeline.

    Preserves:
    - Adja special chars (ɛ, ɔ, ŋ, ɖ)
    - Combining tone marks (é, è, ẽ, ɔ̀, etc.)
    - Quotation marks (just leaves them alone — we don't try to balance them)

    Modifies:
    - Multiple spaces → single space
    - Whitespace around punctuation
    - Non-standard exclamation variants (ǃ → !)
    """
    if not text or not isinstance(text, str):
        return ""
    text = standardize_exclamation(text)
    text = normalize_punctuation_spacing(text)
    text = collapse_multi_spaces(text)
    return text.strip()
