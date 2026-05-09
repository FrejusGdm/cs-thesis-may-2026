#!/usr/bin/env python3
"""
Unicode NFC normalization for Adja text.

Source: adapted from
  <LOCAL_PATH>
  experiments/preprocessing/unicode-normalization/normalize_unicode.py
Original author: Josue Godeme (ACL paper dataset prep, 2025-11-19).

Adaptations:
- Simplified to a single pure function operating on strings/iterables,
  instead of a DataFrame-based class.
- NFC logic is unchanged — verbatim `unicodedata.normalize('NFC', str(text))`.
- Why NFC: critical for Adja because tone marks and IPA chars (ɛ, ɔ, ŋ, ɖ)
  can otherwise be stored as multi-codepoint decompositions, bloating the
  char vocab and breaking exact-match metrics like CER.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable, Iterator


def normalize_unicode_text(text: str) -> str:
    """Apply Unicode NFC (Canonical Composition) normalization.

    Args:
        text: input string. Empty / non-string input returns "".

    Returns:
        NFC-normalized text. Adja special chars (ɛ, ɔ, ŋ, ɖ) and tone
        marks (é, è, ẽ, ɔ̀, ...) are preserved — NFC just canonicalizes
        their encoding.
    """
    if not text or not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFC", text)


def normalize_unicode_iter(texts: Iterable[str]) -> Iterator[str]:
    """Lazy NFC normalization over a stream of strings."""
    for t in texts:
        yield normalize_unicode_text(t)
