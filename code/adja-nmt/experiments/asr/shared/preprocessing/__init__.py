"""Preprocessing utilities for Adja text.

Adapted from the NMT project at
<LOCAL_PATH>

Exposes:
- normalize_unicode_text: NFC normalization
- clean_punctuation_spacing: single-text punctuation + spacing normalization
- preprocess_for_lm: full pipeline (normalize + clean) used for LM text prep
"""
from .normalize_unicode import normalize_unicode_text
from .punctuation_spacing_cleaner import clean_punctuation_spacing


def preprocess_for_lm(text: str) -> str:
    """Full preprocessing pipeline for LM training text.

    1. Unicode NFC normalization (preserves ɛ, ɔ, ŋ, ɖ, tone marks)
    2. Punctuation + spacing cleanup

    Args:
        text: raw Adja sentence

    Returns:
        Normalized sentence ready for LM training.
    """
    if not text or not isinstance(text, str):
        return ""
    text = normalize_unicode_text(text)
    text = clean_punctuation_spacing(text)
    return text.strip()
