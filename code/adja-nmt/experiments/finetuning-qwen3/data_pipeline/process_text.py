#!/usr/bin/env python3
"""
Stage 3: Text Processing
========================
Language-specific text normalization for Qwen3 TTS/ASR fine-tuning.

WHY THIS MATTERS:
If your text says "Dr." but the speaker says "Doctor", the model sees a mismatch.
Similarly, "100" vs "one hundred", "St." vs "Street" vs "Saint" — these
inconsistencies confuse the model during training. For non-English languages,
there are additional challenges:
- Different numeral systems (Arabic numerals vs native script)
- Diacritical marks and combining characters
- Honorifics and abbreviations specific to the language
- Mixed-script text (e.g., English words within non-Latin text)

This script provides a framework you MUST customize for your specific language.

Input:  JSONL with "text" field
Output: JSONL with normalized "text" field + optional "text_original" backup

Usage:
    python process_text.py \
        --input_jsonl processed_tts.jsonl \
        --output_jsonl normalized_tts.jsonl \
        --language auto
"""

import argparse
import json
import logging
import os
import re
import unicodedata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Universal text normalization (language-agnostic)
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Apply Unicode NFC normalization.

    WHY: The same visual character can have multiple Unicode representations.
    For example, "é" can be:
    - U+00E9 (single codepoint, NFC form)
    - U+0065 + U+0301 (base 'e' + combining acute accent, NFD form)

    If your dataset has both forms, the tokenizer sees them as different,
    wasting vocabulary space and making training inconsistent.

    NFC is the standard for web content and most text, so we normalize to it.
    """
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace to single spaces, strip edges."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_control_characters(text: str) -> str:
    """Remove invisible control characters that can break tokenization."""
    return "".join(
        char for char in text
        if unicodedata.category(char) != "Cc" or char in ("\n", "\t")
    )


# ---------------------------------------------------------------------------
# Number-to-words conversion framework
# ---------------------------------------------------------------------------
# IMPORTANT: You must implement this for your specific language.
# These are placeholder examples. Consider using the `num2words` library
# which supports many languages:
#     pip install num2words
#     from num2words import num2words
#     num2words(42, lang='fr')  → 'quarante-deux'

def number_to_words(text: str, language: str = "en") -> str:
    """Convert numeric digits to spoken-form words.

    This is a FRAMEWORK — you must customize for your language.
    For production use, install and use the num2words library.
    """
    try:
        from num2words import num2words as n2w
    except ImportError:
        logger.warning(
            "num2words not installed. Numbers will not be converted. "
            "Install with: pip install num2words"
        )
        return text

    def replace_number(match):
        num_str = match.group(0)
        try:
            # Handle integers and decimals
            if "." in num_str:
                num = float(num_str)
            else:
                num = int(num_str)
            return n2w(num, lang=language)
        except (ValueError, NotImplementedError):
            return num_str  # Keep original if conversion fails

    # Match numbers (with optional decimal point)
    # Be careful: this regex avoids matching numbers inside words
    return re.sub(r"\b\d+\.?\d*\b", replace_number, text)


# ---------------------------------------------------------------------------
# Common abbreviation expansion
# ---------------------------------------------------------------------------

# Add your language-specific abbreviations here
ABBREVIATIONS = {
    "en": {
        r"\bDr\.": "Doctor",
        r"\bMr\.": "Mister",
        r"\bMrs\.": "Missus",
        r"\bMs\.": "Ms",
        r"\bSt\.": "Street",
        r"\betc\.": "etcetera",
        r"\bvs\.": "versus",
    },
    # ADD YOUR LANGUAGE:
    # "fr": {
    #     r"\bM\.": "Monsieur",
    #     r"\bMme\.?": "Madame",
    #     ...
    # },
}


def expand_abbreviations(text: str, language: str = "en") -> str:
    """Expand common abbreviations to full spoken form."""
    abbrevs = ABBREVIATIONS.get(language, {})
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)
    return text


# ---------------------------------------------------------------------------
# Punctuation normalization
# ---------------------------------------------------------------------------

def normalize_punctuation(text: str) -> str:
    """Normalize punctuation for TTS consistency.

    Different quote styles, dashes, and ellipses should be standardized.
    """
    replacements = {
        "\u2018": "'",   # Left single quote → apostrophe
        "\u2019": "'",   # Right single quote → apostrophe
        "\u201C": '"',   # Left double quote
        "\u201D": '"',   # Right double quote
        "\u2013": "-",   # En dash
        "\u2014": " - ", # Em dash → space-separated dash
        "\u2026": "...", # Horizontal ellipsis
        "\u00A0": " ",   # Non-breaking space → regular space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


# ---------------------------------------------------------------------------
# TTS-specific formatting
# ---------------------------------------------------------------------------

def format_for_tts(text: str) -> str:
    """Final formatting pass for TTS training.

    The Qwen3-TTS model expects clean, natural-reading text.
    It handles punctuation-based prosody internally.
    """
    # Remove multiple consecutive punctuation (e.g., "!!!" → "!")
    text = re.sub(r"([!?.]){2,}", r"\1", text)
    # Ensure single space after punctuation
    text = re.sub(r"([.!?,;:])\s*", r"\1 ", text)
    # Clean up any resulting double spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# ASR-specific formatting
# ---------------------------------------------------------------------------

def format_for_asr(text: str, language: str = "en") -> str:
    """Format text as required by Qwen3-ASR fine-tuning.

    The ASR model expects:
    - Language prefix: "language English<asr_text>..."
    - Or no language: "language None<asr_text>..."
    """
    # Map language codes to ASR language names
    asr_language_map = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "pt": "Portuguese",
        "ru": "Russian",
        "it": "Italian",
        "ar": "Arabic",
        "th": "Thai",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "tr": "Turkish",
        "hi": "Hindi",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "pl": "Polish",
        "cs": "Czech",
        "fil": "Filipino",
        "fa": "Persian",
        "el": "Greek",
        "hu": "Hungarian",
        "mk": "Macedonian",
        "ro": "Romanian",
        "ms": "Malay",
    }

    lang_name = asr_language_map.get(language, "None")
    return f"language {lang_name}<asr_text>{text}"


# ---------------------------------------------------------------------------
# Full normalization pipeline
# ---------------------------------------------------------------------------

def normalize_text(
    text: str,
    language: str = "en",
    mode: str = "tts",
    convert_numbers: bool = True,
) -> str:
    """Apply the full text normalization pipeline.

    Args:
        text: Raw text string
        language: ISO 639-1 language code (e.g., "en", "fr", "zh")
        mode: "tts" or "asr" — determines output format
        convert_numbers: Whether to convert digits to words
    """
    # Step 1: Unicode normalization (always do this first)
    text = normalize_unicode(text)

    # Step 2: Remove control characters
    text = remove_control_characters(text)

    # Step 3: Normalize punctuation
    text = normalize_punctuation(text)

    # Step 4: Expand abbreviations
    text = expand_abbreviations(text, language)

    # Step 5: Convert numbers to words
    if convert_numbers:
        text = number_to_words(text, language)

    # Step 6: Whitespace normalization
    text = normalize_whitespace(text)

    # Step 7: Mode-specific formatting
    if mode == "tts":
        text = format_for_tts(text)
    elif mode == "asr":
        text = format_for_asr(text, language)

    return text


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_text_dataset(
    input_jsonl: str,
    output_jsonl: str,
    language: str = "en",
    mode: str = "tts",
    convert_numbers: bool = True,
    keep_original: bool = True,
):
    """Process text in all records of a JSONL file."""

    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Processing text for {len(records)} records (language={language}, mode={mode})")

    processed_count = 0
    changed_count = 0

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for rec in records:
            original = rec["text"]
            normalized = normalize_text(
                original,
                language=language,
                mode=mode,
                convert_numbers=convert_numbers,
            )

            if keep_original and normalized != original:
                rec["text_original"] = original
                changed_count += 1

            rec["text"] = normalized
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            processed_count += 1

    logger.info(f"Processed: {processed_count} records")
    logger.info(f"Changed:   {changed_count} records had text modifications")
    logger.info(f"Output:    {output_jsonl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Text normalization for Qwen3 TTS/ASR fine-tuning"
    )
    parser.add_argument("--input_jsonl", required=True, help="Input JSONL file")
    parser.add_argument("--output_jsonl", required=True, help="Output JSONL file")
    parser.add_argument(
        "--language", default="en",
        help="ISO 639-1 language code (e.g., en, fr, zh, ar). Default: en"
    )
    parser.add_argument(
        "--mode", choices=["tts", "asr"], default="tts",
        help="Output format: 'tts' (clean text) or 'asr' (with language prefix)"
    )
    parser.add_argument(
        "--no_numbers", action="store_true",
        help="Skip number-to-words conversion"
    )
    parser.add_argument(
        "--no_keep_original", action="store_true",
        help="Don't save original text as 'text_original' field"
    )
    args = parser.parse_args()

    process_text_dataset(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        language=args.language,
        mode=args.mode,
        convert_numbers=not args.no_numbers,
        keep_original=not args.no_keep_original,
    )


if __name__ == "__main__":
    main()
