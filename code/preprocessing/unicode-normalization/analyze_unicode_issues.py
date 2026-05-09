#!/usr/bin/env python3
"""
Unicode Normalization Issue Analyzer for Aja-French Dataset

This diagnostic script identifies Unicode normalization issues in your dataset,
specifically focusing on characters that exist in multiple Unicode forms
(composed vs decomposed).

Author: Created for Aja language dataset preprocessing
Date: 2025-11-19
"""

import pandas as pd
import unicodedata
from collections import defaultdict
import sys


def analyze_text_unicode(text):
    """
    Analyze a text string for Unicode normalization issues.

    Returns:
        dict: Analysis results including NFD/NFC differences
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    nfd_form = unicodedata.normalize('NFD', text)
    nfc_form = unicodedata.normalize('NFC', text)

    return {
        'original': text,
        'original_len': len(text),
        'nfd_len': len(nfd_form),
        'nfc_len': len(nfc_form),
        'has_combining_marks': nfd_form != nfc_form,
        'would_change': text != nfc_form,
        'nfc_form': nfc_form
    }


def get_character_breakdown(text):
    """
    Break down text into individual characters with their Unicode info.
    Useful for understanding what's happening with IPA characters.
    """
    if pd.isna(text) or not isinstance(text, str):
        return []

    breakdown = []
    for char in text:
        breakdown.append({
            'char': char,
            'name': unicodedata.name(char, 'UNNAMED'),
            'category': unicodedata.category(char),
            'codepoint': f'U+{ord(char):04X}',
            'is_combining': unicodedata.category(char).startswith('M')
        })
    return breakdown


def find_problematic_characters(df, text_columns):
    """
    Find all characters that would be affected by NFC normalization.

    Returns:
        dict: Character frequency maps and examples
    """
    affected_chars = defaultdict(lambda: {'count': 0, 'examples': []})
    combining_marks = defaultdict(lambda: {'count': 0, 'examples': []})

    for col in text_columns:
        if col not in df.columns:
            continue

        for idx, text in df[col].items():
            if pd.isna(text):
                continue

            # Check if normalization would change this text
            nfc_form = unicodedata.normalize('NFC', text)
            if text != nfc_form:
                # Find the specific characters involved
                breakdown = get_character_breakdown(text)
                for char_info in breakdown:
                    if char_info['is_combining']:
                        key = f"{char_info['char']} ({char_info['name']})"
                        combining_marks[key]['count'] += 1
                        if len(combining_marks[key]['examples']) < 5:
                            combining_marks[key]['examples'].append({
                                'row_id': idx,
                                'column': col,
                                'context': text[:50]
                            })

                    # Track all characters in affected rows
                    if char_info['char'] not in [' ', ',', '.', '!', '?']:
                        key = f"{char_info['char']} ({char_info['codepoint']})"
                        affected_chars[key]['count'] += 1

    return {
        'affected_chars': dict(affected_chars),
        'combining_marks': dict(combining_marks)
    }


def analyze_dataset(input_file, text_columns=None):
    """
    Main analysis function for the dataset.

    Args:
        input_file: Path to input CSV
        text_columns: List of columns to analyze (default: ['French', 'Translation'])
    """
    if text_columns is None:
        text_columns = ['French', 'Translation']

    print("=" * 80)
    print("Unicode Normalization Analysis for Aja-French Dataset")
    print("=" * 80)
    print(f"\nAnalyzing: {input_file}")
    print(f"Text columns: {', '.join(text_columns)}\n")

    # Read dataset
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print(f"✓ Loaded dataset: {len(df)} rows")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Analyze each text column
    results = {}
    for col in text_columns:
        if col not in df.columns:
            print(f"⚠ Warning: Column '{col}' not found in dataset")
            continue

        print(f"\n{'─' * 80}")
        print(f"Analyzing column: {col}")
        print('─' * 80)

        rows_with_issues = 0
        total_char_diff = 0
        examples = []

        for idx, text in df[col].items():
            analysis = analyze_text_unicode(text)
            if analysis and analysis['would_change']:
                rows_with_issues += 1
                total_char_diff += abs(analysis['original_len'] - analysis['nfc_len'])

                if len(examples) < 10:
                    examples.append({
                        'row_id': idx,
                        'original': text,
                        'nfc': analysis['nfc_form'],
                        'length_diff': analysis['original_len'] - analysis['nfc_len']
                    })

        results[col] = {
            'total_rows': len(df),
            'rows_with_issues': rows_with_issues,
            'percentage': (rows_with_issues / len(df)) * 100,
            'total_char_diff': total_char_diff,
            'examples': examples
        }

        print(f"  Total rows: {results[col]['total_rows']:,}")
        print(f"  Rows that would change: {results[col]['rows_with_issues']:,} ({results[col]['percentage']:.2f}%)")
        print(f"  Total character length difference: {results[col]['total_char_diff']}")

        if examples:
            print(f"\n  Sample changes (showing up to 10):")
            for i, ex in enumerate(examples[:5], 1):
                print(f"\n  Example {i} (Row {ex['row_id']}):")
                print(f"    Before: {ex['original'][:80]}")
                print(f"    After:  {ex['nfc'][:80]}")
                print(f"    Length: {len(ex['original'])} → {len(ex['nfc'])} (diff: {ex['length_diff']})")

    # Find problematic characters across all columns
    print(f"\n{'=' * 80}")
    print("Character-Level Analysis")
    print('=' * 80)

    char_analysis = find_problematic_characters(df, text_columns)

    if char_analysis['combining_marks']:
        print("\n🔍 Combining Marks Found (these cause composed/decomposed issues):")
        print("─" * 80)
        for char, info in sorted(char_analysis['combining_marks'].items(),
                                key=lambda x: x[1]['count'], reverse=True)[:20]:
            print(f"\n  {char}")
            print(f"    Occurrences: {info['count']}")
            if info['examples']:
                print(f"    Example contexts:")
                for ex in info['examples'][:3]:
                    print(f"      Row {ex['row_id']} ({ex['column']}): {ex['context']}")

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary & Recommendations")
    print('=' * 80)

    total_affected = sum(r['rows_with_issues'] for r in results.values())
    if total_affected > 0:
        print(f"\n✓ NFC normalization RECOMMENDED")
        print(f"  • {total_affected:,} total rows will be normalized across all columns")
        print(f"  • This will ensure consistent Unicode representation")
        print(f"  • Critical for ML/NLP tasks with low-resource languages")
        print(f"\n  Next step: Run normalize_unicode.py to apply NFC normalization")
    else:
        print(f"\n✓ Dataset is already in NFC form")
        print(f"  • No normalization needed")
        print(f"  • All characters are consistently represented")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Default file path
    default_input = "../10_000_for_data_paper_LREC_cleaned_v2.csv"

    # Allow command-line argument for input file
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input

    # Run analysis
    analyze_dataset(input_file, text_columns=['French', 'Translation'])
