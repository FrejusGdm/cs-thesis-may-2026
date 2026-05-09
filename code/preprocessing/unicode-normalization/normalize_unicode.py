#!/usr/bin/env python3
"""
Unicode NFC Normalization Script for Aja-French Dataset

This script normalizes all text in the dataset to Unicode NFC (Canonical Composition)
form to ensure consistent character representation. This is critical for:
- Low-resource language NLP tasks
- Consistent tokenization
- Reducing vocabulary fragmentation
- Handling IPA characters with tone marks

Recommended by Stanford researchers for linguistic dataset preprocessing.

Author: Created for Aja language dataset preprocessing
Date: 2025-11-19
"""

import pandas as pd
import unicodedata
from datetime import datetime
import sys
import os


class UnicodeNormalizer:
    """Handles NFC Unicode normalization with detailed tracking and reporting."""

    def __init__(self, input_file, text_columns=None):
        """
        Initialize the normalizer.

        Args:
            input_file: Path to input CSV file
            text_columns: List of columns to normalize (default: ['French', 'Translation'])
        """
        self.input_file = input_file
        self.text_columns = text_columns or ['French', 'Translation']
        self.df = None
        self.changes = []
        self.stats = {
            'total_rows': 0,
            'rows_changed': 0,
            'by_column': {}
        }

    def load_dataset(self):
        """Load the CSV dataset."""
        print(f"Loading dataset: {self.input_file}")
        try:
            self.df = pd.read_csv(self.input_file, encoding='utf-8')
            self.stats['total_rows'] = len(self.df)
            print(f"✓ Loaded {len(self.df):,} rows")
            return True
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False

    def normalize_text(self, text):
        """
        Normalize text to NFC form.

        Args:
            text: Input text string

        Returns:
            Normalized text in NFC form
        """
        if pd.isna(text):
            return text
        if not isinstance(text, str):
            return text

        return unicodedata.normalize('NFC', str(text))

    def normalize_column(self, column_name):
        """
        Normalize a specific column and track changes.

        Args:
            column_name: Name of the column to normalize
        """
        if column_name not in self.df.columns:
            print(f"⚠ Warning: Column '{column_name}' not found, skipping")
            return

        print(f"\nNormalizing column: {column_name}")
        changes_in_column = 0
        char_diff_total = 0

        for idx, original_text in self.df[column_name].items():
            if pd.isna(original_text):
                continue

            normalized_text = self.normalize_text(original_text)

            # Track if this row changed
            if original_text != normalized_text:
                changes_in_column += 1
                char_diff = len(original_text) - len(normalized_text)
                char_diff_total += abs(char_diff)

                # Record the change
                self.changes.append({
                    'row_id': idx,
                    'column': column_name,
                    'original_text': original_text,
                    'normalized_text': normalized_text,
                    'original_length': len(original_text),
                    'normalized_length': len(normalized_text),
                    'char_difference': char_diff
                })

                # Update the dataframe
                self.df.at[idx, column_name] = normalized_text

        # Update stats
        self.stats['by_column'][column_name] = {
            'rows_changed': changes_in_column,
            'char_diff_total': char_diff_total
        }

        print(f"  ✓ {changes_in_column:,} rows normalized")
        print(f"  ✓ Total character difference: {char_diff_total}")

    def normalize_all(self):
        """Normalize all specified text columns."""
        print("\n" + "=" * 80)
        print("Starting Unicode NFC Normalization")
        print("=" * 80)

        for column in self.text_columns:
            self.normalize_column(column)

        # Calculate total unique rows changed
        unique_rows_changed = len(set(change['row_id'] for change in self.changes))
        self.stats['rows_changed'] = unique_rows_changed

    def save_normalized_dataset(self, output_file):
        """
        Save the normalized dataset to a new CSV file.

        Args:
            output_file: Path for output CSV file
        """
        print(f"\nSaving normalized dataset: {output_file}")
        try:
            self.df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"✓ Saved successfully")
            return True
        except Exception as e:
            print(f"✗ Error saving dataset: {e}")
            return False

    def save_change_report(self, report_file):
        """
        Save detailed change report to CSV.

        Args:
            report_file: Path for change report CSV file
        """
        if not self.changes:
            print("\nNo changes to report - dataset already in NFC form")
            return

        print(f"\nSaving change report: {report_file}")
        try:
            changes_df = pd.DataFrame(self.changes)
            changes_df.to_csv(report_file, index=False, encoding='utf-8')
            print(f"✓ Saved {len(self.changes):,} change records")
            return True
        except Exception as e:
            print(f"✗ Error saving change report: {e}")
            return False

    def print_summary(self):
        """Print summary statistics to console."""
        print("\n" + "=" * 80)
        print("Normalization Summary")
        print("=" * 80)

        print(f"\nDataset: {self.input_file}")
        print(f"Total rows: {self.stats['total_rows']:,}")
        print(f"Rows with changes: {self.stats['rows_changed']:,} "
              f"({(self.stats['rows_changed'] / self.stats['total_rows'] * 100):.2f}%)")

        print("\nChanges by column:")
        for col, col_stats in self.stats['by_column'].items():
            print(f"  {col}:")
            print(f"    Rows normalized: {col_stats['rows_changed']:,}")
            print(f"    Character differences: {col_stats['char_diff_total']}")

        if self.changes:
            print("\nSample changes (first 5):")
            for i, change in enumerate(self.changes[:5], 1):
                print(f"\n  {i}. Row {change['row_id']} ({change['column']}):")
                print(f"     Before: {change['original_text'][:70]}")
                print(f"     After:  {change['normalized_text'][:70]}")
                print(f"     Length: {change['original_length']} → {change['normalized_length']}")

        print("\n" + "=" * 80)


def generate_output_filename(input_file, suffix='_normalized'):
    """
    Generate output filename based on input filename.

    Args:
        input_file: Original input file path
        suffix: Suffix to add before extension

    Returns:
        Output filename
    """
    base, ext = os.path.splitext(input_file)
    return f"{base}{suffix}{ext}"


def main():
    """Main execution function."""
    # Configuration
    default_input = "../10_000_for_data_paper_LREC_cleaned_v1.csv"
    input_file = sys.argv[1] if len(sys.argv) > 1 else default_input

    # Generate output filenames
    output_file = generate_output_filename(input_file, '_normalized')
    report_file = "normalization_changes_report.csv"

    print("=" * 80)
    print("Unicode NFC Normalization for Aja-French Dataset")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input:  {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Report: {report_file}")
    print(f"  Columns: French, Translation")
    print()

    # Create normalizer
    normalizer = UnicodeNormalizer(input_file, text_columns=['French', 'Translation'])

    # Load dataset
    if not normalizer.load_dataset():
        sys.exit(1)

    # Normalize all columns
    normalizer.normalize_all()

    # Save results
    normalizer.save_normalized_dataset(output_file)
    normalizer.save_change_report(report_file)

    # Print summary
    normalizer.print_summary()

    print("\n✓ Normalization complete!")
    print("\nNext steps:")
    print("  1. Review the change report for validation")
    print("  2. Use the normalized dataset for ML/NLP tasks")
    print("  3. Continue with punctuation/spacing normalization if needed")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
