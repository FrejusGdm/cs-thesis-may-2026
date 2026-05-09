#!/usr/bin/env python3
"""
French-Adja Dataset Punctuation and Spacing Normalizer (v2 - CORRECTED)

This script normalizes punctuation and spacing in parallel French-Adja corpus data.
It ensures consistency across source (French) and target (Translation) sentences.

KEY FIXES in v2:
- Does NOT confuse apostrophes with quotation marks
- Does NOT normalize quote characters (leaves « », " ", << >> as-is)
- DOES fix French when it's clearly wrong (e.g., missing ? on obvious questions)
- Only matches quote PRESENCE, not characters

Author: Dataset Cleaning Pipeline
Date: November 19, 2025
Version: 2.0 (Corrected)
"""

import csv
import re
from typing import Dict, List, Tuple
from collections import defaultdict
import sys

class PunctuationSpacingNormalizer:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = input_file
        self.output_dir = output_dir
        self.stats = defaultdict(int)
        self.changes_log = []
        self.flagged_cases = []

        # French question words that indicate a sentence should end with ?
        self.french_question_words = [
            'pourquoi', 'comment', 'est-ce', 'quel', 'quelle', 'quels', 'quelles',
            'où', 'quand', 'qui', 'que', 'quoi', 'combien', 'lequel', 'laquelle',
            'lesquels', 'lesquelles', 'avez-vous', 'as-tu', 'puis-je', 'peut-on',
            'devrait', 'veux-tu', 'voulez-vous', 'peux-tu', 'pouvez-vous',
            'sais-tu', 'savez-vous', 'vois-tu', 'voyez-vous', 'es-tu', 'êtes-vous',
            'a-t-il', 'a-t-elle', 'ont-ils', 'ont-elles', 'suis-je', 'sommes-nous',
            'y a-t-il', 'n\'y a-t-il'
        ]

    def is_likely_question(self, text: str) -> bool:
        """Check if French text is likely a question based on question words"""
        text_lower = text.lower().strip()

        for question_word in self.french_question_words:
            # Check if starts with question word
            if text_lower.startswith(question_word):
                return True
            # Check for inverted questions like "Va-t-il"
            if question_word.endswith('-') and question_word in text_lower[:30]:
                return True

        return False

    def detect_actual_quotes(self, text: str) -> bool:
        """
        Detect if text contains ACTUAL quotation marks, not apostrophes.

        Quotation marks: « », " ", << >>, " " (curly quotes)
        NOT apostrophes: ' in n'a, l'orientation, etc.
        """
        # French guillemets
        if '«' in text or '»' in text:
            return True

        # Angle bracket quotes (used in some corpora)
        if '<<' in text or '>>' in text:
            return True

        # Straight double quotes " (but must be paired, not just one)
        double_quote_count = text.count('"')
        if double_quote_count >= 2:
            return True

        # Curly quotes " "
        if '"' in text or '"' in text:
            return True

        return False

    def normalize_exclamation_char(self, text: str) -> str:
        """Standardize exclamation mark characters (ǃ → !)"""
        # Replace non-standard exclamation marks
        text = text.replace('ǃ', '!')  # Latin letter retroflex click
        text = text.replace('\u01c3', '!')  # Another variant
        return text

    def remove_multiple_spaces(self, text: str) -> str:
        """Replace multiple consecutive spaces with single space"""
        return re.sub(r' {2,}', ' ', text)

    def normalize_spacing_around_punctuation(self, text: str) -> str:
        """
        Remove space before punctuation: . , ! ? ; :
        Ensure single space after punctuation (except at end)
        """
        # Remove space before punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)

        # Ensure single space after punctuation (if followed by non-space, non-punctuation)
        text = re.sub(r'([.,!?;:])([^\s.,!?;:\"\'\»\"\'])', r'\1 \2', text)

        # Fix multiple spaces that might have been created
        text = re.sub(r' {2,}', ' ', text)

        return text

    def get_final_punctuation(self, text: str) -> str:
        """Extract the final punctuation mark(s) from text"""
        text = text.strip()
        if not text:
            return ''

        # Check for punctuation at end (possibly with quotes/guillemets after)
        match = re.search(r'[.!?]+[»\"\"\'\s]*$', text)
        if match:
            punct = match.group()
            # Extract just the punctuation, remove quotes/spaces
            return re.sub(r'[»\"\"\'\s]', '', punct)
        return ''

    def set_final_punctuation(self, text: str, punctuation: str) -> str:
        """Set the final punctuation of text, preserving any trailing quotes/guillemets"""
        text = text.strip()
        if not text:
            return text

        # Check if there are trailing quotes/guillemets
        trailing_chars = ''
        match = re.search(r'[»\"\"\']$', text)
        if match:
            trailing_chars = match.group()
            text = text[:-len(trailing_chars)].strip()

        # Remove existing final punctuation
        text = re.sub(r'[.!?]+$', '', text).strip()

        # Add new punctuation
        result = text + punctuation

        # Add back trailing quote if there was one
        if trailing_chars:
            result += trailing_chars

        return result

    def match_punctuation(self, french: str, translation: str, row_id: str) -> Tuple[str, str]:
        """
        Ensure punctuation matches between French and Translation.
        BIDIRECTIONAL: Can fix both French and Translation.
        """
        french_punct = self.get_final_punctuation(french)
        translation_punct = self.get_final_punctuation(translation)

        original_french = french
        original_translation = translation

        # Rule 1: If French ends with ?, Translation must end with ?
        if '?' in french_punct and '?' not in translation_punct:
            translation = self.set_final_punctuation(translation, '?')
            self.stats['question_marks_added_to_translation'] += 1
            self.changes_log.append({
                'id': row_id,
                'type': 'question_mark_added_to_translation',
                'column': 'Translation',
                'french': french,
                'original_translation': original_translation,
                'new_translation': translation
            })
            self.flagged_cases.append({
                'id': row_id,
                'reason': 'Question mark added to Translation',
                'french': french,
                'original_translation': original_translation,
                'new_translation': translation
            })

        # Rule 2: If Translation ends with ? but French doesn't
        elif '?' in translation_punct and '?' not in french_punct:
            # Check if French is clearly a question
            if self.is_likely_question(french):
                # Fix French by adding ?
                french = self.set_final_punctuation(french, '?')
                self.stats['question_marks_added_to_french'] += 1
                self.changes_log.append({
                    'id': row_id,
                    'type': 'question_mark_added_to_french',
                    'column': 'French',
                    'original_french': original_french,
                    'new_french': french,
                    'translation': translation
                })
                self.flagged_cases.append({
                    'id': row_id,
                    'reason': 'Question mark added to French (was clearly a question)',
                    'original_french': original_french,
                    'new_french': french,
                    'translation': translation
                })
            else:
                # Not clearly a question - just flag it
                self.stats['translation_has_question_french_doesnt'] += 1
                self.flagged_cases.append({
                    'id': row_id,
                    'reason': 'Translation has ? but French does not (and not clearly a question)',
                    'french': french,
                    'translation': translation
                })

        # Rule 3: If French ends with !, Translation must end with !
        elif '!' in french_punct and '!' not in translation_punct:
            translation = self.set_final_punctuation(translation, '!')
            self.stats['exclamation_marks_added_to_translation'] += 1
            self.changes_log.append({
                'id': row_id,
                'type': 'exclamation_mark_added_to_translation',
                'column': 'Translation',
                'french': french,
                'original_translation': original_translation,
                'new_translation': translation
            })

        # Rule 4: If Translation ends with ! but French doesn't
        elif '!' in translation_punct and '!' not in french_punct:
            # Add ! to French too (exclamations should match)
            french = self.set_final_punctuation(french, '!')
            self.stats['exclamation_marks_added_to_french'] += 1
            self.changes_log.append({
                'id': row_id,
                'type': 'exclamation_mark_added_to_french',
                'column': 'French',
                'original_french': original_french,
                'new_french': french,
                'translation': translation
            })

        return french, translation

    def match_quotes(self, french: str, translation: str, row_id: str) -> Tuple[str, str]:
        """
        Match quote PRESENCE (not characters) between French and Translation.
        Does NOT normalize quote characters - leaves them as-is.
        Does NOT confuse apostrophes with quotes.
        """
        french_has_quotes = self.detect_actual_quotes(french)
        translation_has_quotes = self.detect_actual_quotes(translation)

        original_french = french
        original_translation = translation

        # If French has quotes but Translation doesn't
        if french_has_quotes and not translation_has_quotes:
            # Detect which quote style French uses
            if '«' in french and '»' in french:
                # Use guillemets for Translation too
                translation = f'«{translation}»'
            elif '<<' in french and '>>' in french:
                # Use angle brackets
                translation = f'<<{translation}>>'
            elif '"' in french or '"' in french:
                # Use curly quotes
                translation = f'"{translation}"'
            else:
                # Default to straight quotes
                translation = f'"{translation}"'

            self.stats['quotes_added_to_translation'] += 1
            self.changes_log.append({
                'id': row_id,
                'type': 'quotes_added_to_translation',
                'column': 'Translation',
                'french': french,
                'original_translation': original_translation,
                'new_translation': translation
            })

        # If Translation has quotes but French doesn't
        elif translation_has_quotes and not french_has_quotes:
            # Add quotes to French using same style as Translation
            if '«' in translation and '»' in translation:
                french = f'«{french}»'
            elif '<<' in translation and '>>' in translation:
                french = f'<<{french}>>'
            elif '"' in translation or '"' in translation:
                french = f'"{french}"'
            else:
                french = f'"{french}"'

            self.stats['quotes_added_to_french'] += 1
            self.changes_log.append({
                'id': row_id,
                'type': 'quotes_added_to_french',
                'column': 'French',
                'original_french': original_french,
                'new_french': french,
                'translation': translation
            })

        return french, translation

    def normalize_text(self, text: str) -> str:
        """Apply spacing normalizations to text (NOT quote character normalization!)"""
        # Step 1: Trim leading/trailing spaces
        text = text.strip()

        # Step 2: Normalize exclamation characters (ǃ → !)
        text = self.normalize_exclamation_char(text)

        # Step 3: Remove multiple consecutive spaces
        text = self.remove_multiple_spaces(text)

        # Step 4: Normalize spacing around punctuation
        text = self.normalize_spacing_around_punctuation(text)

        # Step 5: Final trim
        text = text.strip()

        return text

    def track_spacing_changes(self, original: str, normalized: str, row_id: str, column: str):
        """Track what spacing changes were made"""
        if original != normalized:
            # Count double spaces removed
            original_double_spaces = len(re.findall(r' {2,}', original))
            if original_double_spaces > 0:
                self.stats[f'{column}_double_spaces_removed'] += original_double_spaces

            # Count spaces before punctuation removed
            spaces_before_punct = len(re.findall(r'\s+[.,!?;:]', original))
            if spaces_before_punct > 0:
                self.stats[f'{column}_spaces_before_punct_removed'] += spaces_before_punct

            # Track if exclamation char was normalized
            if 'ǃ' in original:
                self.stats[f'{column}_exclamation_normalized'] += 1

    def process_dataset(self):
        """Process the entire dataset"""
        rows_processed = 0
        rows_with_changes = 0

        output_rows = []

        print(f"Reading dataset from: {self.input_file}")

        with open(self.input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames

            for row in reader:
                rows_processed += 1
                row_id = row.get('ID', str(rows_processed))

                original_french = row['French']
                original_translation = row['Translation']

                # Step 1: Normalize spacing (NOT quote characters!)
                french_normalized = self.normalize_text(original_french)
                translation_normalized = self.normalize_text(original_translation)

                # Track spacing changes
                self.track_spacing_changes(original_french, french_normalized, row_id, 'french')
                self.track_spacing_changes(original_translation, translation_normalized, row_id, 'translation')

                # Step 2: Match punctuation between French and Translation
                french_final, translation_final = self.match_punctuation(
                    french_normalized, translation_normalized, row_id
                )

                # Step 3: Match quote PRESENCE (not characters)
                french_final, translation_final = self.match_quotes(
                    french_final, translation_final, row_id
                )

                # Track if row changed
                if (original_french != french_final or original_translation != translation_final):
                    rows_with_changes += 1

                # Store the cleaned row
                new_row = row.copy()
                new_row['French'] = french_final
                new_row['Translation'] = translation_final
                output_rows.append(new_row)

        self.stats['total_rows'] = rows_processed
        self.stats['rows_with_changes'] = rows_with_changes

        return output_rows, fieldnames

    def write_cleaned_dataset(self, rows, fieldnames, version='v2'):
        """Write cleaned dataset to CSV"""
        output_file = f"{self.output_dir}/10_000_for_data_paper_LREC_cleaned_{version}.csv"

        print(f"\nWriting cleaned dataset to: {output_file}")

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"✓ Wrote {len(rows)} rows to cleaned dataset")
        return output_file

    def write_report(self, version='v2'):
        """Write detailed report of all changes"""
        report_file = f"{self.output_dir}/cleaning_report_{version}.txt"

        print(f"\nWriting cleaning report to: {report_file}")

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FRENCH-ADJA DATASET CLEANING REPORT (v2 - CORRECTED)\n")
            f.write("Punctuation and Spacing Normalization\n")
            f.write("=" * 80 + "\n\n")

            f.write("KEY FIXES IN v2:\n")
            f.write("-" * 80 + "\n")
            f.write("✓ Does NOT confuse apostrophes (n'a) with quotation marks\n")
            f.write("✓ Does NOT normalize quote characters (leaves « », \" \", << >> as-is)\n")
            f.write("✓ DOES fix French when clearly wrong (e.g., missing ? on questions)\n")
            f.write("✓ Only matches quote PRESENCE, not characters\n\n")

            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total rows processed: {self.stats['total_rows']}\n")
            f.write(f"Rows with changes: {self.stats['rows_with_changes']}\n")
            f.write(f"Percentage changed: {self.stats['rows_with_changes']/self.stats['total_rows']*100:.2f}%\n\n")

            # Punctuation changes
            f.write("PUNCTUATION CHANGES\n")
            f.write("-" * 80 + "\n")
            f.write(f"Question marks added to Translation: {self.stats.get('question_marks_added_to_translation', 0)}\n")
            f.write(f"Question marks added to French: {self.stats.get('question_marks_added_to_french', 0)}\n")
            f.write(f"Exclamation marks added to Translation: {self.stats.get('exclamation_marks_added_to_translation', 0)}\n")
            f.write(f"Exclamation marks added to French: {self.stats.get('exclamation_marks_added_to_french', 0)}\n")
            f.write(f"Quotes added to Translation: {self.stats.get('quotes_added_to_translation', 0)}\n")
            f.write(f"Quotes added to French: {self.stats.get('quotes_added_to_french', 0)}\n")
            f.write(f"Translation has ? but French doesn't (ambiguous): {self.stats.get('translation_has_question_french_doesnt', 0)}\n\n")

            # Spacing changes
            f.write("SPACING CHANGES\n")
            f.write("-" * 80 + "\n")
            f.write(f"French - Double spaces removed: {self.stats.get('french_double_spaces_removed', 0)}\n")
            f.write(f"French - Spaces before punctuation removed: {self.stats.get('french_spaces_before_punct_removed', 0)}\n")
            f.write(f"Translation - Double spaces removed: {self.stats.get('translation_double_spaces_removed', 0)}\n")
            f.write(f"Translation - Spaces before punctuation removed: {self.stats.get('translation_spaces_before_punct_removed', 0)}\n\n")

            # Character normalization
            f.write("CHARACTER NORMALIZATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"French - Exclamation marks normalized (ǃ → !): {self.stats.get('french_exclamation_normalized', 0)}\n")
            f.write(f"Translation - Exclamation marks normalized (ǃ → !): {self.stats.get('translation_exclamation_normalized', 0)}\n\n")

            # Detailed change log
            f.write("\n" + "=" * 80 + "\n")
            f.write("DETAILED CHANGE LOG\n")
            f.write("=" * 80 + "\n\n")

            # Group changes by type
            changes_by_type = defaultdict(list)
            for change in self.changes_log:
                changes_by_type[change['type']].append(change)

            for change_type, changes in changes_by_type.items():
                f.write(f"\n{change_type.upper().replace('_', ' ')} ({len(changes)} cases)\n")
                f.write("-" * 80 + "\n")
                for i, change in enumerate(changes[:50], 1):  # Show first 50 of each type
                    f.write(f"\n{i}. ID: {change['id']}\n")

                    if change['column'] == 'French':
                        f.write(f"   Original French: {change.get('original_french', 'N/A')}\n")
                        f.write(f"   New French: {change.get('new_french', 'N/A')}\n")
                        f.write(f"   Translation: {change.get('translation', 'N/A')}\n")
                    else:  # Translation
                        f.write(f"   French: {change.get('french', 'N/A')}\n")
                        f.write(f"   Original Translation: {change.get('original_translation', 'N/A')}\n")
                        f.write(f"   New Translation: {change.get('new_translation', 'N/A')}\n")

                if len(changes) > 50:
                    f.write(f"\n   ... and {len(changes) - 50} more cases\n")

        print(f"✓ Report written successfully")
        return report_file

    def write_flagged_cases(self, version='v2'):
        """Write cases that need manual review"""
        flagged_file = f"{self.output_dir}/flagged_for_review_{version}.csv"

        print(f"\nWriting flagged cases to: {flagged_file}")

        # Create consistent fieldnames for all flagged cases
        with open(flagged_file, 'w', encoding='utf-8', newline='') as f:
            fieldnames = ['id', 'reason', 'details']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for case in self.flagged_cases:
                # Format details based on what's available
                details = []
                if 'french' in case:
                    details.append(f"French: {case['french']}")
                if 'original_french' in case:
                    details.append(f"Original French: {case['original_french']}")
                if 'new_french' in case:
                    details.append(f"New French: {case['new_french']}")
                if 'translation' in case:
                    details.append(f"Translation: {case['translation']}")
                if 'original_translation' in case:
                    details.append(f"Original Translation: {case['original_translation']}")
                if 'new_translation' in case:
                    details.append(f"New Translation: {case['new_translation']}")

                writer.writerow({
                    'id': case['id'],
                    'reason': case['reason'],
                    'details': ' | '.join(details)
                })

        print(f"✓ Wrote {len(self.flagged_cases)} flagged cases")
        return flagged_file


def main():
    import os

    # Configuration
    input_file = "<LOCAL_PATH>"
    output_dir = "<LOCAL_PATH>"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("FRENCH-ADJA DATASET PUNCTUATION & SPACING NORMALIZER (v2 - CORRECTED)")
    print("=" * 80)
    print()
    print("KEY IMPROVEMENTS:")
    print("✓ Does NOT confuse apostrophes with quotes")
    print("✓ Does NOT normalize quote characters")
    print("✓ DOES fix French when clearly wrong")
    print("✓ Bidirectional punctuation fixing")
    print()

    # Initialize normalizer
    normalizer = PunctuationSpacingNormalizer(input_file, output_dir)

    # Process dataset
    print("Processing dataset...")
    cleaned_rows, fieldnames = normalizer.process_dataset()

    # Write outputs
    print("\n" + "=" * 80)
    print("GENERATING OUTPUTS")
    print("=" * 80)

    cleaned_file = normalizer.write_cleaned_dataset(cleaned_rows, fieldnames)
    report_file = normalizer.write_report()
    flagged_file = normalizer.write_flagged_cases()

    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nFiles generated:")
    print(f"  1. Cleaned dataset: {cleaned_file}")
    print(f"  2. Cleaning report: {report_file}")
    print(f"  3. Flagged cases: {flagged_file}")
    print(f"\nTotal rows processed: {normalizer.stats['total_rows']}")
    print(f"Rows with changes: {normalizer.stats['rows_with_changes']}")
    print(f"Cases flagged for review: {len(normalizer.flagged_cases)}")
    print()


if __name__ == "__main__":
    main()
