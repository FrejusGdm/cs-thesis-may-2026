#!/usr/bin/env python3
"""
Validation script to compare original and cleaned datasets
Shows before/after examples with highlighting
"""

import csv

def compare_datasets():
    original_file = "<LOCAL_PATH>"
    cleaned_file = "<LOCAL_PATH>"

    # Read both files
    with open(original_file, 'r', encoding='utf-8') as f:
        original_rows = list(csv.DictReader(f))

    with open(cleaned_file, 'r', encoding='utf-8') as f:
        cleaned_rows = list(csv.DictReader(f))

    # Find interesting examples
    examples = []

    for orig, clean in zip(original_rows, cleaned_rows):
        if orig['French'] != clean['French'] or orig['Translation'] != clean['Translation']:
            row_id = orig.get('ID', '?')

            # Categorize the change
            change_types = []

            # Check for question mark
            if '?' in clean['Translation'] and '?' not in orig['Translation']:
                change_types.append('? added')

            # Check for exclamation
            if '!' in clean['Translation'] and '!' not in orig['Translation']:
                change_types.append('! added')

            # Check for quotes
            if '"' in clean['Translation'] and '"' not in orig['Translation']:
                change_types.append('quotes added')

            # Check for spacing
            if '  ' in orig['Translation'] and '  ' not in clean['Translation']:
                change_types.append('double spaces removed')

            if orig['Translation'] != clean['Translation']:
                examples.append({
                    'id': row_id,
                    'types': change_types,
                    'french_before': orig['French'],
                    'french_after': clean['French'],
                    'translation_before': orig['Translation'],
                    'translation_after': clean['Translation']
                })

    # Show examples by category
    print("=" * 100)
    print("VALIDATION: BEFORE & AFTER COMPARISON")
    print("=" * 100)
    print()

    # Question marks
    print("\n1. QUESTION MARKS ADDED TO TRANSLATION")
    print("-" * 100)
    question_examples = [e for e in examples if '? added' in e['types']][:10]
    for i, ex in enumerate(question_examples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   French: {ex['french_after']}")
        print(f"   BEFORE: {ex['translation_before']}")
        print(f"   AFTER:  {ex['translation_after']}")

    # Exclamations
    print("\n\n2. EXCLAMATION MARKS ADDED TO TRANSLATION")
    print("-" * 100)
    exclaim_examples = [e for e in examples if '! added' in e['types']][:10]
    for i, ex in enumerate(exclaim_examples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   French: {ex['french_after']}")
        print(f"   BEFORE: {ex['translation_before']}")
        print(f"   AFTER:  {ex['translation_after']}")

    # Quotes
    print("\n\n3. QUOTES ADDED TO TRANSLATION")
    print("-" * 100)
    quote_examples = [e for e in examples if 'quotes added' in e['types']][:10]
    for i, ex in enumerate(quote_examples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   French: {ex['french_after']}")
        print(f"   BEFORE: {ex['translation_before']}")
        print(f"   AFTER:  {ex['translation_after']}")

    # Spacing
    print("\n\n4. DOUBLE SPACES REMOVED")
    print("-" * 100)
    space_examples = [e for e in examples if 'double spaces removed' in e['types']][:10]
    for i, ex in enumerate(space_examples, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   BEFORE: {repr(ex['translation_before'])}")
        print(f"   AFTER:  {repr(ex['translation_after'])}")

    # French changes
    print("\n\n5. FRENCH COLUMN CHANGES (Spacing/Quotes normalized)")
    print("-" * 100)
    french_changes = [e for e in examples if e['french_before'] != e['french_after']][:10]
    for i, ex in enumerate(french_changes, 1):
        print(f"\n{i}. ID: {ex['id']}")
        print(f"   BEFORE: {repr(ex['french_before'])}")
        print(f"   AFTER:  {repr(ex['french_after'])}")

    print("\n\n" + "=" * 100)
    print(f"Total changes: {len(examples)}")
    print(f"Question marks added: {len(question_examples)}")
    print(f"Exclamation marks added: {len(exclaim_examples)}")
    print(f"Quotes added: {len(quote_examples)}")
    print(f"Spacing fixes: {len(space_examples)}")
    print("=" * 100)

if __name__ == "__main__":
    compare_datasets()
