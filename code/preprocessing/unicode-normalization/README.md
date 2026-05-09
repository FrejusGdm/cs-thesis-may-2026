# Unicode NFC Normalization for Aja-French Dataset

This directory contains scripts for normalizing your Aja-French translation dataset to Unicode NFC (Normal Form Composed) format. This is a critical preprocessing step for low-resource language NLP tasks.

## Why Unicode Normalization Matters

### The Problem

Unicode allows the same visual character to be represented in multiple ways:

- **Composed (NFC)**: `é` = single character (U+00E9)
- **Decomposed (NFD)**: `e` + combining acute accent = two characters (U+0065 + U+0301)

Both look identical but are different at the byte level. This causes:

1. **Vocabulary fragmentation**: ML models treat them as different tokens
2. **Reduced training effectiveness**: Same phoneme splits across multiple representations
3. **Inconsistent tokenization**: Different representations produce different subword tokens
4. **Data quality issues**: Statistics become unreliable

### Why It's Critical for Aja

Your Aja dataset uses extensive IPA characters with tone marks:
- `ɔ̀` (open-o with low tone)
- `ɔ́` (open-o with high tone)
- `ɛ̀`, `ɛ́` (open-e variants)
- `ŋ` (eng), `ɖ` (d with tail)

These characters can exist in both composed and decomposed forms. NFC normalization ensures **consistent representation** across your entire dataset.

## Scripts Overview

### 1. `analyze_unicode_issues.py` - Diagnostic Tool

**Purpose**: Scan your dataset to identify Unicode normalization issues before fixing them.

**What it does**:
- Identifies rows with composed/decomposed character inconsistencies
- Shows which specific characters are affected
- Provides examples of changes that will occur
- Reports statistics on how many rows need normalization

**Usage**:
```bash
cd <LOCAL_PATH>
python analyze_unicode_issues.py
```

Or specify a custom input file:
```bash
python analyze_unicode_issues.py /path/to/your/dataset.csv
```

**Output**:
- Console report showing:
  - Number of rows affected per column
  - Specific combining marks found
  - Sample before/after examples
  - Recommendation on whether normalization is needed

### 2. `normalize_unicode.py` - Main Normalization Script

**Purpose**: Apply NFC normalization to your dataset and generate clean output.

**What it does**:
- Reads your CSV dataset
- Applies NFC normalization to French and Translation columns
- Creates a new normalized CSV (original preserved)
- Generates detailed change report CSV
- Prints summary statistics to console

**Usage**:
```bash
cd <LOCAL_PATH>
python normalize_unicode.py
```

Or specify a custom input file:
```bash
python normalize_unicode.py /path/to/your/dataset.csv
```

**Default behavior**:
- Input: `../10_000_for_data_paper_LREC_cleaned_v1.csv`
- Output: `../10_000_for_data_paper_LREC_cleaned_v1_normalized.csv`
- Change report: `normalization_changes_report.csv`

**Output files**:

1. **Normalized dataset CSV**: Your cleaned data ready for ML use
2. **Change report CSV**: Row-by-row details of all modifications with columns:
   - `row_id`: Original row number
   - `column`: Which column changed (French or Translation)
   - `original_text`: Text before normalization
   - `normalized_text`: Text after normalization
   - `original_length`: Character count before
   - `normalized_length`: Character count after
   - `char_difference`: Length difference (usually negative due to composition)

## Recommended Workflow

### Step 1: Analyze (Optional but Recommended)

First, run the diagnostic to see what will change:

```bash
python analyze_unicode_issues.py
```

Review the output to understand:
- How many rows will be affected
- Which characters are causing issues
- Examples of the changes

### Step 2: Normalize

Run the main normalization script:

```bash
python normalize_unicode.py
```

This will:
- Create `10_000_for_data_paper_LREC_cleaned_v1_normalized.csv`
- Generate `normalization_changes_report.csv`
- Print summary to console

### Step 3: Validate

Review the outputs:

1. **Check the summary statistics** in the console output
2. **Inspect the change report CSV** to verify changes are correct
3. **Spot-check a few rows** to ensure IPA characters remain correct
4. **Compare file sizes**: Normalized file is often slightly smaller

### Step 4: Integrate with Pipeline

Once validated, use the normalized dataset for:
1. ✓ Further cleaning (punctuation, spacing) - your existing scripts
2. ✓ ML/NLP training (translation models, etc.)
3. ✓ Tokenization and vocabulary building

## Integration with Your Existing Pipeline

Your current preprocessing flow should be:

```
Original Dataset
       ↓
1. Remove invisible characters
   (your existing: scripts/remove_invisible_char.py)
       ↓
2. Unicode NFC normalization  ← THIS STEP (NEW)
   (this directory: normalize_unicode.py)
       ↓
3. Punctuation & spacing normalization
   (your existing: punctuation_spacing_cleaner.py)
       ↓
Final Clean Dataset for ML
```

**Why this order matters**:
- Invisible characters should be removed first (they can interfere with normalization)
- Unicode normalization before punctuation/spacing (ensures character consistency)
- Punctuation/spacing last (operates on clean, normalized text)

## Technical Details

### What is NFC Normalization?

NFC (Normal Form Composed) is one of four Unicode normalization forms:

| Form | Description | Use Case |
|------|-------------|----------|
| **NFC** | Canonical Composition | **Recommended for NLP** - preserves meaning, composed form |
| NFD | Canonical Decomposition | Analysis tasks, base char separation |
| NFKC | Compatibility Composition | Aggressive normalization (may lose distinctions) |
| NFKD | Compatibility Decomposition | Rarely used |

**Why NFC for Aja**:
- Preserves linguistic meaning (tone marks stay with vowels)
- Standard for NLP preprocessing
- Compatible with most tokenizers
- Recommended by Stanford researchers for low-resource languages

### Character Encoding

All scripts use UTF-8 encoding throughout:
- Input: UTF-8
- Processing: Python 3 native Unicode strings
- Output: UTF-8 with BOM removed

### IPA Character Safety

The scripts are designed to safely handle IPA characters:
- Tone-marked vowels: `ɔ̀`, `ɔ́`, `ɛ̀`, `ɛ́`
- Special consonants: `ŋ`, `ɖ`
- Combining diacritics: grave, acute, nasal marks

NFC composition **preserves phonetic meaning** while ensuring consistency.

## Validation & Quality Checks

After normalization, verify:

1. **No data loss**: Row count should remain identical
2. **Character integrity**: Spot-check IPA characters still appear correct
3. **Length reduction**: Normalized text is often slightly shorter (combining marks composed)
4. **Consistency**: All instances of same phoneme now use same Unicode representation

## Troubleshooting

### "No changes to report"

This means your dataset is already in NFC form. No action needed.

### "Column not found" warning

Verify your CSV has columns named exactly `French` and `Translation` (case-sensitive).

### Large number of changes

This is expected! If 30-50% of rows change, it indicates your dataset had significant composed/decomposed inconsistency. This validates the need for normalization.

### Character appears different after normalization

Very rare, but if a character looks different:
1. Check the change report CSV for that specific row
2. Verify it's actually different (not just a font rendering issue)
3. Compare Unicode codepoints before/after

## References & Background

- **Unicode Standard**: [Unicode Normalization Forms](https://unicode.org/reports/tr15/)
- **Python docs**: `unicodedata.normalize()` function
- **Best practices**: Stanford NLP research on low-resource languages
- **Your notes**: `../notes.txt` (lines 5-7 discuss normalization importance)

## File Locations

- **This directory**: `<LOCAL_PATH>`
- **Input dataset**: `../10_000_for_data_paper_LREC_cleaned_v1.csv`
- **Output dataset**: `../10_000_for_data_paper_LREC_cleaned_v1_normalized.csv`
- **Change report**: `./normalization_changes_report.csv`

## Questions?

For more context on why this is important, see your notes in:
`<LOCAL_PATH>`

Key points from Stanford researcher:
- Spaces matter (handled by your other scripts)
- Punctuation consistency matters (handled by your other scripts)
- **Unicode formalization (NFC) matters** ← THIS DIRECTORY

---

**Created**: 2025-11-19
**Purpose**: Unicode NFC normalization for Aja-French translation dataset
**Context**: Part A2 of data cleaning pipeline (see notes.txt)
