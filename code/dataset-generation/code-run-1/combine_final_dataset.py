# combine_final_dataset.py
import pandas as pd

print("📦 Combining all modules into final dataset...\n")

# Load all modules
df_m1 = pd.read_csv('module1_base.csv')
df_m2 = pd.read_csv('module2_negation.csv')
df_m3 = pd.read_csv('module3_past.csv')
df_m4 = pd.read_csv('module4_future.csv')
df_m5 = pd.read_csv('module5_questions_dedup.csv')  # Deduplicated version

print("📊 Module sizes:")
print(f"   Module 1 (Present):   {len(df_m1)} sentences")
print(f"   Module 2 (Negation):  {len(df_m2)} sentences")
print(f"   Module 3 (Past):      {len(df_m3)} sentences")
print(f"   Module 4 (Future):    {len(df_m4)} sentences")
print(f"   Module 5 (Questions): {len(df_m5)} sentences")

# Combine all modules
df_full = pd.concat([df_m1, df_m2, df_m3, df_m4, df_m5], ignore_index=True)

# Remove any ERROR rows (if any)
original_count = len(df_full)
df_full = df_full[df_full['french'] != 'ERROR']
if len(df_full) < original_count:
    print(f"\n⚠️  Removed {original_count - len(df_full)} ERROR sentences")

# Check for duplicates across ALL modules
overall_dups = df_full.duplicated('french').sum()
if overall_dups > 0:
    print(f"\n⚠️  Found {overall_dups} duplicates across all modules")
    df_full = df_full.drop_duplicates(subset='french', keep='first')
    print(f"   After removing: {len(df_full)} unique sentences")

# Add metadata
df_full['date_generated'] = pd.Timestamp.now().strftime('%Y-%m-%d')

# Reset index and create sequential IDs
df_full = df_full.reset_index(drop=True)

print(f"\n✅ Total unique sentences: {len(df_full)}")
print(f"\n📊 Breakdown by module:")
for module in sorted(df_full['module'].unique()):
    count = len(df_full[df_full['module'] == module])
    print(f"   {module}: {count} sentences")

# ============================================
# SAVE 1: Full CSV with all metadata
# ============================================
csv_filename = 'ADJA_FRENCH_FULL_DATASET.csv'
df_full.to_csv(csv_filename, index=False)
print(f"\n💾 Saved full CSV: {csv_filename}")

# ============================================
# SAVE 2: Excel for translators (with columns)
# ============================================
# Create translator-friendly version with specific columns
translation_df = df_full[[
    'sentence_id',
    'module', 
    'french',
    'base_sentence_id'
]].copy()

# Add empty columns for translation
translation_df['adja_translation'] = ''
translation_df['translator_notes'] = ''
translation_df['confidence'] = ''  # high/medium/low

excel_filename = 'ADJA_TRANSLATION_SPREADSHEET.xlsx'
translation_df.to_excel(excel_filename, index=False, engine='openpyxl')
print(f"💾 Saved Excel: {excel_filename}")

# ============================================
# SAVE 3: Simple numbered TXT file (JUST sentences)
# ============================================
txt_filename = 'ADJA_FRENCH_SENTENCES.txt'
with open(txt_filename, 'w', encoding='utf-8') as f:
    for idx, row in df_full.iterrows():
        # Write numbered sentence
        f.write(f"{idx + 1}. {row['french']}\n")

print(f"💾 Saved numbered TXT: {txt_filename}")

# ============================================
# SAVE 4: TXT file organized by module
# ============================================
txt_by_module_filename = 'ADJA_FRENCH_SENTENCES_BY_MODULE.txt'
with open(txt_by_module_filename, 'w', encoding='utf-8') as f:
    for module in sorted(df_full['module'].unique()):
        module_df = df_full[df_full['module'] == module]
        
        f.write("=" * 60 + "\n")
        f.write(f"{module.upper()} ({len(module_df)} sentences)\n")
        f.write("=" * 60 + "\n\n")
        
        for idx, row in module_df.iterrows():
            global_num = idx + 1
            f.write(f"{global_num}. {row['french']}\n")
        
        f.write("\n\n")

print(f"💾 Saved organized TXT: {txt_by_module_filename}")

# ============================================
# SAVE 5: Translation template TXT (with space for Adja)
# ============================================
template_filename = 'ADJA_TRANSLATION_TEMPLATE.txt'
with open(template_filename, 'w', encoding='utf-8') as f:
    f.write("ADJA TRANSLATION TEMPLATE\n")
    f.write("=" * 60 + "\n")
    f.write("Instructions: Write the Adja translation after the arrow (→)\n")
    f.write("=" * 60 + "\n\n")
    
    for idx, row in df_full.iterrows():
        f.write(f"{idx + 1}. {row['french']}\n")
        f.write(f"   → \n\n")  # Space for translation

print(f"💾 Saved translation template: {template_filename}")

# ============================================
# Statistics Summary
# ============================================
print("\n" + "=" * 60)
print("📊 FINAL DATASET STATISTICS")
print("=" * 60)

print(f"\nTotal sentences: {len(df_full)}")
print(f"Unique French sentences: {df_full['french'].nunique()}")
print(f"Unique verbs: {df_full['verb'].nunique()}")
print(f"Unique pronouns: {df_full['pronoun'].nunique()}")

print(f"\nSentence length statistics:")
df_full['word_count'] = df_full['french'].str.split().str.len()
print(f"   Mean: {df_full['word_count'].mean():.1f} words")
print(f"   Min: {df_full['word_count'].min()} words")
print(f"   Max: {df_full['word_count'].max()} words")

print(f"\nStructure distribution:")
for structure in sorted(df_full['structure'].unique()):
    count = len(df_full[df_full['structure'] == structure])
    pct = count / len(df_full) * 100
    print(f"   {structure}: {count} ({pct:.1f}%)")

# ============================================
# Create README
# ============================================
readme_filename = 'DATASET_README.txt'
readme_content = f"""
ADJA-FRENCH PARALLEL DATASET
============================
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}

OVERVIEW
--------
Total sentences: {len(df_full)}
Language pair: French → Adja
Purpose: Neural Machine Translation training

STRUCTURE
---------
Module 1 - Present Tense ({len(df_m1)} sentences)
  Base affirmative sentences in present tense
  Example: "Je mange du riz"

Module 2 - Negation ({len(df_m2)} sentences)
  Negated versions of Module 1 sentences
  Example: "Je ne mange pas du riz"
  Links to: Module 1 (minimal pairs)

Module 3 - Past Tense ({len(df_m3)} sentences)
  Past tense versions of Module 1 sentences
  Example: "J'ai mangé du riz"
  Links to: Module 1 (minimal pairs)

Module 4 - Future Tense ({len(df_m4)} sentences)
  Future tense versions of Module 1 sentences
  Example: "Je vais manger du riz"
  Links to: Module 1 (minimal pairs)

Module 5 - Questions ({len(df_m5)} sentences, deduplicated)
  Yes/no and wh-questions based on Module 1
  Example: "Est-ce que je mange du riz ?"
  Links to: Module 1 (minimal pairs)

VOCABULARY
----------
Pronouns: je, tu, il, elle, nous, vous, ils, elles
Verbs: manger, boire, voir, aller, venir, faire, avoir, prendre, donner, vouloir
Objects: ~40+ common nouns (foods, places, people, things)

FILES
-----
1. ADJA_FRENCH_FULL_DATASET.csv
   - Complete dataset with all metadata
   - Columns: sentence_id, module, french, pronoun, verb, object, structure, 
     base_sentence_id, date_generated
   
2. ADJA_TRANSLATION_SPREADSHEET.xlsx
   - Excel file for translators
   - Includes empty columns for Adja translation and notes
   
3. ADJA_FRENCH_SENTENCES.txt
   - Simple numbered list of French sentences only
   - For quick reference or simple translation workflow
   
4. ADJA_FRENCH_SENTENCES_BY_MODULE.txt
   - Sentences organized by module
   - Helpful for understanding dataset structure
   
5. ADJA_TRANSLATION_TEMPLATE.txt
   - Template with space for writing translations
   - Format: "1. French sentence → [space for Adja]"

TRANSLATION GUIDELINES
----------------------
- Translate naturally, not word-for-word
- Be consistent across related sentences
- Sentences in Modules 2-5 are transformations of Module 1
- Try to maintain similar structure when translating related sentences

MINIMAL PAIRS
-------------
This dataset uses minimal pairs methodology:
- Each sentence in Modules 2-5 links back to a Module 1 base sentence
- Only ONE grammatical feature changes at a time
- This helps ML models learn specific grammatical patterns

Example minimal pair set:
  Base:     "Je mange du riz" (M1_0001)
  Negation: "Je ne mange pas du riz" (M2_0001) → links to M1_0001
  Past:     "J'ai mangé du riz" (M3_0001) → links to M1_0001
  Future:   "Je vais manger du riz" (M4_0001) → links to M1_0001
  Question: "Est-ce que je mange du riz ?" (M5_0001) → links to M1_0001

CONTACT
-------
Questions about this dataset: [Your contact info]

CITATION
--------
If you use this dataset in research, please cite:
[Will add citation info after paper publication]
"""

with open(readme_filename, 'w', encoding='utf-8') as f:
    f.write(readme_content)

print(f"💾 Saved README: {readme_filename}")

# ============================================
# Final summary
# ============================================
print("\n" + "=" * 60)
print("🎉 ALL FILES GENERATED!")
print("=" * 60)
print("\nFiles created:")
print(f"  1. {csv_filename} - Full dataset (CSV)")
print(f"  2. {excel_filename} - For translators (Excel)")
print(f"  3. {txt_filename} - Simple numbered list (TXT)")
print(f"  4. {txt_by_module_filename} - Organized by module (TXT)")
print(f"  5. {template_filename} - Translation template (TXT)")
print(f"  6. {readme_filename} - Documentation (TXT)")

print("\n📧 SEND TO TRANSLATORS:")
print(f"  - {excel_filename} (if they prefer spreadsheets)")
print(f"  - OR {txt_filename} (if they prefer simple text)")
print(f"  - AND {readme_filename} (instructions)")

print(f"\n💰 Translation estimate:")
print(f"  Total sentences: {len(df_full)}")
print(f"  At 50 sentences/hour: {len(df_full)/50:.1f} hours")
print(f"  At $10/hour: ${len(df_full)/50*10:.2f}")
print(f"  At $0.10/sentence: ${len(df_full)*0.10:.2f}")

print("\n✅ Ready to send!")