# deduplicate_module5.py
import pandas as pd

# Load Module 5
df = pd.read_csv('module5_questions.csv')

print(f"📊 Original Module 5: {len(df)} sentences")
print(f"   Duplicates found: {df.duplicated('french').sum()}")

# Show some duplicates before removing
duplicates = df[df.duplicated('french', keep=False)].sort_values('french')
if len(duplicates) > 0:
    print(f"\n📝 Examples of duplicates:")
    for french_text in duplicates['french'].unique()[:5]:
        count = len(duplicates[duplicates['french'] == french_text])
        print(f"   '{french_text}' appears {count} times")

# Remove duplicates, keep first occurrence
df_unique = df.drop_duplicates(subset='french', keep='first')

print(f"\n✅ After deduplication: {len(df_unique)} sentences")
print(f"   Removed: {len(df) - len(df_unique)} duplicates")

# Reassign sentence IDs to be sequential
df_unique = df_unique.reset_index(drop=True)
df_unique['sentence_id'] = [f'M5_{i+1:04d}' for i in range(len(df_unique))]

# Save
df_unique.to_csv('module5_questions_dedup.csv', index=False)
print(f"\n💾 Saved to: module5_questions_dedup.csv")

# Show stats by question type
print(f"\n📈 Question type distribution:")
print(df_unique['structure'].value_counts())

# Sample
print(f"\n📝 Sample questions after deduplication:")
for sent in df_unique.sample(min(10, len(df_unique)))['french'].tolist():
    print(f"   - {sent}")