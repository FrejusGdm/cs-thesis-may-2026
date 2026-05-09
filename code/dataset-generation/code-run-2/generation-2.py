# generate_simple_modules.py
import pandas as pd

# Load Module 1
df_module1 = pd.read_csv('module1_base.csv')

print(f"📂 Loaded {len(df_module1)} base sentences\n")

# ============================================
# MODULE 2: NEGATION (Pure Python)
# ============================================

def negate_french_sentence(sentence, pronoun):
    """
    Add ne...pas negation to French sentence
    Handles special cases like j'ai, je vais, etc.
    """
    words = sentence.split()
    
    # Handle contractions like j'ai, j'aime
    if words[0].startswith("j'"):
        # j'ai → je n'ai pas
        verb = words[0][2:]  # Remove j'
        negated = f"je n'{verb} pas {' '.join(words[1:])}"
    else:
        pronoun = words[0]
        verb = words[1]
        rest = ' '.join(words[2:]) if len(words) > 2 else ''
        
        # Check if verb starts with vowel → use n' instead of ne
        if verb[0] in 'aeiouhéèêàô':
            negated = f"{pronoun} n'{verb} pas {rest}".strip()
        else:
            negated = f"{pronoun} ne {verb} pas {rest}".strip()
    
    return negated

print("⚙️  Generating Module 2: Negation...")

df_module2 = df_module1.copy()
df_module2['sentence_id'] = df_module2['sentence_id'].str.replace('M1_', 'M2_')
df_module2['module'] = 'module2_negation'
df_module2['structure'] = 'negative_present'
df_module2['base_sentence_id'] = df_module1['sentence_id']

# Apply negation
df_module2['french'] = df_module2.apply(
    lambda row: negate_french_sentence(row['french'], row['pronoun']), 
    axis=1
)

print(f"✅ Generated {len(df_module2)} negated sentences")
print("\n📝 Sample negations:")
for i in range(5):
    base = df_module1.iloc[i]['french']
    neg = df_module2.iloc[i]['french']
    print(f"  {base} → {neg}")

# Save
df_module2.to_csv('module2_negation.csv', index=False)
print(f"\n💾 Saved to module2_negation.csv")


# ============================================
# MODULE 5: PLURAL (Python with simple rules)
# ============================================

PLURAL_TRANSFORMATIONS = {
    # Singular → Plural
    'du riz': 'des riz',  # Actually riz doesn't change, but des changes
    'du pain': 'des pains',
    'du poisson': 'des poissons',
    'de la viande': 'des viandes',
    'des fruits': 'des fruits',  # Already plural
    'des légumes': 'des légumes',  # Already plural
    
    "de l'eau": "des eaux",
    'du lait': 'des laits',
    'du thé': 'des thés',
    'du café': 'des cafés',
    'du jus': 'des jus',
    
    'mon ami': 'mes amis',
    'ma mère': 'mes mères',  # Not natural but systematic
    'mon père': 'mes pères',
    'le professeur': 'les professeurs',
    "l'enfant": 'les enfants',
    'le chien': 'les chiens',
    
    'à la maison': 'aux maisons',
    "à l'école": 'aux écoles',
    'au marché': 'aux marchés',
    'au village': 'aux villages',
    'au travail': 'aux travaux',
    
    'de la maison': 'des maisons',
    "de l'école": 'des écoles',
    'du marché': 'des marchés',
    'du village': 'des villages',
    'du travail': 'des travaux',
    
    'le travail': 'les travaux',
    'les devoirs': 'les devoirs',  # Already plural
    'la cuisine': 'les cuisines',
    'le ménage': 'les ménages',
    'le lit': 'les lits',
    
    'un livre': 'des livres',
    'une maison': 'des maisons',
    'un ami': 'des amis',
    'une sœur': 'des sœurs',
    'un frère': 'des frères',
    "de l'argent": "de l'argent",  # Uncountable
    
    'le livre': 'les livres',
    "l'eau": 'les eaux',
    'le pain': 'les pains',
    'le bus': 'les bus',
    'le stylo': 'les stylos',
    'le cadeau': 'les cadeaux',
}

# # Verb conjugations for plural subjects (nous, vous, ils, elles)
# # These are already correct in Module 1, but singular objects need to become plural

# def pluralize_object(sentence, object_singular):
#     """Replace singular object with plural form"""
#     if object_singular in PLURAL_TRANSFORMATIONS:
#         plural_object = PLURAL_TRANSFORMATIONS[object_singular]
#         return sentence.replace(object_singular, plural_object)
#     else:
#         # Fallback: try to add 's' or keep as is
#         return sentence

# print("\n⚙️  Generating Module 5: Plural Objects...")

# df_module5 = df_module1.copy()
# df_module5['sentence_id'] = df_module5['sentence_id'].str.replace('M1_', 'M5_')
# df_module5['module'] = 'module5_plural'
# df_module5['structure'] = 'affirmative_present_plural'
# df_module5['base_sentence_id'] = df_module1['sentence_id']

# # Apply pluralization
# df_module5['french'] = df_module5.apply(
#     lambda row: pluralize_object(row['french'], row['object']),
#     axis=1
# )

# # Update object column to reflect plural
# df_module5['object'] = df_module5.apply(
#     lambda row: PLURAL_TRANSFORMATIONS.get(row['object'], row['object']),
#     axis=1
# )

# print(f"✅ Generated {len(df_module5)} plural sentences")
# print("\n📝 Sample pluralizations:")
# for i in range(5):
#     base = df_module1.iloc[i]['french']
#     plural = df_module5.iloc[i]['french']
#     print(f"  {base} → {plural}")

# # Save
# df_module5.to_csv('module5_plural.csv', index=False)
# print(f"\n💾 Saved to module5_plural.csv")


# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("📊 SUMMARY - FREE MODULES GENERATED")
print("="*50)
print(f"Module 1 (Present):  {len(df_module1)} sentences ✅")
print(f"Module 2 (Negation): {len(df_module2)} sentences ✅")
# print(f"Module 5 (Plural):   {len(df_module5)} sentences ✅")
print(f"\nTotal so far: {len(df_module1) + len(df_module2)} sentences") # + len(df_module5)
print("\nNext: Use OpenAI API for Past, Future, Questions")
print("="*50)