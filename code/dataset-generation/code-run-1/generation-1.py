# generate_module1.py
import pandas as pd
import itertools

# Core vocabulary - KEEP IT SIMPLE AND HIGH-FREQUENCY
PRONOUNS = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles']

VERBS = {
    'manger': {'je': 'mange', 'tu': 'manges', 'il': 'mange', 'elle': 'mange', 
               'nous': 'mangeons', 'vous': 'mangez', 'ils': 'mangent', 'elles': 'mangent'},
    'boire': {'je': 'bois', 'tu': 'bois', 'il': 'boit', 'elle': 'boit',
              'nous': 'buvons', 'vous': 'buvez', 'ils': 'boivent', 'elles': 'boivent'},
    'voir': {'je': 'vois', 'tu': 'vois', 'il': 'voit', 'elle': 'voit',
             'nous': 'voyons', 'vous': 'voyez', 'ils': 'voient', 'elles': 'voient'},
    'aller': {'je': 'vais', 'tu': 'vas', 'il': 'va', 'elle': 'va',
              'nous': 'allons', 'vous': 'allez', 'ils': 'vont', 'elles': 'vont'},
    'venir': {'je': 'viens', 'tu': 'viens', 'il': 'vient', 'elle': 'vient',
              'nous': 'venons', 'vous': 'venez', 'ils': 'viennent', 'elles': 'viennent'},
    'faire': {'je': 'fais', 'tu': 'fais', 'il': 'fait', 'elle': 'fait',
              'nous': 'faisons', 'vous': 'faites', 'ils': 'font', 'elles': 'font'},
    'avoir': {'je': 'ai', 'tu': 'as', 'il': 'a', 'elle': 'a',
              'nous': 'avons', 'vous': 'avez', 'ils': 'ont', 'elles': 'ont'},
    'prendre': {'je': 'prends', 'tu': 'prends', 'il': 'prend', 'elle': 'prend',
                'nous': 'prenons', 'vous': 'prenez', 'ils': 'prennent', 'elles': 'prennent'},
    'donner': {'je': 'donne', 'tu': 'donnes', 'il': 'donne', 'elle': 'donne',
               'nous': 'donnons', 'vous': 'donnez', 'ils': 'donnent', 'elles': 'donnent'},
    'vouloir': {'je': 'veux', 'tu': 'veux', 'il': 'veut', 'elle': 'veut',
                'nous': 'voulons', 'vous': 'voulez', 'ils': 'veulent', 'elles': 'veulent'}
}

# Objects - organized by verb compatibility
OBJECTS = {
    'manger': ['du riz', 'du pain', 'du poisson', 'de la viande', 'des fruits', 'des légumes'],
    'boire': ['de l\'eau', 'du lait', 'du thé', 'du café', 'du jus'],
    'voir': ['mon ami', 'ma mère', 'mon père', 'le professeur', 'l\'enfant', 'le chien'],
    'aller': ['à la maison', 'à l\'école', 'au marché', 'au village', 'au travail'],
    'venir': ['de la maison', 'de l\'école', 'du marché', 'du village', 'du travail'],
    'faire': ['le travail', 'les devoirs', 'la cuisine', 'le ménage', 'le lit'],
    'avoir': ['un livre', 'une maison', 'un ami', 'une sœur', 'un frère', 'de l\'argent'],
    'prendre': ['le livre', 'l\'eau', 'le pain', 'le bus', 'le stylo'],
    'donner': ['le livre', 'de l\'eau', 'du pain', 'de l\'argent', 'le cadeau'],
    'vouloir': ['du riz', 'de l\'eau', 'le livre', 'aller à la maison', 'manger']
}

def generate_module1(target_count=500):
    """Generate systematic base sentences"""
    
    sentences = []
    sentence_id = 1
    
    # Generate all combinations
    for verb, conjugations in VERBS.items():
        for pronoun in PRONOUNS:
            for obj in OBJECTS[verb]:
                
                conjugated_verb = conjugations[pronoun]
                
                # Special handling for je + avoir
                if pronoun == 'je' and verb == 'avoir':
                    sentence = f"j'ai {obj}"
                else:
                    sentence = f"{pronoun} {conjugated_verb} {obj}"
                
                sentences.append({
                    'sentence_id': f'M1_{sentence_id:04d}',
                    'module': 'module1_base',
                    'french': sentence,
                    'pronoun': pronoun,
                    'verb': verb,
                    'object': obj,
                    'structure': 'affirmative_present',
                    'adja_translation': '',
                    'notes': ''
                })
                
                sentence_id += 1
    
    # Convert to dataframe
    df = pd.DataFrame(sentences)
    
    # If we have more than target, sample stratified
    if len(df) > target_count:
        # Sample evenly across pronouns and verbs
        df = df.groupby(['pronoun', 'verb'], group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, target_count // 80)))
        ).head(target_count)
        df = df.reset_index(drop=True)
        # Reassign sentence IDs
        df['sentence_id'] = [f'M1_{i+1:04d}' for i in range(len(df))]
    
    return df

# Generate it
print("🔧 Generating Module 1 (base sentences)...")
df_module1 = generate_module1(500)

print(f"✅ Generated {len(df_module1)} base sentences")
print(f"\nDistribution:")
print(f"  Pronouns: {df_module1['pronoun'].value_counts().to_dict()}")
print(f"  Verbs: {df_module1['verb'].value_counts().to_dict()}")

print(f"\n📝 Sample sentences:")
for sent in df_module1.sample(10)['french'].tolist():
    print(f"  - {sent}")

# Save
df_module1.to_csv('module1_base.csv', index=False)
print(f"\n💾 Saved to module1_base.csv")