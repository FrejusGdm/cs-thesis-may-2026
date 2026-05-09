# generate_module1.py
import pandas as pd
import itertools

# Core vocabulary - KEEP IT SIMPLE AND HIGH-FREQUENCY
PRONOUNS = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles']

VERBS = {
# ========== NEW 10 VERBS ==========
    
    # 1. AIMER (to like/love) - Emotion/preference
    'aimer': {'je': 'aime', 'tu': 'aimes', 'il': 'aime', 'elle': 'aime',
              'nous': 'aimons', 'vous': 'aimez', 'ils': 'aiment', 'elles': 'aiment'},
    
    # 2. ACHETER (to buy) - Commerce
    'acheter': {'je': 'achète', 'tu': 'achètes', 'il': 'achète', 'elle': 'achète',
                'nous': 'achetons', 'vous': 'achetez', 'ils': 'achètent', 'elles': 'achètent'},
    
    # 3. CHERCHER (to look for/search) - Seeking
    'chercher': {'je': 'cherche', 'tu': 'cherches', 'il': 'cherche', 'elle': 'cherche',
                 'nous': 'cherchons', 'vous': 'cherchez', 'ils': 'cherchent', 'elles': 'cherchent'},
    
    # 4. TROUVER (to find) - Discovery
    'trouver': {'je': 'trouve', 'tu': 'trouves', 'il': 'trouve', 'elle': 'trouve',
                'nous': 'trouvons', 'vous': 'trouvez', 'ils': 'trouvent', 'elles': 'trouvent'},
    
    # 5. PARLER (to speak) - Communication
    'parler': {'je': 'parle', 'tu': 'parles', 'il': 'parle', 'elle': 'parle',
               'nous': 'parlons', 'vous': 'parlez', 'ils': 'parlent', 'elles': 'parlent'},
    
    # 6. SAVOIR (to know) - Knowledge (irregular but important!)
    'savoir': {'je': 'sais', 'tu': 'sais', 'il': 'sait', 'elle': 'sait',
               'nous': 'savons', 'vous': 'savez', 'ils': 'savent', 'elles': 'savent'},
    
    # 7. METTRE (to put/place) - Placement
    'mettre': {'je': 'mets', 'tu': 'mets', 'il': 'met', 'elle': 'met',
               'nous': 'mettons', 'vous': 'mettez', 'ils': 'mettent', 'elles': 'mettent'},
    
    # 8. LAISSER (to leave/let) - Abandonment/permission
    'laisser': {'je': 'laisse', 'tu': 'laisses', 'il': 'laisse', 'elle': 'laisse',
                'nous': 'laissons', 'vous': 'laissez', 'ils': 'laissent', 'elles': 'laissent'},
    
    # 9. APPORTER (to bring) - Motion + object
    'apporter': {'je': 'apporte', 'tu': 'apportes', 'il': 'apporte', 'elle': 'apporte',
                 'nous': 'apportons', 'vous': 'apportez', 'ils': 'apportent', 'elles': 'apportent'},
    
    # 10. MONTRER (to show) - Demonstration
    'montrer': {'je': 'montre', 'tu': 'montres', 'il': 'montre', 'elle': 'montre',
                'nous': 'montrons', 'vous': 'montrez', 'ils': 'montrent', 'elles': 'montrent'}
}

# Objects - organized by verb compatibility
OBJECTS = {
# 1. AIMER - things/people you like
    'aimer': ['le riz', 'le pain', 'le poisson', 'la viande', 'mon ami', 'ma famille'],
    
    # 2. ACHETER - things you buy at market
    'acheter': ['du riz', 'du pain', 'du poisson', 'de la viande', 'des fruits', 'un livre'],
    
    # 3. CHERCHER - things/people you look for
    'chercher': ['mon ami', 'le livre', 'le stylo', 'le travail', 'mon père', 'l\'enfant'],
    
    # 4. TROUVER - things/people you find
    'trouver': ['le livre', 'le stylo', 'mon ami', 'le travail', 'de l\'argent', 'la maison'],
    
    # 5. PARLER - languages/communication partners
    'parler': ['français', 'adja', 'anglais', 'avec mon ami', 'avec le professeur', 'avec ma mère'],
    
    # 6. SAVOIR - things you know/can do
    'savoir': ['la vérité', 'la réponse', 'le chemin', 'lire', 'écrire', 'parler français'],
    
    # 7. METTRE - placing objects in locations
    'mettre': ['le livre sur la table', 'le pain dans le sac', 'l\'eau dans le verre', 
               'le stylo dans le sac', 'les vêtements dans la maison', 'le cadeau sur la table'],
    
    # 8. LAISSER - things/people you leave
    'laisser': ['le livre', 'mon ami', 'l\'enfant à la maison', 'le sac', 
                'le travail', 'la maison'],
    
    # 9. APPORTER - things you bring
    'apporter': ['le livre', 'de l\'eau', 'du pain', 'le cadeau', 
                 'les fruits', 'le travail'],
    
    # 10. MONTRER - things/people you show
    'montrer': ['le livre', 'la maison', 'le chemin', 'mon ami', 
                'le travail', 'le cadeau']
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
                
                if pronoun == 'je' and conjugated_verb[0] in 'aeiouhéèêàô':
                    sentence = f"j'{conjugated_verb} {obj}"
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