"""
reconstruct_metadata.py — Reconstruct linguistic metadata from simple-dataset.csv.

The raw CSV has columns (ID, French, Translation, original_id, dataset_source) but
lacks the structured metadata (module, pronoun, verb, sentence_id, base_sentence_id)
needed by the experiment pipeline's splitting and ablation scripts.

Since all structured sentences were generated deterministically by generation-1.py
(Run-1 and Run-2), we can recover metadata by parsing French sentence patterns:

  M5 (questions):  ends with '?' or starts with 'Est-ce que', 'Qu'est-ce que', etc.
  M2 (negation):   contains 'ne ... pas' or "n'...pas"
  M4 (future):     conjugated aller + infinitive (e.g. 'va manger', 'vais boire')
  M3 (past):       auxiliary avoir/être + past participle (e.g. 'ai mangé', 'a bu')
  M1 (present):    everything else (default)

Usage:
    python reconstruct_metadata.py \
        --input experiments/data/raw/simple-dataset.csv \
        --output experiments/data/raw/simple-dataset-enriched.csv
"""

import argparse
import csv
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Verb conjugation tables from generation-1.py (both code runs)
# ---------------------------------------------------------------------------

# Run 1 verbs (phrases_simples)
RUN1_VERBS = {
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
                'nous': 'voulons', 'vous': 'voulez', 'ils': 'veulent', 'elles': 'veulent'},
}

# Run 2 verbs (partie2)
RUN2_VERBS = {
    'aimer': {'je': 'aime', 'tu': 'aimes', 'il': 'aime', 'elle': 'aime',
              'nous': 'aimons', 'vous': 'aimez', 'ils': 'aiment', 'elles': 'aiment'},
    'acheter': {'je': 'achète', 'tu': 'achètes', 'il': 'achète', 'elle': 'achète',
                'nous': 'achetons', 'vous': 'achetez', 'ils': 'achètent', 'elles': 'achètent'},
    'chercher': {'je': 'cherche', 'tu': 'cherches', 'il': 'cherche', 'elle': 'cherche',
                 'nous': 'cherchons', 'vous': 'cherchez', 'ils': 'cherchent', 'elles': 'cherchent'},
    'trouver': {'je': 'trouve', 'tu': 'trouves', 'il': 'trouve', 'elle': 'trouve',
                'nous': 'trouvons', 'vous': 'trouvez', 'ils': 'trouvent', 'elles': 'trouvent'},
    'parler': {'je': 'parle', 'tu': 'parles', 'il': 'parle', 'elle': 'parle',
               'nous': 'parlons', 'vous': 'parlez', 'ils': 'parlent', 'elles': 'parlent'},
    'savoir': {'je': 'sais', 'tu': 'sais', 'il': 'sait', 'elle': 'sait',
               'nous': 'savons', 'vous': 'savez', 'ils': 'savent', 'elles': 'savent'},
    'mettre': {'je': 'mets', 'tu': 'mets', 'il': 'met', 'elle': 'met',
               'nous': 'mettons', 'vous': 'mettez', 'ils': 'mettent', 'elles': 'mettent'},
    'laisser': {'je': 'laisse', 'tu': 'laisses', 'il': 'laisse', 'elle': 'laisse',
                'nous': 'laissons', 'vous': 'laissez', 'ils': 'laissent', 'elles': 'laissent'},
    'apporter': {'je': 'apporte', 'tu': 'apportes', 'il': 'apporte', 'elle': 'apporte',
                 'nous': 'apportons', 'vous': 'apportez', 'ils': 'apportent', 'elles': 'apportent'},
    'montrer': {'je': 'montre', 'tu': 'montres', 'il': 'montre', 'elle': 'montre',
                'nous': 'montrons', 'vous': 'montrez', 'ils': 'montrent', 'elles': 'montrent'},
}

ALL_VERBS = {**RUN1_VERBS, **RUN2_VERBS}

# All 20 infinitive forms
ALL_INFINITIVES = set(ALL_VERBS.keys())

# Past participles for the 20 verbs
PAST_PARTICIPLES = {
    'manger': 'mangé', 'boire': 'bu', 'voir': 'vu', 'aller': 'allé',
    'venir': 'venu', 'faire': 'fait', 'avoir': 'eu', 'prendre': 'pris',
    'donner': 'donné', 'vouloir': 'voulu',
    'aimer': 'aimé', 'acheter': 'acheté', 'chercher': 'cherché',
    'trouver': 'trouvé', 'parler': 'parlé', 'savoir': 'su',
    'mettre': 'mis', 'laisser': 'laissé', 'apporter': 'apporté',
    'montrer': 'montré',
}

# Reverse lookup: past participle -> infinitive
PP_TO_INF = {pp: inf for inf, pp in PAST_PARTICIPLES.items()}

# Conjugated "aller" forms (used for future detection)
ALLER_FORMS = {
    'vais': 'je', 'vas': 'tu', 'va': 'il/elle',
    'allons': 'nous', 'allez': 'vous', 'vont': 'ils/elles',
}

# Auxiliary "avoir" forms (used for passé composé detection)
AVOIR_AUX = {'ai', 'as', 'a', 'avons', 'avez', 'ont'}

# Auxiliary "être" forms (for verbs using être in passé composé: aller, venir)
ETRE_AUX = {'suis', 'es', 'est', 'sommes', 'êtes', 'sont'}

PRONOUNS = ['je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles']

# Build reverse lookup: conjugated form -> list of (infinitive, pronoun) pairs
CONJUGATION_LOOKUP = defaultdict(list)
for inf, conj_table in ALL_VERBS.items():
    for pronoun, form in conj_table.items():
        CONJUGATION_LOOKUP[form].append((inf, pronoun))


# ---------------------------------------------------------------------------
# Pronoun extraction
# ---------------------------------------------------------------------------

def clean_sentence(text):
    """Strip leading line numbers, tabs, and whitespace artifacts from a sentence."""
    text = text.strip()
    # Remove leading "1541. " or "1720.\t" patterns (line numbers baked into text)
    text = re.sub(r'^\d+\.\s*', '', text)
    # Fix missing space: "Est-ce queil" -> "Est-ce que il"
    text = re.sub(r"(Est-ce qu[e'])\s*([a-zéèêà])", r"\1 \2", text, flags=re.IGNORECASE)
    text = re.sub(r"(Qu'est-ce qu[e'])\s*([a-zéèêà])", r"\1 \2", text, flags=re.IGNORECASE)
    # Fix missing spaces between pronoun and verb in question forms
    # Data has issues like "Est-ce que ellemange" -> "Est-ce que elle mange"
    # Only fix after "que " to avoid breaking "ils"/"elles" -> "il s"/"elle s"
    text = re.sub(r'(que\s+)(il|elle|ils|elles)([a-zéèêà])',
                  r'\1\2 \3', text, flags=re.IGNORECASE)
    return text.strip()


def extract_pronoun(text):
    """Extract the subject pronoun from the beginning of a French sentence."""
    text = clean_sentence(text).lower()

    # Handle elision: j'ai, j'achète, etc.
    if text.startswith("j'") or text.startswith("j\u2019"):
        return "je"

    # Handle question forms: "Est-ce que je ...", "Qu'est-ce que je ..."
    # Strip question prefix to find the pronoun
    q_prefixes = [
        "est-ce que ", "est-ce qu'", "est-ce qu\u2019",
        "qu'est-ce que ", "qu'est-ce qu'", "qu\u2019est-ce que ", "qu\u2019est-ce qu\u2019",
    ]
    lowered = text.lower()
    for prefix in q_prefixes:
        if lowered.startswith(prefix):
            rest = text[len(prefix):].strip()
            return extract_pronoun(rest)

    # Handle "D'où", "Quel", "Comment" questions
    for q_start in ["d'où ", "d\u2019où ", "quel ", "quelle ", "comment "]:
        if lowered.startswith(q_start):
            rest = text[len(q_start):].strip()
            # Try "est-ce que" after
            for prefix in q_prefixes:
                if rest.lower().startswith(prefix):
                    return extract_pronoun(rest[len(prefix):].strip())
            # Inverted: "viennent-elles", "prend-il"
            for w in rest.split():
                if '-' in w:
                    parts = w.lower().rstrip('?.,!').split('-')
                    # Handle "t-il", "t-elle"
                    for p in parts:
                        if p in PRONOUNS:
                            return p
            return None

    # Handle "Où" questions: "Où est-ce que je...", "Où veux-tu..."
    if lowered.startswith("où "):
        rest = text[3:].strip()
        # "Où est-ce que <pronoun>..."
        for prefix in q_prefixes:
            if rest.lower().startswith(prefix):
                return extract_pronoun(rest[len(prefix):].strip())
        # "Où vas-tu", "Où veut-il" — inverted: verb-pronoun
        inv_match = re.match(r'\w+-(\w+)', rest)
        if inv_match:
            candidate = inv_match.group(1).lower()
            # Handle "t-il", "t-elle" (euphonic t)
            if candidate == 't':
                inv_match2 = re.match(r'\w+-t-(\w+)', rest)
                if inv_match2:
                    candidate = inv_match2.group(1).lower()
            if candidate in PRONOUNS:
                return candidate
        return None

    # Handle inverted questions: "Que fait-il?", "Que faisons-nous?"
    # Also: "Qui est-ce que je laisse?"
    if lowered.startswith("que ") or lowered.startswith("qui "):
        rest = text[4:].strip()
        # Check for "est-ce que" after "Qui/Que"
        for prefix in q_prefixes:
            if rest.lower().startswith(prefix):
                return extract_pronoun(rest[len(prefix):].strip())
        # Inverted: "fait-il", "faisons-nous"
        inv_match = re.match(r'\w+-(\w+)', rest)
        if inv_match:
            candidate = inv_match.group(1).lower()
            if candidate == 't':
                inv_match2 = re.match(r'\w+-t-(\w+)', rest)
                if inv_match2:
                    candidate = inv_match2.group(1).lower()
            if candidate in PRONOUNS:
                return candidate
        return None

    # Standard: pronoun is the first word
    # Check two-word start first for "ils"/"elles" vs "il"/"elle"
    words = text.split()
    if not words:
        return None
    first_word = words[0].lower()
    # Check exact match, preferring longer forms
    for p in sorted(PRONOUNS, key=len, reverse=True):
        if first_word == p:
            return p

    return None


# ---------------------------------------------------------------------------
# Module detection
# ---------------------------------------------------------------------------

def detect_module(text):
    """Classify a French sentence into its grammatical module.

    Returns one of: 'M1' (present), 'M2' (negation), 'M3' (past),
    'M4' (future), 'M5' (question), or None if unclassifiable.
    """
    text = clean_sentence(text)
    lowered = text.lower()
    words = lowered.split()

    # M5: Questions — must check first since questions can also contain
    # negation/tense markers
    if '?' in text:
        return 'M5'
    if lowered.startswith('est-ce que') or lowered.startswith('est-ce qu'):
        return 'M5'
    if lowered.startswith("qu'est-ce que") or lowered.startswith("qu'est-ce qu"):
        return 'M5'
    if lowered.startswith("qu'est-ce que") or lowered.startswith("qu'est-ce qu"):
        return 'M5'
    if any(lowered.startswith(q) for q in ['où ', 'qui ', 'que ', "d'où ", "d\u2019où ",
                                            'quel ', 'quelle ', 'comment ']):
        return 'M5'

    # M2: Negation — contains "ne...pas" or "n'...pas"
    if ' ne ' in lowered and ' pas' in lowered:
        return 'M2'
    if " n'" in lowered and ' pas' in lowered:
        return 'M2'
    if lowered.startswith("n'") and ' pas' in lowered:
        return 'M2'
    # Handle "je ne" at start
    if re.search(r'\bne\b.*\bpas\b', lowered):
        return 'M2'

    # M4: Future — conjugated aller + infinitive
    # Pattern: pronoun + aller_form + infinitive_verb
    for i, word in enumerate(words):
        if word in ALLER_FORMS and i + 1 < len(words):
            next_word = words[i + 1]
            if next_word in ALL_INFINITIVES:
                return 'M4'

    # M3: Past (passé composé) — auxiliary avoir/être + past participle
    for i, word in enumerate(words):
        # Handle elided auxiliary: "j'ai" -> check "ai"
        w = word
        if w.startswith("j'") or w.startswith("j\u2019"):
            w = w[2:]
        if w in AVOIR_AUX and i + 1 < len(words):
            next_word = re.sub(r'\([^)]*\)', '', words[i + 1])
            if next_word in PP_TO_INF:
                return 'M3'
        if w in ETRE_AUX and i + 1 < len(words):
            next_word = re.sub(r'\([^)]*\)', '', words[i + 1])
            # être verbs: allé(e)(s), venu(e)(s)
            base = re.sub(r'(e|s|es)$', '', next_word)
            if base in PP_TO_INF or next_word in PP_TO_INF:
                return 'M3'

    # M1: Present tense (default)
    return 'M1'


# ---------------------------------------------------------------------------
# Verb detection
# ---------------------------------------------------------------------------

def detect_verb(text, module, source):
    """Identify the main verb (infinitive) from a French sentence.

    Uses the conjugation tables and the sentence's module to pick the right
    verb form to search for.
    """
    text = clean_sentence(text)
    lowered = text.lower()
    words = lowered.split()

    # Select verb pool based on source
    if source == 'phrases_simples':
        verb_pool = RUN1_VERBS
    elif source == 'partie2':
        verb_pool = RUN2_VERBS
    else:
        verb_pool = ALL_VERBS

    if module == 'M4':
        # Future: aller + infinitive — the infinitive IS the main verb
        for i, word in enumerate(words):
            if word in ALLER_FORMS and i + 1 < len(words):
                candidate = words[i + 1]
                if candidate in verb_pool:
                    return candidate
        return None

    if module == 'M3':
        # Past: aux + past participle — look up participle
        for i, word in enumerate(words):
            w = word
            if w.startswith("j'") or w.startswith("j\u2019"):
                w = w[2:]
            if w in AVOIR_AUX | ETRE_AUX:
                if i + 1 < len(words):
                    pp = words[i + 1]
                    # Strip parenthetical agreement: "allé(e)" -> "allé"
                    pp = re.sub(r'\([^)]*\)', '', pp)
                    # Try exact match
                    if pp in PP_TO_INF and PP_TO_INF[pp] in verb_pool:
                        return PP_TO_INF[pp]
                    # Try stripping agreement markers (allée, venus, etc.)
                    base = re.sub(r'(e|s|es)$', '', pp)
                    if base in PP_TO_INF and PP_TO_INF[base] in verb_pool:
                        return PP_TO_INF[base]
        return None

    if module == 'M5':
        # Questions — various forms. Try to find the verb after stripping
        # question prefixes.
        # "Est-ce que je mange du riz?" -> find "mange"
        # "Que fait-il?" -> find "fait"
        # "Où veux-tu aller?" -> find "veux" (vouloir)
        # "D'où viennent-elles?" -> find "viennent" (venir)
        # First: try finding past participles (for questions about past)
        for word in words:
            clean = re.sub(r'\([^)]*\)', '', word.rstrip('?.,!'))
            if clean in PP_TO_INF and PP_TO_INF[clean] in verb_pool:
                return PP_TO_INF[clean]
        # Then: try conjugated forms
        for word in words:
            clean = word.rstrip('?.,!')
            # Strip hyphenated pronoun: fait-il -> fait, viennent-elles -> viennent
            if '-' in clean:
                clean = clean.split('-')[0]
            # Handle elision: "j'ai" -> "ai", "n'ai" -> "ai"
            if clean.startswith("j'") or clean.startswith("j\u2019"):
                clean = clean[2:]
            if clean.startswith("n'") or clean.startswith("n\u2019"):
                clean = clean[2:]
            if clean in CONJUGATION_LOOKUP:
                for inf, _ in CONJUGATION_LOOKUP[clean]:
                    if inf in verb_pool:
                        return inf
        return None

    # M1 (present) and M2 (negation): look for conjugated verb forms
    # For M2, skip "ne" and "pas", look for the conjugated form between them
    for word in words:
        clean = word.rstrip('?.,!')
        if clean in ('ne', 'pas', "n'") or clean in PRONOUNS:
            continue
        # Handle elision: "j'ai" -> "ai", "j'achète" -> "achète"
        if clean.startswith("j'") or clean.startswith("j\u2019"):
            clean = clean[2:]
        # Handle negation elision: "n'ai" -> "ai", "n'achète" -> "achète"
        if clean.startswith("n'") or clean.startswith("n\u2019"):
            clean = clean[2:]
        if clean in CONJUGATION_LOOKUP:
            for inf, _ in CONJUGATION_LOOKUP[clean]:
                if inf in verb_pool:
                    return inf

    return None


# ---------------------------------------------------------------------------
# Base sentence matching
# ---------------------------------------------------------------------------

def build_base_sentence_index(rows):
    """Build an index of M1 sentences keyed by (pronoun, verb, object_fragment).

    For M2-M5 sentences, we strip the grammatical transformation to recover
    the underlying M1 sentence and look it up.
    """
    index = {}  # (pronoun, verb, object_approx) -> sentence_id
    for row in rows:
        if row['module'] == 'M1':
            key = (row['pronoun'], row['verb'])
            if key not in index:
                index[key] = []
            index[key].append(row['sentence_id'])
    return index


def find_base_sentence(row, m1_by_pronoun_verb):
    """Find the M1 base_sentence_id for an M2-M5 sentence.

    Since we can't perfectly reconstruct the object from transformed sentences,
    we match on (pronoun, verb) and assign the first available M1 sentence_id
    from that group. This is approximate but sufficient for group-aware splitting.
    """
    if row['module'] == 'M1':
        return row['sentence_id']
    key = (row['pronoun'], row['verb'])
    candidates = m1_by_pronoun_verb.get(key, [])
    if candidates:
        return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct linguistic metadata from simple-dataset.csv"
    )
    parser.add_argument(
        "--input",
        default="experiments/data/raw/simple-dataset.csv",
        help="Path to the raw CSV with (ID, French, Translation, original_id, dataset_source)",
    )
    parser.add_argument(
        "--output",
        default="experiments/data/raw/simple-dataset-enriched.csv",
        help="Path for the enriched output CSV",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    rows = []
    with open(args.input, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} rows from {args.input}")
    print(f"Columns: {list(rows[0].keys()) if rows else '(empty)'}")

    # ------------------------------------------------------------------
    # Detect metadata for each row
    # ------------------------------------------------------------------
    module_counts = defaultdict(int)
    pronoun_counts = defaultdict(int)
    verb_counts = defaultdict(int)
    unmatched_module = []
    unmatched_pronoun = []
    unmatched_verb = []

    for i, row in enumerate(rows):
        french = row.get('French', '').strip()
        source = row.get('dataset_source', '').strip()

        # Detect module
        module = detect_module(french)
        row['module'] = module
        module_counts[module] += 1

        # Detect pronoun
        pronoun = extract_pronoun(french)
        row['pronoun'] = pronoun
        if pronoun:
            pronoun_counts[pronoun] += 1
        else:
            unmatched_pronoun.append((i, french))

        # Detect verb
        verb = detect_verb(french, module, source)
        row['verb'] = verb
        if verb:
            verb_counts[verb] += 1
        else:
            unmatched_verb.append((i, french))

        # Track run
        if source == 'phrases_simples':
            row['run'] = 'run1'
        elif source == 'partie2':
            row['run'] = 'run2'
        else:
            row['run'] = 'unknown'

    # ------------------------------------------------------------------
    # Assign sentence IDs
    # ------------------------------------------------------------------
    module_seq = defaultdict(int)
    for row in rows:
        mod = row['module'] or 'UNK'
        mod_num = {'M1': 1, 'M2': 2, 'M3': 3, 'M4': 4, 'M5': 5}.get(mod, 0)
        module_seq[mod_num] += 1
        row['sentence_id'] = f"M{mod_num}_{module_seq[mod_num]:04d}"

    # ------------------------------------------------------------------
    # Build base sentence index and assign base_sentence_id
    # ------------------------------------------------------------------
    m1_index = build_base_sentence_index(rows)
    unmatched_base = 0
    for row in rows:
        base_id = find_base_sentence(row, m1_index)
        row['base_sentence_id'] = base_id if base_id else ''
        if row['module'] != 'M1' and not base_id:
            unmatched_base += 1

    # ------------------------------------------------------------------
    # Rename columns to match pipeline expectations
    # ------------------------------------------------------------------
    for row in rows:
        row['french'] = row.pop('French', '')
        row['adja_translation'] = row.pop('Translation', '')

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    fieldnames = [
        'sentence_id', 'base_sentence_id', 'module', 'french', 'adja_translation',
        'pronoun', 'verb', 'run', 'dataset_source', 'original_id', 'ID',
    ]
    # Keep any extra columns
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    for key in sorted(all_keys):
        if key not in fieldnames:
            fieldnames.append(key)

    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nWrote {len(rows)} rows to {args.output}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("MODULE DISTRIBUTION")
    print(f"{'='*60}")
    for mod in ['M1', 'M2', 'M3', 'M4', 'M5', None]:
        label = mod if mod else 'UNCLASSIFIED'
        count = module_counts.get(mod, 0)
        pct = count / len(rows) * 100
        print(f"  {label:15s}: {count:5d}  ({pct:5.1f}%)")

    print(f"\n{'='*60}")
    print("PRONOUN DISTRIBUTION")
    print(f"{'='*60}")
    for p in PRONOUNS:
        count = pronoun_counts.get(p, 0)
        print(f"  {p:10s}: {count:5d}")
    print(f"  {'UNMATCHED':10s}: {len(unmatched_pronoun):5d}")

    print(f"\n{'='*60}")
    print("VERB DISTRIBUTION")
    print(f"{'='*60}")
    for v in sorted(verb_counts, key=verb_counts.get, reverse=True):
        print(f"  {v:12s}: {verb_counts[v]:5d}")
    print(f"  {'UNMATCHED':12s}: {len(unmatched_verb):5d}")

    print(f"\n{'='*60}")
    print("BASE SENTENCE MATCHING")
    print(f"{'='*60}")
    m1_count = module_counts.get('M1', 0)
    matched_base = sum(1 for r in rows if r['module'] != 'M1' and r['base_sentence_id'])
    total_non_m1 = len(rows) - m1_count
    print(f"  M1 base sentences: {m1_count}")
    print(f"  M2-M5 with base_sentence_id: {matched_base}/{total_non_m1}")
    print(f"  M2-M5 without base match: {unmatched_base}")

    # Show some unmatched examples
    if unmatched_pronoun:
        print(f"\n--- Sample unmatched pronouns (first 5) ---")
        for idx, sent in unmatched_pronoun[:5]:
            print(f"  [{idx}] {sent}")
    if unmatched_verb:
        print(f"\n--- Sample unmatched verbs (first 5) ---")
        for idx, sent in unmatched_verb[:5]:
            print(f"  [{idx}] {sent}")

    # Coverage stats
    total = len(rows)
    full_match = sum(1 for r in rows if r['module'] and r['pronoun'] and r['verb'])
    print(f"\n{'='*60}")
    print(f"COVERAGE: {full_match}/{total} rows ({full_match/total*100:.1f}%) "
          f"have module + pronoun + verb")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
