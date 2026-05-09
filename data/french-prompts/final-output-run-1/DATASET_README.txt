
ADJA-FRENCH PARALLEL DATASET
============================
Generated: 2025-10-30

OVERVIEW
--------
Total sentences: 1982
Language pair: French → Adja
Purpose: Neural Machine Translation training

STRUCTURE
---------
Module 1 - Present Tense (424 sentences)
  Base affirmative sentences in present tense
  Example: "Je mange du riz"

Module 2 - Negation (424 sentences)
  Negated versions of Module 1 sentences
  Example: "Je ne mange pas du riz"
  Links to: Module 1 (minimal pairs)

Module 3 - Past Tense (424 sentences)
  Past tense versions of Module 1 sentences
  Example: "J'ai mangé du riz"
  Links to: Module 1 (minimal pairs)

Module 4 - Future Tense (424 sentences)
  Future tense versions of Module 1 sentences
  Example: "Je vais manger du riz"
  Links to: Module 1 (minimal pairs)

Module 5 - Questions (286 sentences, deduplicated)
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
