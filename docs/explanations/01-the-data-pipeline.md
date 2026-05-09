# 1. The Data Pipeline

This file explains what "structured data" means in this project, how the 5 modules work, and why the structure matters.

---

## The Core Idea

Instead of collecting random translated sentences, we **designed** our training data using a systematic grammar-based approach. We start with simple present-tense sentences and then apply one grammatical transformation at a time (negation, past tense, future tense, questions) to create a **curriculum** the model can learn from.

Think of it like teaching a child: you don't throw 10,000 random sentences at them. You start with "I eat rice", then teach "I don't eat rice", then "I ate rice", then "I will eat rice", then "Do I eat rice?" Each step changes exactly one thing.

---

## Module 1: Base Sentences (Present Tense)

Module 1 generates simple Subject-Verb-Object sentences in the present tense. It does this by combining three ingredients:

**8 Pronouns:**
```
je (I), tu (you), il (he), elle (she),
nous (we), vous (you-plural), ils (they-masc), elles (they-fem)
```

**10 Verbs (Run-1):**
```
manger (eat), boire (drink), voir (see), aller (go), venir (come),
faire (do/make), avoir (have), prendre (take), donner (give), vouloir (want)
```

**~5-6 Objects per verb:**
```
manger: du riz, du pain, du poisson, de la viande, des fruits, des legumes
boire:  de l'eau, du lait, du the, du cafe, du jus
voir:   mon ami, ma mere, mon pere, le professeur, l'enfant
aller:  a la maison, a l'ecole, au marche, au village, au travail
```

Each pronoun gets conjugated with each verb, then paired with each object. For example:

```
je mange du riz          (I eat rice)
tu manges du pain        (you eat bread)
il boit du lait          (he drinks milk)
elle voit son ami        (she sees her friend)
nous allons a l'ecole   (we go to school)
vous venez du marche    (you come from the market)
ils font le travail      (they do the work)
elles ont un livre       (they have a book)
```

This is done in pure Python -- no AI involved. The script just combines the lists. This gives us about **400 base sentences per run** (8 pronouns x 10 verbs x ~5 objects each).

Each sentence gets a unique ID: `M1_0001`, `M1_0002`, `M1_0003`, etc.

---

## Module 2: Negation

Module 2 takes every Module 1 sentence and adds `ne...pas` around the verb. This is done with simple string manipulation in Python (no GPT-4 needed):

```
M1_0001: je mange du riz         ->  M2_0001: je ne mange pas du riz
M1_0002: tu manges du pain       ->  M2_0002: tu ne manges pas du pain
M1_0003: il boit du lait         ->  M2_0003: il ne boit pas du lait
M1_0007: j'ai un livre           ->  M2_0007: je n'ai pas un livre
M1_0015: nous allons a l'ecole   ->  M2_0015: nous n'allons pas a l'ecole
```

Notice the last two examples: when the verb starts with a vowel (`ai`, `allons`), French requires elision -- `je ne` stays as-is but `j'ai` becomes `je n'ai`. The script handles this.

**The key point**: each M2 sentence is linked back to exactly one M1 sentence via its ID. M2_0001 is always the negation of M1_0001. This link is called the **base_sentence_id**.

---

## Module 3: Past Tense (Passe Compose)

Module 3 takes every Module 1 sentence and transforms it to past tense. This requires GPT-4 because French past tense is complex (choice of auxiliary avoir/etre, irregular past participles):

```
M1_0001: je mange du riz         ->  M3_0001: j'ai mange du riz
M1_0003: il boit du lait         ->  M3_0003: il a bu du lait
M1_0009: elle va a l'ecole       ->  M3_0009: elle est allee a l'ecole
```

Notice:
- `mange` -> `mange` (regular)
- `boit` -> `a bu` (irregular past participle)
- `va` -> `est allee` (uses etre as auxiliary, not avoir, because `aller` is a movement verb)

GPT-4 processes these in batches of 15 sentences with strict instructions to only change the tense, keeping everything else identical.

---

## Module 4: Future Tense (aller + infinitive)

Module 4 takes every Module 1 sentence and transforms it to near-future using "aller + infinitive":

```
M1_0001: je mange du riz         ->  M4_0001: je vais manger du riz
M1_0003: il boit du lait         ->  M4_0003: il va boire du lait
M1_0015: nous allons a l'ecole   ->  M4_0015: nous allons aller a l'ecole
```

The pattern is straightforward: conjugate `aller` (to go) + infinitive of the original verb. But `aller` itself gets interesting: `nous allons` + `aller` = `nous allons aller` (we are going to go).

---

## Module 5: Questions

Module 5 takes every Module 1 sentence and creates two types of questions:

**Yes/No questions** (add "Est-ce que" at the beginning):
```
M1_0001: je mange du riz    ->  M5_0001: Est-ce que je mange du riz ?
M1_0003: il boit du lait    ->  M5_0003: Est-ce qu'il boit du lait ?
```

**Wh-questions** (replace the object with a question word):
```
M1_0001: je mange du riz    ->  M5_0010: Qu'est-ce que je mange ?
M1_0009: elle va a l'ecole  ->  M5_0019: Ou va-t-elle ?
M1_0005: il voit mon ami    ->  M5_0015: Qui voit-il ?
```

The wh-question word depends on what the object is:
- Food/thing -> `Qu'est-ce que` (what)
- Person -> `Qui` (who)
- Place -> `Ou` (where)

---

## The Complete Chain: One Base Sentence, Five Modules

Here's the full picture for a single base sentence:

```
MODULE 1 (base):     je mange du riz              (I eat rice)
    |
    +-- MODULE 2 (negation):  je ne mange pas du riz      (I don't eat rice)
    |
    +-- MODULE 3 (past):      j'ai mange du riz           (I ate rice)
    |
    +-- MODULE 4 (future):    je vais manger du riz       (I will eat rice)
    |
    +-- MODULE 5a (yes/no):   Est-ce que je mange du riz ? (Do I eat rice?)
    |
    +-- MODULE 5b (wh):       Qu'est-ce que je mange ?     (What do I eat?)
```

All six sentences share the same `base_sentence_id = M1_0001`. They form a **family** of related sentences.

---

## What Are Minimal Pairs?

A **minimal pair** is two sentences that differ in exactly one feature. In our data, each transformation creates a minimal pair with its base sentence:

```
Pair 1 (tense):    je mange du riz   vs   j'ai mange du riz
                   (present)              (past)
                   Only the tense changed. Everything else is identical.

Pair 2 (polarity): je mange du riz   vs   je ne mange pas du riz
                   (affirmative)          (negative)
                   Only the polarity changed.

Pair 3 (mood):     je mange du riz   vs   Est-ce que je mange du riz ?
                   (statement)            (question)
                   Only the sentence type changed.
```

This is powerful for learning because the model can compare the pairs and isolate what each grammatical change looks like in both French and Adja. If the model sees:

```
French: je mange du riz      ->  Adja: un du nu
French: je ne mange pas du riz ->  Adja: un me du nu o
```

It can learn that `ne...pas` in French corresponds to `me...o` in Adja, because everything else in the sentence is the same.

---

## Sentence IDs and the Linkage System

Every sentence has a unique ID with this format:

```
M{module}_{number}

Examples:
  M1_0001  = Module 1, sentence #1
  M2_0001  = Module 2, sentence #1 (negation of M1_0001)
  M3_0042  = Module 3, sentence #42 (past tense of M1_0042)
  M5_0100  = Module 5, sentence #100 (question from M1_0100)
```

Every Module 2-5 sentence has a `base_sentence_id` field that points back to its Module 1 source. This linkage is what we call **structure** -- it's not just a collection of sentences, it's a web of grammatical relationships.

---

## Run-1 vs Run-2

We created two separate datasets with **different verbs** to increase lexical diversity:

| | Run-1 (~1,982 sentences) | Run-2 (~2,284 sentences) |
|---|---|---|
| Verb 1 | manger (eat) | aimer (love/like) |
| Verb 2 | boire (drink) | acheter (buy) |
| Verb 3 | voir (see) | chercher (search) |
| Verb 4 | aller (go) | trouver (find) |
| Verb 5 | venir (come) | parler (speak) |
| Verb 6 | faire (do/make) | savoir (know) |
| Verb 7 | avoir (have) | mettre (put) |
| Verb 8 | prendre (take) | laisser (leave) |
| Verb 9 | donner (give) | apporter (bring) |
| Verb 10 | vouloir (want) | montrer (show) |

Combined, we have **20 different verbs** across both runs, giving us ~4,266 structured sentences total. Both runs use the same 8 pronouns and the same 5-module structure.

The verb diversity turns out to matter a lot: our experiments show that going from 5 verbs to 10 verbs triples the BLEU score (8.1 -> 22.9). More on this in [04-ablation-experiments.md](04-ablation-experiments.md).

---

## Summary

| Concept | What It Is |
|---------|-----------|
| **Structured data** | Sentences generated by combining grammar rules, not collected randomly |
| **Module** | One type of grammatical transformation (base, negation, past, future, questions) |
| **Base sentence** | A Module 1 present-tense sentence that all other modules transform |
| **Transformation** | Applying one grammatical change to a base sentence (e.g., adding negation) |
| **Minimal pair** | Two sentences that differ in exactly one feature |
| **base_sentence_id** | The link connecting a transformed sentence back to its base |
| **Structure/Linkage** | The web of relationships between base and transformed sentences |
| **Sentence ID** | Unique identifier in M{module}_{number} format |
