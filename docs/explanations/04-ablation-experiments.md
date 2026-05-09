# 4. Ablation Experiments

This file explains what ablations are, each ablation we ran, and what the results tell us about why our structured data works.

---

## What Is an Ablation?

An **ablation** is an experiment where you **remove one component** to see how much it contributes. The word comes from surgery -- "ablating" means removing tissue to study its function.

Think of it like baking a cake. You made a great cake with flour, sugar, eggs, butter, and vanilla. Now you want to know which ingredients are essential:

```
Full recipe (cake):           Delicious (score: 10/10)
Remove vanilla:               Still good (score: 9/10)  -> vanilla adds a little
Remove butter:                 Dry and bland (score: 5/10) -> butter is important
Remove eggs:                   Flat, crumbly (score: 3/10) -> eggs are critical
Remove flour:                  Not a cake anymore (score: 1/10) -> flour is essential
```

We did the same thing with our structured data. The "full recipe" is all 5 modules with all pronouns, all verbs, and intact minimal pairs. We removed components one at a time to measure their contribution.

---

## Ablation 1: Module Leave-One-Out

**Question:** Which grammatical modules contribute most to translation quality?

**Method:** Train with all modules (FULL), then remove one module at a time and retrain:

| Condition | Modules Included | What's Missing | BLEU | Drop |
|-----------|-----------------|----------------|------|------|
| FULL | 1+2+3+4+5 | Nothing | 22.9 | -- |
| NO-FUTURE | 1+2+3+5 | Module 4 (future tense) | 17.3 | **-5.6** |
| NO-QUESTIONS | 1+2+3+4 | Module 5 (questions) | 19.1 | -3.8 |
| NO-NEGATION | 1+3+4+5 | Module 2 (negation) | 19.4 | -3.5 |
| NO-PAST | 1+2+4+5 | Module 3 (past tense) | 20.1 | -2.8 |
| BASE-ONLY | 1 only | All transformations | 8.0 | **-14.9** |

### What this tells us

**Every module contributes.** There's no "useless" module -- removing any one of them hurts performance.

**Future tense contributes the most** (-5.6 BLEU). Why? The future tense transformation (`je mange` -> `je vais manger`) introduces a new grammatical structure: "aller + infinitive." This teaches the model about:
- Verb periphrasis (using two verbs together)
- Infinitive forms (which appear in many other contexts)
- The conjugation of `aller` (one of the most irregular French verbs)

**BASE-ONLY is devastating** (BLEU 8.0). Having only present-tense SVO sentences -- no negation, no tense changes, no questions -- gives the model almost nothing to work with. The transformations are what make structured data powerful.

This makes intuitive sense: if you only taught someone "I eat rice" and "She drinks water" in 400 variations, they'd struggle to translate "I didn't eat rice" or "Will she drink water?" The transformations expose the model to grammatical patterns it needs.

---

## Ablation 1b: Size-Controlled Module Ablation

**Problem with Ablation 1:** When we remove a module, we also reduce the total training data. NO-NEGATION has fewer sentences than FULL. Maybe the drop is just because of less data, not because negation is important?

**Solution:** Fix total training size at 1,000 sentences for all conditions. Distribute sentences equally across included modules:

```
FULL-1K:      200 per module x 5 modules = 1,000 sentences
NO-NEG-1K:    250 per module x 4 modules = 1,000 sentences
NO-PAST-1K:   250 per module x 4 modules = 1,000 sentences
BASE-1K:      1,000 from module 1 only   = 1,000 sentences (if available)
```

| Condition | Size | BLEU |
|-----------|------|------|
| FULL-1K | 1,000 | 21.1 |
| NO-PAST-1K | 1,000 | 20.1 |
| NO-QUEST-1K | 1,000 | 18.9 |
| NO-NEG-1K | 1,000 | 17.9 |
| NO-FUT-1K | 1,000 | 17.4 |
| BASE-1K | ~400 (capped) | 7.9 |

The pattern holds even when size is controlled. The modules genuinely contribute grammatical knowledge, not just data volume.

---

## Ablation 2: Verb Diversity

**Question:** How many different verbs does the model need to learn well?

**Method:** Train with all 10 verbs, then restrict to random subsets of 5, 3, or 1 verb. For 5-verb and 3-verb conditions, we ran 3 random subsets (a, b, c) to account for different verb combinations.

| Condition | # Verbs | BLEU | Notes |
|-----------|---------|------|-------|
| 1-VERB | 1 | 2.8 | Only "manger" (eat) |
| 3-VERBS-a | 3 | 3.9 | Random subset a |
| 3-VERBS-b | 3 | 5.9 | Random subset b |
| 3-VERBS-c | 3 | 5.7 | Random subset c |
| 5-VERBS-a | 5 | 8.7 | Random subset a |
| 5-VERBS-b | 5 | 8.0 | Random subset b |
| 5-VERBS-c | 5 | 7.7 | Random subset c |
| **10-VERBS** | **10** | **22.9** | **All verbs** |

### What this tells us

This is the most dramatic ablation result. The jump from 5 verbs (BLEU ~8) to 10 verbs (BLEU 22.9) is almost a **3x improvement**.

**Why does verb count matter so much?**

Each French verb conjugates differently. By seeing 10 verbs, the model learns:
- Regular patterns: `-er` verbs (manger, donner), `-ir` verbs (venir), `-re` verbs (prendre)
- Irregular forms: avoir (j'ai, il a, nous avons), aller (je vais, nous allons), faire (je fais, ils font)
- Auxiliary patterns: "j'ai mange" (avoir) vs "elle est allee" (etre)

With only 1 verb (manger), the model memorizes how to translate "mange/manges/mangeons" but has no idea what to do with "boit" or "va" or "fait." It's like learning to conjugate only one verb in Spanish class -- you'd fail any test with other verbs.

**The a/b/c variation is interesting too.** Look at 3-VERBS: subset b (BLEU 5.9) beats subset a (BLEU 3.9). This is because different verbs carry different amounts of grammatical information. A subset with "avoir" (highly irregular, most common auxiliary) teaches more than a subset with three regular `-er` verbs.

---

## Ablation 3: Pronoun Coverage

**Question:** Does the model need all 8 pronouns, or can it learn from fewer?

**Method:** Restrict training data to different subsets of pronouns:

| Condition | Pronouns | Count | BLEU |
|-----------|----------|-------|------|
| ALL-8 | je, tu, il, elle, nous, vous, ils, elles | 8 | 22.9 |
| REDUCED-4 | je, tu, il, nous | 4 | 19.5 |
| SINGULAR-3 | je, tu, il | 3 | 18.6 |
| MINIMAL-1 | je | 1 | 15.9 |

### What this tells us

Pronoun diversity helps but is less critical than verb diversity. Going from 1 to 8 pronouns improves BLEU by 7 points (15.9 -> 22.9), while going from 1 to 10 verbs improves by 20 points (2.8 -> 22.9).

Even with just "je" (first person singular), the model can reach BLEU 15.9 -- it learns something useful from seeing one pronoun across all verbs and modules. But it struggles with pronouns it has never seen, especially plural forms (nous, vous, ils, elles) which conjugate differently.

The REDUCED-4 result (BLEU 19.5) is close to ALL-8 (BLEU 22.9), suggesting diminishing returns -- the jump from 4 to 8 pronouns is smaller than from 1 to 4. The first few pronouns teach the most, and additional ones provide incremental improvement.

Note: REDUCED-4 has high variance (19.5 +/- 4.2) -- one seed got 13.5 while another got 22.6. This suggests that with limited pronoun coverage, the model's learning is more sensitive to the random training order.

---

## Ablation 4: Minimal Pairs (Structure Linkage)

**Question:** Does the structural relationship between base and transformed sentences matter, or is it just the content?

This is the most important ablation because it directly tests our central hypothesis: that the **structure** of the data matters, not just having the right sentences.

### What PAIRS-INTACT means

The data as designed: each Module 2-5 sentence is correctly linked to its Module 1 base. The model sees systematic transformations:

```
"je mange du riz"           -> [Adja translation A]
"je ne mange pas du riz"    -> [Adja translation B]
"j'ai mange du riz"         -> [Adja translation C]
```

The model can compare A, B, and C to learn what negation and past tense look like in Adja.

### What PAIRS-BROKEN means

We take the exact same sentences but **shuffle the pairings within each module**. The French and Adja sides are shuffled independently, so:

```
BEFORE (intact):
  French: "je ne mange pas du riz"     -> Adja: [correct negation of "I eat rice"]
  French: "tu ne bois pas du lait"     -> Adja: [correct negation of "you drink milk"]
  French: "il ne voit pas mon ami"     -> Adja: [correct negation of "he sees my friend"]

AFTER (broken - French shuffled, Adja shuffled independently):
  French: "il ne voit pas mon ami"     -> Adja: [correct negation of "I eat rice"]      WRONG!
  French: "je ne mange pas du riz"     -> Adja: [correct negation of "he sees my friend"] WRONG!
  French: "tu ne bois pas du lait"     -> Adja: [correct negation of "you drink milk"]   LUCKY MATCH!
```

The data still contains the same French sentences and the same Adja sentences, but the **pairings** are wrong. The French-Adja correspondences are broken.

Critically, Module 1 (base sentences) remains intact -- only the transformed modules (2-5) are shuffled. And the shuffling is independent: the French sentences are shuffled in one order, and the Adja translations in a different order. So most pairs are mismatched.

### The result

| Condition | BLEU | chrF |
|-----------|------|------|
| PAIRS-INTACT | 22.9 | 30.1 |
| PAIRS-BROKEN | **5.4** | **14.0** |

**Destroying the structural linkage drops BLEU from 22.9 to 5.4** -- a 17.5 point collapse. The model goes from "decent translations" to "barely functional."

### Why this is the killer finding

This proves that it's not just about having the right sentences in your training data. PAIRS-BROKEN contains the exact same French sentences and the exact same Adja sentences as PAIRS-INTACT. The only difference is whether the French-Adja pairings are correct.

When the pairings are correct, the model can learn systematic patterns:
- "ne...pas" in French always corresponds to the right negation marker in Adja
- Past tense auxiliaries in French correspond to the right tense markers in Adja

When the pairings are broken, the model gets contradictory signals:
- "ne mange pas" sometimes maps to a negation, sometimes to a question, sometimes to a future tense
- The model can't learn any consistent rules

**This is the strongest evidence that our structured approach works because of its structure, not just its content.** The minimal-pair linkage is what enables the model to isolate and learn grammatical correspondences between French and Adja.

---

## Summary Table

| Ablation | What We Tested | Key Finding | Strongest Evidence |
|----------|---------------|-------------|-------------------|
| Module LOO | Individual module contributions | Future tense most valuable (-5.6) | Every module helps |
| Size-controlled | Same as above, controlling for size | Pattern holds at 1K each | Not just data volume |
| Verb diversity | How many verbs needed | 5->10 verbs triples BLEU | Lexical diversity critical |
| Pronoun coverage | How many pronouns needed | Diminishing returns after 4 | Less critical than verbs |
| **Minimal pairs** | **Does structure matter?** | **BLEU drops 22.9 -> 5.4 when broken** | **Structure is essential** |
