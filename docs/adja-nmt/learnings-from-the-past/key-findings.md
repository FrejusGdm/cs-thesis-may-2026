# Key Research Findings

From the neurosymbolic/ACL paper experiments (2025-2026).

## The Core Result

- ~4K systematically-structured sentences + 10K random Tatoeba sentences -> BLEU ~20
- 10K random sentences alone -> BLEU 2-3
- The structured subset drives the improvement, not the extra data volume

## What "Structured" Means

Five-module curriculum design, each building on Module 1 base sentences:
1. Present tense SVO (combinatorial: 8 pronouns x 10 verbs x ~5 objects)
2. Negation (ne...pas) — rule-based transformation
3. Past tense (passe compose) — GPT-4 transformation
4. Future tense (aller + inf) — GPT-4 transformation
5. Questions (yes/no + wh-) — GPT-4 transformation

Every Module 2-5 sentence links to exactly one Module 1 sentence via `base_sentence_id` (minimal pairs).

## Models Tested

- NLLB-200 (600M and 1.3B)
- mBART-50 (French-initialized and random-initialized)
- Gemini (API, seed 42 only due to cost)

## Metrics

- BLEU, chrF, chrF++ as primary
- COMET and BERTScore discussed in limitations (require reference-heavy setup)
- ROUGE-L and Perplexity added later per professor feedback
