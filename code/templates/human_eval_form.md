# Human Evaluation Scoring Form

## Instructions for Evaluators

You will evaluate machine translations from French to Adja. For each sentence, you will see the original French text and one or more Adja translations produced by different systems. You do NOT know which system produced which translation.

Score each translation on two dimensions:

### Adequacy (How much meaning is preserved?)

| Score | Description |
|-------|-------------|
| 5 | All meaning is preserved. Perfect or near-perfect translation. |
| 4 | Most meaning is preserved. Minor omissions or additions that don't change the core message. |
| 3 | Partial meaning preserved. Some important information is lost or distorted. |
| 2 | Little meaning preserved. Major content is missing or wrong. |
| 1 | No meaning preserved. Completely wrong, unintelligible, or not in Adja. |

### Fluency (How natural is the Adja?)

| Score | Description |
|-------|-------------|
| 5 | Perfectly natural Adja. Could have been written by a native speaker. |
| 4 | Minor awkwardness, but fully understandable. Slightly unnatural phrasing. |
| 3 | Understandable, but clearly not natural. Awkward word order or phrasing. |
| 2 | Barely understandable. Requires significant effort to parse. |
| 1 | Not Adja. Gibberish, fragments, or another language entirely. |

---

## Scoring Sheet

### Sentence [N]

**French:** [source sentence]

| System | Adja Translation | Adequacy (1-5) | Fluency (1-5) | Notes |
|--------|-----------------|----------------|---------------|-------|
| A | | | | |
| B | | | | |
| C | | | | |
| D | | | | |
| E | | | | |

**Pairwise preference (if applicable):**
- Between A and B: [ ] A is better [ ] B is better [ ] Equal
- Between A and C: [ ] A is better [ ] C is better [ ] Equal
- (repeat for all pairs of interest)

---

## Evaluator Information

- **Name:** _______________
- **Native language(s):** _______________
- **Adja dialect/region:** _______________
- **French proficiency:** [ ] Native [ ] Near-native [ ] Fluent
- **Date:** _______________
- **Total sentences evaluated:** _______________
- **Time spent (hours):** _______________

## Notes / General Observations

[Space for evaluator to note patterns, common errors, or general impressions]

---

## Inter-Annotator Agreement (filled by experiment coordinator)

**Krippendorff's alpha:**
- Adequacy: ___
- Fluency: ___
- Target: > 0.6

**Cohen's kappa (pairwise between annotators):**
- Evaluator 1 vs 2: ___
- Evaluator 1 vs 3: ___
- Evaluator 2 vs 3: ___

**Disagreement analysis:**
- Number of sentences with >1 point spread: ___
- Most common disagreement pattern: ___

```python
# Computing Krippendorff's alpha
import krippendorff
import numpy as np

# reliability_data: shape (n_annotators, n_items), NaN for missing
# Example: 3 annotators, 300 sentences, adequacy scores
alpha = krippendorff.alpha(reliability_data, level_of_measurement="ordinal")
```
