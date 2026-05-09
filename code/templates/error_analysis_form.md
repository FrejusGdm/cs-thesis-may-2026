# Error Analysis Form

## Instructions

For each system output, classify ALL errors into the categories below. One sentence may have multiple errors. Mark the PRIMARY error (most severe) and all SECONDARY errors.

---

## Error Taxonomy

### Lexical Errors

| Code | Error Type | Description | Example |
|------|-----------|-------------|---------|
| L1 | Wrong word | Correct word class but wrong lexical choice | "eau" translated as Adja word for "food" |
| L2 | Untranslated | Source language word left in output | French word appears in Adja translation |
| L3 | Hallucinated | Word in output has no source correspondence | Extra content invented by model |
| L4 | Omitted | Source word has no target correspondence | Content word dropped from translation |

### Morphological Errors

| Code | Error Type | Description | Example |
|------|-----------|-------------|---------|
| M1 | Wrong tense | Incorrect tense/aspect marker | Present marker used for past tense |
| M2 | Wrong person/number | Incorrect agreement | Singular form for plural subject |
| M3 | Wrong tone/diacritic | Tonal or diacritical error | (If detectable in orthography) |
| M4 | Wrong class marker | Incorrect noun class agreement | (If applicable in Adja) |

### Syntactic Errors

| Code | Error Type | Description | Example |
|------|-----------|-------------|---------|
| S1 | Wrong word order | Incorrect constituent order | SVO produced instead of correct Adja order |
| S2 | Missing argument | Required argument not present | Verb without required object |
| S3 | Extra argument | Spurious argument added | Unnecessary word inserted |
| S4 | Wrong structure | Incorrect grammatical construction | Declarative structure for question |

### Semantic Errors

| Code | Error Type | Description | Example |
|------|-----------|-------------|---------|
| E1 | Negation error | Negation dropped or incorrectly added | Affirmative translated as negative |
| E2 | Question not formed | Question rendered as statement | Missing interrogative structure |
| E3 | Meaning distortion | Overall meaning significantly changed | Sentence means something different |

---

## Scoring Sheet

### Sentence [N]

**French:** [source]
**Reference Adja:** [reference translation]
**System output:** [model output]

| Error Code | Span in Output | Correct Form | Primary? |
|------------|---------------|-------------|----------|
| | | | [ ] |
| | | | [ ] |
| | | | [ ] |

**Overall:** [ ] Correct [ ] Minor errors [ ] Major errors [ ] Completely wrong

---

## Summary Statistics (fill after all sentences are analyzed)

| Error Type | Count | Rate (per 100 sentences) |
|------------|-------|--------------------------|
| L1 Wrong word | | |
| L2 Untranslated | | |
| L3 Hallucinated | | |
| L4 Omitted | | |
| M1 Wrong tense | | |
| M2 Wrong person/number | | |
| M3 Wrong tone/diacritic | | |
| M4 Wrong class marker | | |
| S1 Wrong word order | | |
| S2 Missing argument | | |
| S3 Extra argument | | |
| S4 Wrong structure | | |
| E1 Negation error | | |
| E2 Question not formed | | |
| E3 Meaning distortion | | |
| **TOTAL** | | |

**Completely correct sentences:** ___ / 100 (___ %)
**Completely wrong sentences:** ___ / 100 (___ %)

## Cross-System Comparison (filled by experiment coordinator)

After completing error analysis for all systems, run chi-squared tests to determine whether error distributions differ significantly across systems.

```python
import numpy as np
from scipy.stats import chi2_contingency

# contingency_table: rows = error types, columns = systems
# Each cell = count of that error type for that system
contingency_table = np.array([
    # [RANDOM-10K, STRUCTURED-4K, COMBINED, ...]
    # Lexical errors
    # Morphological errors
    # Syntactic errors
    # Semantic errors
])

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-squared: {chi2:.2f}, p={p_value:.4f}, dof={dof}")
```

**Inter-annotator agreement (Cohen's kappa):**
- Annotator 1 vs 2 (on primary error category): ___
- Target: > 0.6

## Annotator Information
- **Name:** _______________
- **Date:** _______________
- **System evaluated:** _______________
