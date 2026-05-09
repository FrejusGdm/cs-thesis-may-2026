# Anticipated Reviewer Objections and Defenses

## Objection 1: "The structured data is too artificial / narrow in linguistic coverage"

**Attack:** Your structured data only covers SVO with 10 verbs and 5-6 objects. Real language is infinitely more complex.

**Defense:**
- Test set includes 20% out-of-distribution structures (relative clauses, adjectives, multi-clause) that neither approach trains on
- Report disaggregated results: IN-DISTRIBUTION vs OUT-OF-DISTRIBUTION
- Frame as proof-of-concept for data efficiency, not as "solved low-resource MT"
- The NATURAL-UTTERANCES test set (100 real-speech sentences) tests ecological validity

## Objection 2: "You confound data composition with data quality"

**Attack:** The random data is likely noisy or inconsistent. You're measuring quality vs noise, not structure vs randomness.

**Defense:**
- Quality-audit the random data with a native speaker; report error rates
- If quality issues exist, run RANDOM-CLEAN condition and compare
- If RANDOM-CLEAN still loses, composition argument holds; if it catches up, reframe contribution

## Objection 3: "Your test set is biased toward the structured approach"

**Attack:** If the test set uses similar grammar as structured training data, you're testing memorization.

**Defense:**
- NATURAL-UTTERANCES test (100 sentences) from real speech is guaranteed OOD
- HELDOUT-DIVERSE includes unseen verbs AND unseen structures
- Report vocabulary overlap statistics between test set and each training condition

## Objection 4: "Single language pair — no generalizability"

**Attack:** Results on Adja tell us nothing about other languages.

**Defense:**
- Run pseudo-low-resource simulation on French-Yoruba or English-Swahili (subsample existing high-resource pair to small sizes, apply same structured vs random protocol)
- Discuss adaptation framework for other Gbe languages and typologically different languages
- Note that Adja is typologically representative of isolating/analytic SVO languages (many West African languages share this profile)
- Acknowledge as primary limitation; propose multi-language validation as future work

## Objection 5: "No comparison with active learning / submodular selection"

**Attack:** You only compare against random. Smarter selection methods exist.

**Defense:**
- Baselines 2-4 cover common intelligent selection (length-stratified, vocab-maximized, TF-IDF diverse)
- Acknowledge active learning as future work
- If compute allows, implement one round of active learning as additional baseline

## Objection 6: "BLEU unreliable for African languages"

**Attack:** AfriCOMET/AfriMTE showed BLEU correlates poorly with human judgments.

**Defense:**
- chrF/chrF++ as primary automatic metric (better for morphologically rich languages)
- Human evaluation is a first-class result, not an afterthought
- Fine-tune COMET on Adja data following AfriCOMET methodology
- Report all metrics; discuss disagreements

## Objection 7: "Only 5 seeds is insufficient"

**Attack:** With small BLEU differences, variance may dominate.

**Defense:**
- 5 seeds meets ARR standard (minimum 3, preferred 5)
- Report confidence intervals, not just means
- Paired bootstrap on test set (1,000 samples) + cross-seed Wilcoxon
- Report Cohen's d effect sizes

## Objection 8: "Minimal pairs are redundant — inflating apparent dataset size"

**Attack:** Your 4K structured sentences are really ~900 base sentences repeated with trivial transformations.

**THIS IS THE MOST DANGEROUS OBJECTION. Address head-on:**
- BASE-ONLY ablation: test ~900 base sentences alone
- Compare BASE-ONLY vs RANDOM-900 at truly equal size
- If FULL >> BASE-ONLY, minimal pairs provide genuine signal
- Compute information-theoretic diversity: type-token ratio, vocabulary entropy, average surprisal under a French LM
- If structured data is more diverse by these measures, report it

## Objection 9: "The neurosymbolic framing is overstated"

**Attack:** This isn't neurosymbolic AI — there's no hybrid reasoning during inference.

**Defense:**
- Reframe as "neurosymbolic data engineering" — symbolic linguistic knowledge at construction stage, neural learning at inference
- Or drop the framing entirely and call it "linguistically-informed corpus design"
- Cite precedent: "applied neurosymbolic" is used broadly in recent literature for any integration of symbolic knowledge with neural systems

## Objection 10: "The BLEU ~20 from prior work is uncontrolled"

**Attack:** The original 10K+4K result has no controls. How do we know it's not an artifact?

**Defense:**
- Experiment 1 is specifically designed to reproduce and control this result
- Test RANDOM-14K to disentangle size vs composition
- Test STRUCTURED-4K-ONLY to isolate the structured data's contribution
- Multiple seeds + significance testing
