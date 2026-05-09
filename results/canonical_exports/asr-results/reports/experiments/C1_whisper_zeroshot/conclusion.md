# C1: Conclusion

## Verdict
**baseline_only** — expected failure mode, serves as the "why you need fine-tuning" evidence.

## What happened
All three Whisper sizes (tiny, small, large-v3) produced catastrophic zero-shot output on Adja. The tiny model hallucinates single words such as "banana" or collapses into character-repetition loops ("r-r-r-r..."). Whisper-small's language-ID misfires — it misidentifies Adja audio as Georgian and emits long strings of "ლ", or drops into English filler ("I'm just gonna say..."), or loops on "www. www. www...". Large-v3 behaves marginally better in that it at least produces bounded output, but it still guesses the language per utterance (Italian, Russian, pseudo-Romance caps) with effectively zero semantic overlap.

The CER and WER numbers above 100% are a direct consequence of these hallucinations: hypotheses are much longer than references, so normalized edit distance exceeds the reference length.

## Why
Adja (`aj_Latn`) is entirely out-of-distribution for the Whisper pretraining mix. Whisper was trained primarily on high-resource European/Asian languages; Gbe-family languages from Benin/Togo are not represented. The decoder has no prior mass on Adja-specific graphemes (ɛ, ɔ, ŋ, ɖ) or tone marks, and the language-ID head has no class that matches the audio distribution. The model therefore picks the closest language it has seen and produces fluent-looking text in that language instead.

## What to do next
Done. Keep as reference baseline. No further action on this experiment — the point has been made and future reports can cite these numbers when arguing for fine-tuning. The natural next step is `C2_whisper_finetune/` which fine-tunes Whisper-small on the Adja train split.

## Known follow-ups already queued
None — this is a terminal baseline.
