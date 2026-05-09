# C1: Decode Samples (zero-shot inference)

Samples pulled directly from the HF Jobs log (`69e02cb9ac288e522d8eed44`). All three Whisper sizes hallucinate — none recover any real Adja text.

## whisper-tiny

**Reference**: Mina enu gbonnu ŋu
**Hypothesis**: p
**Notes**: Single character; effectively empty output.

**Reference**: Mimi Freud tamɛ bubu kpɔ́
**Hypothesis**: banana
**Notes**: Classic Whisper-tiny hallucination on unintelligible input.

**Reference**: Mì ava ɖu kpɔnnɔ miwó tɔ̀
**Hypothesis**: (empty)
**Notes**: Decoder emitted no tokens.

**Reference**: Gbe xonuxu nɔ nɔvinyɛ nyɔnuvi ahan o
**Hypothesis**: R-r-r-r-r-r-r-r-r-r-r-r-r-r-r-r-r-r-... (repetition loop)
**Notes**: Stuck in a repetitive decoding trap — explains the 2047% CER on test.

**Reference**: Yi lɔ wo le tootu
**Hypothesis**: banana
**Notes**: Same "banana" hallucination reappears on different audio.

## whisper-small

**Reference**: Mɛɖeka ɖeka ɖo ado emɔ nɔ agbe yitɔ́
**Hypothesis**: .
**Notes**: Just a period — language ID failed and decoder gave up.

**Reference**: ŋugbe sésé ɔ nyɔ ɖe ba
**Hypothesis**: ლლლლლლლლლლლლლლლლლლლლლლ... (Georgian letter repetition)
**Notes**: Whisper-small misidentified Adja as Georgian and emitted the letter "ლ" in a loop.

**Reference**: Ed'asɛŋnɔ edalɔ
**Hypothesis**: I'm just gonna say that I'm not gonna say anything about this.
**Notes**: English hallucination — decoder invented fluent but unrelated English.

**Reference**: Nɔ mi anu ɔ́ , agbetɔ yi le ahan
**Hypothesis**: www. www. www. www. www. www. www. www. ... (repeated)
**Notes**: Infinite repetition of "www." — another decoder collapse mode.

## whisper-large-v3

**Reference**: Yi ŋutɔ yi ɖo edro mɔ ye akpedonu
**Hypothesis**: IEI TAUBRO ET EMA IACETEM
**Notes**: All caps pseudo-Romance characters; no meaningful overlap.

**Reference**: Ego dodo vevide de gbeli o
**Hypothesis**: Ecco dodovici d'este video!
**Notes**: Italian hallucination — rough phonetic match in the middle of the string.

**Reference**: ŋugbe sɔ yi ɖe kɛ o
**Hypothesis**: Вы стоите кем? (Russian)
**Notes**: Russian hallucination; language ID fluctuates per sample.

**Reference**: É wawà evu do ta nii
**Hypothesis**: Eu awaei fu iutani.
**Notes**: Large-v3 at least tries lowercase phonetic transcription, but it's still wrong.
