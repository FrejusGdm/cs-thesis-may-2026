# Results Table Templates

## Table 1: Main Results (LaTeX)

```latex
\begin{table*}[t]
\centering
\caption{Main results: Structured vs.\ Random data on French$\rightarrow$Adja translation.
All models are NLLB-200-distilled-600M fine-tuned with identical hyperparameters.
Results are mean $\pm$ std over 5 random seeds.
Best results in \textbf{bold}; $^\dagger$ indicates statistically significant improvement over RANDOM-10K (paired bootstrap, $p < 0.05$ with Bonferroni correction).}
\label{tab:main_results}
\begin{tabular}{lrccccc}
\toprule
\textbf{Condition} & \textbf{Size} & \textbf{BLEU}$\uparrow$ & \textbf{chrF}$\uparrow$ & \textbf{chrF++}$\uparrow$ & \textbf{TER}$\downarrow$ & \textbf{COMET}$\uparrow$ \\
\midrule
\multicolumn{7}{l}{\textit{Zero-shot baselines}} \\
ZERO-SHOT NLLB-200 & 0 & & & & & \\
Commercial LLM & 0 & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Random data}} \\
RANDOM-4K & 4,000 & & & & & \\
RANDOM-10K & 10,000 & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Intelligent selection baselines}} \\
LENGTH-STRATIFIED & 2,000 & & & & & \\
VOCAB-MAXIMIZED & 2,000 & & & & & \\
TF-IDF DIVERSE & 2,000 & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Structured data (ours)}} \\
STRUCTURED-2K & 2,000 & & & & & \\
STRUCTURED-4K & 4,266 & & & & & \\
\midrule
\multicolumn{7}{l}{\textit{Combined data}} \\
RANDOM-6K + STRUCTURED-4K & 10,000 & & & & & \\
RANDOM-10K + STRUCTURED-4K & 14,266 & & & & & \\
\bottomrule
\end{tabular}
\end{table*}
```

## Table 2: Module Ablation (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Module ablation results. Each row removes one module from the full structured dataset.
$\Delta$ shows chrF change from FULL condition. 3 seeds per condition.}
\label{tab:ablations}
\begin{tabular}{lrccc}
\toprule
\textbf{Condition} & \textbf{Size} & \textbf{BLEU} & \textbf{chrF} & $\Delta$\textbf{chrF} \\
\midrule
FULL (M1--M5) & 4,266 & & & --- \\
$-$NEGATION & & & & \\
$-$PAST & & & & \\
$-$FUTURE & & & & \\
$-$QUESTIONS & & & & \\
BASE-ONLY (M1) & & & & \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 3: Scaling Curve Data Points (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Scaling curve results. chrF scores (mean $\pm$ std, 5 seeds) at different training set sizes for structured vs.\ random data.}
\label{tab:scaling}
\begin{tabular}{rcc}
\toprule
\textbf{Size} & \textbf{Random chrF} & \textbf{Structured chrF} \\
\midrule
200 & & \\
500 & & \\
1,000 & & \\
2,000 & & \\
4,000 & & \\
6,000 & --- & \\
8,000 & --- & \\
10,000 & --- & \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 4: Cross-Lingual Transfer (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Cross-lingual transfer results. Adding related Gbe language data to Adja training. 3 seeds per condition.}
\label{tab:transfer}
\begin{tabular}{lrccc}
\toprule
\textbf{Condition} & \textbf{Size} & \textbf{BLEU} & \textbf{chrF} & \textbf{COMET} \\
\midrule
ADJA-ONLY & 4,266 & & & \\
ADJA + FON & 57K & & & \\
ADJA + EWE & 27K & & & \\
ADJA + FON + EWE & 80K & & & \\
FON-PRETRAIN $\rightarrow$ ADJA & 4,266 & & & \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 5: Human Evaluation (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Human evaluation results. Adequacy and fluency scored 1--5 by 3 native Adja-French bilingual evaluators.
Krippendorff's $\alpha$ = [VALUE]. $n=300$ test sentences per system.}
\label{tab:human_eval}
\begin{tabular}{lccc}
\toprule
\textbf{System} & \textbf{Adequacy} & \textbf{Fluency} & \textbf{Pref.\ \%} \\
\midrule
ZERO-SHOT NLLB & & & \\
Commercial LLM & & & \\
RANDOM-10K & & & \\
STRUCTURED-4K & & & \\
RANDOM-6K + STRUCTURED-4K & & & \\
RANDOM-10K + STRUCTURED-4K & & & \\
\bottomrule
\end{tabular}
\end{table}
```

## Table 6: Architecture Comparison (LaTeX)

```latex
\begin{table}[t]
\centering
\caption{Architecture comparison on RANDOM-6K + STRUCTURED-4K data (10K total). 5 seeds.}
\label{tab:architectures}
\begin{tabular}{lrccc}
\toprule
\textbf{Model} & \textbf{Params} & \textbf{BLEU} & \textbf{chrF} & \textbf{COMET} \\
\midrule
NLLB-200-distilled-600M & 600M & & & \\
NLLB-200-1.3B & 1.3B & & & \\
mBART-50 & 600M & & & \\
Transformer-base (scratch) & 65M & & & \\
Transformer-tiny (scratch) & 15M & & & \\
\bottomrule
\end{tabular}
\end{table}
```

## SacreBLEU Signatures (always report)

```
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.2.4.3
chrF2+numchars.6+space.false+version.2.4.3
chrF2++numchars.6+numwords.2+space.false+version.2.4.3
```
