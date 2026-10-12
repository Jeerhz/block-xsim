# Results

| file | model | content |
|---|---|---|
| `flores_xsim.json` · `flores_minor.json` | `EvaluationResult` | FLORES+ (dev + devtest), windows of 5 sentences, k = 1..5 |
| `ntrex_xsim.json` · `ntrex_minor.json` | `EvaluationResult` | NTREX-128, windows of 14 sentences, k = 1..14 |
| `corpus_statistics.json` | `CorpusStatisticsResult` | blocks kept per k, their tokens, sentences each edit applies to |

Settings of every file: languages en→de, es, fr, ru; D = 5 distractors; at most 64
distractors per block and regime (seed 0); 512 XLM-R tokens; edits built with seed 42.
A result file has this shape: `scorers.<model>.regimes.<regime>.by_lang.<lang>.<k>` or `scorers.<model>.regimes.<regime>.mean_by_k.<k>`.

Two rules decide which cells appear:

- A cell needs at least 2 blocks.
- Blocks a model would truncate are left out. CometKiwi therefore has no NTREX cell beyond k = 11.

The language mean of a cell sums the counts and averages the rates, with each language weighted equally.

The evaluations ran from 2026-09-28 to 2026-10-04. A rerun with this repository on 2026-10-10
used the same blocks in every cell. Absolute score differences agree within 2e-7. Accuracies
differ by at most 0.003 for language means and 0.01 for single cells, where near-ties flip
under other batch sizes. That rerun produced the scores of the published dataset. One cell was changed in conversion:
NTREX xSIM++, CometKiwi, fr, k = 11 had a 1-block cell in the `every_3rd` and `one_sentence`
regimes only, which is removed here. The k = 11 means of these two regimes therefore cover
7 blocks (de, es) instead of 8.
