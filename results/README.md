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

## MetricX-24 and BLEU (`metricx_bleu/`)

`metricx_bleu/<corpus>_<edits>.json` holds two more scorers, run on 2026-11-10 on the same blocks and distractors:

- **`metricx-qe`:** MetricX-24 hybrid XL (`google/metricx-24-hybrid-xl-v2p6-bfloat16`) in QE mode, reading source and candidate.
  - Its error score (0 to 25) is negated, so higher is better.
  - A block is kept when source and candidate fit in 1,536 mT5 tokens, counted with `google/mt5-xl`.
- **`bleu`:** sentence BLEU (sacrebleu, 13a, effective order) of each candidate against the reference block.
  - The reference scores 100 against itself, so its accuracy is 1 by construction.
  - Its score difference is 100 minus the BLEU of the distractor.

Two caveats apply:

- Only FLORES+ is here; the NTREX-128 runs were still in progress.
- MetricX is computed with the transformers MT5 model and the official weights, following the official code: input format, final EOS removed, logit of `<extra_id_10>`, clamp. Its numbers have not yet been compared with the official `metricx24.predict` script.

## Verification (2026-11-11)

Checks made on the published dataset (German and French NTREX, German FLORES+), the public
NTREX text and the code, after MetricX and BLEU showed the same one-error profile as CometKiwi.

What was verified and holds:

- Every distractor differs from the reference exactly at its stated positions (320,000 distractors checked); none is an exact duplicate; source and reference are the official NTREX lines in order; the block of length k+1 extends the block of length k.
- Every committed cell is recomputable from the per-candidate scores of the dataset to within 0.004 in accuracy (near-ties under another batch size). Ties are below 0.1 %.
- The curves keep their shape on a fixed set of windows (those kept at the shortest k), so the block set growing with k does not create them.
- The perturber does what each category says on 800 NTREX sentences (lexicon pairs, same-width digit scrambles, ±1 digits, subsequence omissions, window entities).
- BLEU recomputed locally falls exactly as 1/k under one error (fitted exponent −1.0) and is flat with one error per sentence: an n-gram average cannot do otherwise. BLEU is the null model of pure dilution, which is why its one-error curve matches CometKiwi's. The decay exponents differ: BLEU −1.0, CometKiwi −0.75 (xSIM++) and −0.99 (minor), MetricX −0.46 and −0.71.
- The QE metrics receive the English source and the candidate only; BLEU receives the candidate and the reference.

Biases found, none of which changes the dilution conclusions (they hold on the minor edits, which have no negation):

- **Inserted negations are often ungrammatical.** They are 72 % of the causality edits, so about a third of the xSIM++ single edits in German and French. On 20 hand-judged sentences per language, 9 (de) and 10 (fr) were ungrammatical: German puts *nicht* after the finite verb even in verb-final clauses ("…erstochen wurde nicht"), French anchors *ne* before subject and relative pronouns ("n'Il a pas", "ne qui se trouvait pas"). CometKiwi detects the fluent causality edits as well as the inserted negations (.93 to 1.00), so its causality level stands; the encoders' levels mix meaning and fluency (LaBSE on German: .81 on inserted negations, .53 on removed ones).
- **The entity bank carries NER noise** from the small spaCy models: about 6 % of entity edits replace or insert a non-entity ("Los Angeles" → "lautstark"). Detection on these is not higher than on clean spans.
- **One edit can reach the same text through two categories** ("Premier League" → "Dernier League" as antonym and as ordinal, 29 French distractor pairs differing in spacing), and one case-only entity edit exists ("Walisischen" → "walisischen", one German window). `check_blocks.py` now reports both.
- An omission can drop an obligatory word ("seit langer Zeit" → "seit Zeit"); the first sentence of a NTREX window is usually a headline; at odd k the every-2nd regime edits slightly more than half of the sentences; the k = 2 cells rest on 24 to 40 windows per language.

Not verified: the MetricX scores against the official `metricx24.predict` script (needs a GPU), and the Spanish and Russian edits.
