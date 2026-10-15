# Evaluation protocol and limitations

The split groups all submissions to the same problem and normalized near-duplicates (RapidFuzz normalized Indel similarity >= 0.88), transitively. This reduces identifier/comment/formatting leakage. It does not guarantee semantic clone detection. The split seed is the first seed starting at 42 whose grouped 70/30 split covers every class in both partitions; selection uses label coverage only, never prediction performance. A small corpus may have fewer/more held-out rows than 30% because groups are indivisible.

TF-IDF vocabulary and IDF, and the CNN vocabulary, fit on training rows only. Unknown neural tokens use UNK. CNN input truncates at 512 tokens and reports no semantic guarantee. The CNN has no test-selected checkpoint: it trains for the user-selected fixed epoch count.

Evaluation manifests contain corpus hash, label order, seed, train/test group counts, group overlap, held-out problem IDs, per-class precision/recall/F1, macro F1, accuracy, and confusion matrix. Real TASTY experiments and synthetic smoke runs are separate. The synthetic corpus has only 28 examples. Scores emitted by prediction are uncalibrated model scores, not confidence guarantees.

For a public corpus, add a validation partition by problem for hyperparameter tuning, freeze a separate test set, audit all near-duplicate clusters, compare majority and structural baselines, measure calibration/abstention, and report error cases by recursion, macros, library functions and input dimensions. No claim of a reliable Big-O analyzer is justified until then.


## Recorded experiments

| Experiment | Train/test rows | Accuracy | Macro F1 | Train-majority accuracy | Group overlap |
|---|---:|---:|---:|---:|---:|
| tasty-linear | 998 / 384 | 0.7109 | 0.3317 | 0.5599 | 0 |
| tasty-cnn | 998 / 384 | 0.6484 | 0.2155 | 0.5599 | 0 |
| tasty-space-linear | 923 / 473 | 0.7696 | 0.4136 | 0.4482 | 0 |
| linear | 13 / 15 | 0.3333 | 0.3452 | 0.0667 | 0 |
| cnn | 13 / 15 | 0.5333 | 0.5190 | 0.0667 | 0 |

`tasty-*` uses public TASTY C++ code; `linear` and `cnn` use synthetic teaching snippets. The TASTY CNN uses 20 fixed epochs and a 512-token truncation; all other hyperparameters are fixed in source. Macro F1 averages observed dataset classes, including rare classes. The space dataset has only three nlogn rows, making that class estimate especially unstable. Class-level metrics and support are in the manifests. These are one grouped split with no hyperparameter search, not confidence intervals or a clinical/production claim.

## Retrospective three-way optimization

[Plan](optimization-plan.md): fixed stratified clone-group fold assignment,830training/277validation/275holdout rows. Twenty-four real logistic candidates and four trained weighted-CNN/context ablations were compared on validation only. The selected balanced token pipeline (C16) achieved validation accuracy .7617/macroF1 .5450. The strongest CNN (1024tokens, sqrt-inverse frequency loss, epoch17 of20) achieved .7545/.5306.

On the once-evaluated retrospective holdout, selected pipeline accuracy=.7091/macroF1=.3790; untuned C4 word pipeline accuracy=.7382/macroF1=.3883; majority accuracy=.5709/macroF1=.1211. **The selected validation gain did not transfer to this holdout.** This result is preserved without switching models after seeing holdout scores. The fixed word comparison shares the new sublinear-TF/min_df2 representation and is not numerically identical to the earliest default TF-IDF implementation.

All fold groups are disjoint, but the underlying TASTY corpus had already been explored in prior experiments. These results do not constitute an independently untouched external evaluation. Source problem IDs remain unavailable; normalized clone/title grouping cannot prove semantic isolation. Per-row predictions are retained locally under ignored data; only aggregates and split/configuration/provenance records are committed.
