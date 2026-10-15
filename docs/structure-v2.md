# Structural follow-up: retrospective TASTY experiment

This is real supervised learning on the existing TASTY C++ corpus. It is not a new external benchmark or a formal complexity analyzer. The corpus and the 275-row holdout were already observed in earlier work; all new choices used only the existing 277-row validation partition. Clone grouping and the original split remain unchanged (830 training rows). Missing source/problem URLs leave residual semantic-family leakage possible. No synthetic benchmark contributes to these results.

## Controlled iterations

Each stage wrote its plan before training and serialized selection before any final holdout prediction. The initial stage compared 11 representations/classifiers. Identifier-equality preservation, small constants, and canonical character features performed worse than legacy lexical normalization. Structural counts (loop nesting, direct recursion, library use) with balanced ExtraTrees improved validation macro F1. These are conservative lexical heuristics rather than a C++ parser.

The second stage tested two neighboring leaf sizes, removal of class weights, grouped sigmoid calibration, and a fixed lexical/structural ensemble. Calibration improved probability loss but hurt classification F1. The ensemble improved both. The final stage tested only 25/75 and 75/25 neighboring mixture weights against the equal blend, then stopped as prespecified. This small weighting plateau does not establish a global optimum.

| Candidate | Validation macro F1 | Validation accuracy | Validation log loss |
|---|---:|---:|---:|
| Replicated legacy balanced logistic C16 | 0.545019 | 0.761733 | 0.741849 |
| Structural balanced ExtraTrees leaf1 | 0.581071 | 0.776173 | 1.420115 |
| Structural grouped sigmoid | 0.499891 | 0.765343 | 0.734252 |
| Equal lexical/structural blend | 0.584867 | 0.805054 | 0.683165 |
| Selected 25% lexical / 75% structural | **0.588250** | 0.790614 | 0.707915 |
| 75% lexical / 25% structural | 0.580577 | 0.790614 | 0.688478 |

Macro F1 then accuracy was the declared selection rule, so the higher-accuracy/lower-loss equal blend was not substituted after selection. Both components are trained: balanced word TF-IDF logistic C16 and 400 balanced ExtraTrees with leaf size3. Preprocessing, vocabulary, trees, and calibration folds use training data only. Calibration uses three explicit stratified group folds; it was not selected. The deployed ensemble scores remain uncalibrated.

## Frozen descriptive holdout comparison

After selection, the selected model and the two matched baselines were refit on training plus validation, with no group overlap against holdout. The holdout was scored once in this study. These comparisons describe an already-observed corpus; they must not be advertised as fresh generalization evidence.

| Refit model | Observed holdout macro F1 | Accuracy | Log loss |
|---|---:|---:|---:|
| Selected ensemble | 0.651499 | 0.814545 | 0.578231 |
| Legacy balanced C16 | 0.379010 | 0.709091 | 0.762208 |
| Legacy unweighted C4 | 0.388322 | 0.738182 | 0.758976 |

Rare validation classes have only 6–7 examples; small prediction changes strongly affect macro F1. Repeated development on one corpus increases selection uncertainty. This experiment has no independent source test, no accuracy confidence guarantee, and no proof of Big-O correctness. The bounded dataset search did not establish another compatible independently expert-labeled C++ corpus. No new test labels were invented. Next research should acquire independently annotated source-disjoint C++ problems before extending this local hyperparameter search.

## Reproduction and artifacts

The three `models/structure-v2*` directories contain plans, aggregate validation histories, frozen selections, and the final aggregate report. Original results remain preserved. Dataset and split SHA256s are in every plan. Raw code, row-level predictions, and fitted checkpoints are ignored. The final checkpoint SHA256, environment, supports, and actual inference mechanics are recorded in aggregate JSON metadata.

From an isolated repository environment, fetch the pinned corpus with `python -m src.data.tasty`. Copy a study plan into a fresh output directory and run `python -m src.models.structure_v2 --output YOUR_DIRECTORY`. Existing selections refuse overwrite. For the final frozen choice, run the same command with `--release` once, then `python -m src.models.predict_model examples/linear.cpp --model YOUR_DIRECTORY`. The production loader retains legacy normalized checkpoints and uses raw source for the new feature pipelines.

The three validation stages took 88.83, 103.58, and 65.01 seconds of measured study wall time; final refits/evaluation took33.04 seconds. Native numeric threads and tree workers were bounded at 2. No paid compute or cloud run was used; no candidate fit failed. A first shell redirection failed before training because its ignored output directory did not yet exist.

The real checkpoint CLI check caught a serialization defect: running the module as `__main__` made its new estimator classes unresolvable in another process. Moving them into `src.models.ensemble_v2` and reserializing the trusted local estimators repaired portability without refitting or rescoring holdout. Eighteen tests cover lexical invariance, hidden comment/string removal, structural nesting/recursion, grouped calibration, raw-input inference, probability alignment, and importable estimator identity. The actual production CLI returns the linear class for the checked-in linear example, with six finite probabilities summing to 1. This checks inference mechanics, not additional accuracy.

## Validation-only stability diagnostic

After the frozen descriptive holdout report, a separately prespecified diagnostic tested the same 25/75 policy under tree seeds 10, 42, and 137. It explicitly prohibited deployment changes and holdout scoring. Validation macro F1 was 0.571911, 0.588250, and 0.594847 (standard deviation 0.009641); the equal three-seed average was 0.589653, only 0.001403 above the frozen seed 42 model. Runtime was 42.72 seconds. No seed was picked after inspecting these outcomes, and the production model stayed unchanged. This modest spread and minimal averaging gain support stopping this local search and prioritizing new independent annotations. These diagnostic results are validation estimates, not a second holdout experiment.
