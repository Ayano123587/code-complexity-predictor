# Group-confirmed code mechanics: no model replacement

The existing v2 ensemble remains deployed. New loop/recursion mechanics improved some validation results but did not pass the prespecified requirement for improvement on both grouped out-of-fold predictions and the original validation partition. No new holdout predictions were made.

The hypothesis and objective were committed before scoring (`680baa5`). Conservative name-invariant features distinguish multiplicative/additive loop updates, fixed bounds, bounds referring to an enclosing loop variable, sorting inside loops, and direct recursive calls with division or subtraction arguments. Comments and string contents are removed; source is never executed. These covariates are not a complete C++ parser, a recurrence solver, or a complexity proof. Unsupported language constructs can be missed.

All four procedures preserve the frozen lexical/structural mixture weights (25%/75%). They compare the existing ExtraTrees leaf 3 baseline, enriched ExtraTrees leaf 3/leaf 1, and enriched balanced RandomForest leaf 3. Each tree ensemble has 400 trees. Training-only three-fold stratified group splits generate out-of-fold predictions for all 830 original training rows. Group overlap is zero in each fold; preprocessing and models are refit inside each fold. Separate 277-row validation scores use models trained on the original training partition only.

A replacement must improve pooled grouped-OOF macro F1 and original validation macro F1 by at least 0.005 each. Eligible candidates are ranked by grouped-OOF F1. This confirmation rule was not relaxed after seeing a large isolated validation gain.

| Candidate | Grouped OOF macro F1 | Validation macro F1 | Passes both gates? |
|---|---:|---:|---|
| Existing v2 ensemble | 0.558688 | 0.588250 | Reference |
| Mechanics ExtraTrees leaf 3 | 0.562338 | 0.676847 | No: OOF gain 0.003650 |
| Mechanics ExtraTrees leaf 1 | 0.579837 | 0.571750 | No: validation worsens |
| Mechanics RandomForest leaf 3 | 0.472710 | 0.576491 | No |

The leaf 3 gain on the small original validation cohort does not establish family-level robustness. Its OOF constant-class F1 falls from 0.1875 to 0.1765 and logarithmic-class F1 from 0.4615 to 0.4286, while cubic F1 rises from 0.5000 to 0.5641. Only 21 constant/cubic and 15 logarithmic training examples contribute to the OOF estimates. Original validation minority supports remain 6–7. These weaknesses discourage more local threshold/seed searching merely to pass a narrow gate.

The training partition contains 647 approximate family/clone groups;132 have multiple rows (315 rows total), with maximum group size 5. The inherited grouping uses source title signatures and normalized clone similarity. Missing original problem URLs prevent proof of semantic-family independence. All this corpus and earlier test results have already been observed, so the study is retrospective development. Grouped confirmation improves the audit but does not create an independent benchmark.

## Reproduction and actual artifacts

Run from the repository's isolated environment:

```sh
python -m pytest -q
python -m src.models.mechanics_v3 --output models/mechanics-reproduction
python -m src.models.predict_model examples/linear.cpp --model models/structure-v2-weights
```

Copy `models/mechanics-v3/plan.json` into the new reproduction directory before running. Existing selections refuse overwrite. The runner uses the pinned processed TASTY corpus and original grouped split; all hashes, candidate metrics, fold reports, runtime and failures are retained in `models/mechanics-v3`. The actual study took 93.99 seconds with two numeric threads/tree workers and no candidate failures.

All 23 tests pass, including mechanics rename invariance, hidden comment/string exclusion, braced/braceless nested bounds, recursive argument recognition, and the two-part deployment gate. Every saved candidate validation checkpoint loads and emits six finite normalized probabilities on the checked-in example. The retained production CLI also runs successfully; its exact checkpoint remains unchanged. This is inference mechanics evidence, not new accuracy evidence. Checkpoints and all row-level OOF predictions stay ignored; only aggregate reports are committed.

No test score was recomputed after the gate retained v2. Independent expert-labeled, source-disjoint C++ examples remain the next priority; the current data do not justify a stronger complexity claim.
