# Family-balanced training: existing model retained

This second grouped robustness hypothesis did not justify a new model. The plan was committed before scoring (`b33993d`); the earlier mechanics gate was neither relaxed nor replaced. Training weights use only the families and labels inside the respective fit fold. Original validation and previously observed test data do not contribute to fitted group sizes, class masses, vocabularies, or model parameters.

Each family contributes inverse-group-size sample weights, normalized to mean 1. The first variant retains the existing automatic class balancing. The second instead balances class mass after family weighting and disables automatic class weights to avoid double weighting. A third adds the previously tested mechanics features. Lexical TF-IDF document frequencies still use the available training documents; sample weighting changes the classifier objective, not corpus deduplication or IDF estimation. Grouping remains approximate.

All procedures retain 25% lexical / 75% structural probabilities and the same three grouped folds on 830 original training rows. Only candidates improving BOTH grouped-OOF and original validation macro F1 by at least 0.005 can be adopted.

| Candidate | Grouped OOF F1 | Validation F1 |
|---|---:|---:|
| Existing v2 | 0.558688 | 0.588250 |
| Family weighting, original class balancing | 0.553432 | 0.566020 |
| Family-aware class-mass balancing | 0.563477 | 0.581089 |
| Family weighting plus mechanics | 0.558972 | 0.680008 |

The highest validation score again fails to improve grouped OOF meaningfully: its gain is only 0.000284. Family/class balancing raises OOF by 0.004789 but worsens validation. This disagreement is evidence against promoting a model from the small validation score alone. No candidate passed both gates, so the production checkpoint remains unchanged and no holdout predictions were made. All fits completed; actual runtime 332.26 seconds, CPU two threads.

The rare-class/source limitations remain. Families have at most five rows; weights cannot replace missing independent constant/logarithmic/cubic examples. This corpus and earlier test results have been repeatedly observed. Neither this experiment nor the approximate title/clone grouping proves independent-source generalization or a global optimum.

The dataset review also checked [CodeComplex's author release](https://github.com/sybaik1/CodeComplex-Data), [CoRCoD's paper](https://arxiv.org/abs/1911.01155), and [Project CodeNet's schema](https://github.com/IBM/Project_CodeNet). The first two do not establish another equivalent expert-labeled C++ benchmark; CodeNet's runtime/difficulty fields are not substituted for asymptotic ground truth. No qualified larger C++ replacement was established in this round.

All 28 tests pass. Each actual saved candidate checkpoint loads and emits six finite normalized probabilities on the checked-in example. Tests cover family/class weight mass, duplicate-family invariance, weighted-model persistence, and malformed fit alignment. Data, row predictions and model artifacts remain ignored; plans, fold/class aggregates and hashes are committed.

## Training, evaluation and report entry points

```sh
python train.py family --output models/family-reproduction
python evaluation_scripts/report.py models/family-reproduction
python evaluation_scripts/evaluate.py family --output models/family-reproduction --observed-holdout
```

Training requires a new output path and copies the declared plan before fitting. It performs training/validation only. Explicit evaluation acknowledges the observed cohort, respects frozen selection, and refuses report overwrite; when the baseline was retained it does not rescore the test. Existing implementation modules and checkpoint import paths remain unchanged. Notebooks remain EDA-only. See the root `Report.md` for the full results overview.
