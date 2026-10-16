# Complexity predictor: results and reproduction

The retained model mixes 25% balanced lexical logistic regression (C=16) with 75% structural ExtraTrees (leaf size 3). On the frozen real TASTY time-complexity split, validation macro-F1 is 0.5882498879 and previously observed test macro-F1 is 0.6514987612. The older balanced classifier scored 0.3790096387 on that compatible observed test. These are retrospective comparisons, not independent-source generalization.

Training has 830 rows, validation 277 and observed test 275. Labels are six coarse classes; ambiguous `np` labels were excluded. Descriptive source comments and normalized clones approximate families because original problem URLs are absent. Minority support remains limited. Dataset/license details are in [docs/datasets.md](docs/datasets.md).

## Later validation studies

| Procedure | Grouped OOF macro-F1 | Validation macro-F1 |
|---|---:|---:|
| Retained v2 baseline | 0.558688 | 0.588250 |
| Mechanics, leaf 3 | 0.562338 | 0.676847 |
| Mechanics, leaf 1 | 0.579837 | 0.571750 |
| Family/class-mass weights | 0.563477 | 0.581089 |
| Family weights + mechanics | 0.558972 | 0.680008 |

Adoption required at least 0.005 macro-F1 improvement on both grouped OOF and validation. No candidate passed; validation spikes did not override grouped evidence. All variants remain in the [mechanics](models/mechanics-v3/results.json) and [family](models/family-weight-v4/results.json) aggregates. No later test rescore was used to select them.

## Reproduce

Install in a separate repository environment, fetch the pinned real dataset with `python -m src.data.tasty`, then preserve the frozen split in `models/tasty-optimized/split.json`. The root `train.py family --output models/NEW_RUN` copies the committed plan and runs development only. Its documented default cached paths must exist; a newly fetched dataset alone does not recreate the original split.

The retained ignored checkpoint is `models/structure-v2-weights/model.joblib`, SHA256 `e61c138907645ca751a8d130dd2471fc359abc6edb0827728086ce3f3bd0e6cf`. Production prediction uses `python -m src.models.predict_model examples/linear.cpp --model models/structure-v2-weights`. Scores are uncalibrated. Serialized models are trusted local artifacts, not arbitrary downloads.

The executed notebook series displays source EDA, validation comparisons and class-level failures, with core definitions visible in the main notebook. Figures in `docs/figures` come from those executed analyses. The notebooks do not train or score tests by default. Existing model inference is shown only on the shipped public example. The last source suite had 28 passing tests; notebook schema, code compilation and real analysis execution are checked separately.
