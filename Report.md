# Complexity predictor: results and reproduction

The retained model mixes 25% balanced lexical logistic regression (C=16) with 75% structural ExtraTrees (leaf size 3). On the frozen real TASTY time-complexity split, validation macro-F1 is 0.5882498879 and previously observed test macro-F1 is 0.6514987612. The lexical baseline scored 0.3790096387 on that compatible observed test. These are retrospective comparisons, not independent-source generalization.

Training has 830 rows, validation 277 and observed test 275. Labels are six coarse classes; ambiguous `np` labels were excluded. Descriptive source comments and normalized clones approximate families because original problem URLs are absent. Minority support remains limited. Dataset/license details are in [docs/datasets.md](docs/datasets.md).

## Validation

The selected ensemble has three-fold grouped OOF macro-F1 0.558688 and frozen validation macro-F1 0.588250. A replacement must improve both measures by at least 0.005. Weights are selected using development evidence, not observed-test outcomes.

## Reproduce

Install in a separate repository environment, fetch the pinned real dataset with `python -m src.data.tasty`, then use the committed frozen split in `models/tasty-optimized/split.json`. Run `python train.py selected --data data/processed/tasty-time_complexity.jsonl --output models/my-ensemble`. This fits on 830 training rows to report validation, then refits the fixed configuration on all 1,107 training-plus-validation rows and saves a production-compatible checkpoint in a fresh directory. It does not repeat candidate search or score test rows. Explicit retrospective evaluation requires `evaluation_scripts/evaluate.py selected`, `--observed-holdout`, the dataset, checkpoint directory and a fresh report path.

The retained ignored checkpoint is `models/structure-v2-weights/model.joblib`, SHA256 `e61c138907645ca751a8d130dd2471fc359abc6edb0827728086ce3f3bd0e6cf`. Production prediction uses `python -m src.models.predict_model examples/linear.cpp --model models/structure-v2-weights`. Scores are uncalibrated. Serialized models are trusted local artifacts, not arbitrary downloads.

The executed notebook series displays source EDA, validation comparisons and class-level failures, with core definitions visible in the main notebook. Figures in `docs/figures` come from those executed analyses. The notebooks do not train or score tests by default. Existing model inference is shown only on the shipped public example. The source suite has 34 passing tests; notebook schema, code compilation and real analysis execution are checked separately.
