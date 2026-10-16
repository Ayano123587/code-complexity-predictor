# Code Complexity Predictor: results overview

The deployed model remains the validation-selected lexical/structural v2 ensemble trained on real TASTY C++. It is a statistical classifier, not a proof of complexity.

| Study | Selection evidence | Outcome |
|---|---|---|
| Legacy balanced logistic | Validation macro F1 0.545019 | Earlier baseline |
| Selected v2 ensemble | Validation macro F1 0.588250 | Retained production model |
| Code mechanics | Best validation 0.676847 but OOF gain 0.003650 | Confirmation gate failed |
| Family-balanced mechanics | Validation 0.680008 but OOF gain 0.000284 | Confirmation gate failed |

Previously observed holdout macro F1 is 0.651499 for v2 versus 0.379010 for the legacy balanced baseline. Those are retrospective, not new independent evidence. No holdout score was recomputed for the rejected mechanics/family studies. Minority classes and missing original problem URLs limit confidence.

- [Full original controlled study](docs/structure-v2.md)
- [Grouped mechanics study](docs/mechanics-v3.md)
- [Family-weighting study](docs/family-weight-v4.md)
- [Dataset provenance and limitations](docs/datasets.md)

`train.py` launches validation-only module implementations into a fresh directory. `evaluation_scripts/report.py` reads aggregates without data/model loading. Explicit observed-holdout evaluation is separate and does not rescore when the existing model is retained. `src/` contains implementation, `tests/` verifies invariants, and notebooks are EDA-only. All datasets, row predictions, trained weights and secrets remain ignored.

This train/evaluate/report organization follows the supplied [project-structure reference](https://github.com/yash9439/Finetuning-Opus-MT-en-fr-WikiData); implementations, protocols, data and results here are original to this project.
