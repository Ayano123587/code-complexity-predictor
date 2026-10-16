# Experiment ledger

| Study | Selection evidence | Outcome | Runtime |
|---|---|---|---:|
| Structure v2 | Frozen validation; later compatible observed test | Retain lexical/structure25/75 ensemble | See per-run JSON |
| Mechanics v3 | Three grouped OOF folds + validation; >=0.005 F1 on both | No candidate passed | 93.99s |
| Family weights v4 | Same folds/gate, no new test score | No candidate passed | 332.26s |

Plans/results are in `models/structure-v2-weights`, `models/mechanics-v3` and `models/family-weight-v4`. They retain exact configs, class supports, histories and implementation/data hashes. The test is previously observed. Dataset SHA256 is recorded in the plans and provenance; do not silently rebuild the split or overwrite cached results. Source data, row predictions and checkpoints stay ignored. Notebook figures are executed aggregate analyses, not additional experiments.
