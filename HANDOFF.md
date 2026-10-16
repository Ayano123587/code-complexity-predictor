# Complexity improvement handoff

Branch: `codex/improve-v2`. Worktree: `D:/CodexData/portfolio-improvements/code-complexity-predictor`. Original checkout unchanged. Parent handles publication.

Three controlled validation stages selected 25% legacy balanced logistic and 75% structural balanced ExtraTrees. Macro F1 rose from 0.545019 to 0.588250; matched previously observed holdout scores are 0.379010 versus 0.651499. This is retrospective development, not fresh independent evidence. Preserve old reports and all caveats in `docs/structure-v2.md`. Subsequent three-seed diagnostic is validation-only, changes no deployment, and averages to 0.589653: local plateau, independent annotation remains the priority.

Validation: `python -m pytest -q` => 18 passed. Actual `python -m src.models.predict_model examples/linear.cpp --model models/structure-v2-weights` returns linear with six finite probabilities summing to 1. Serialization portability was repaired through importable estimator classes; local checkpoints reserialized without refitting/rescoring.

Ignored artifacts to preserve/copy into the original repository if desired: `models/structure-v2-weights/model.joblib` (SHA256 e61c138907645ca751a8d130dd2471fc359abc6edb0827728086ce3f3bd0e6cf), validation-model checkpoints under all four study directories, and `data/interim/*structure-v2*` logs/predictions. Raw/processed data and binaries are never committed. The original corpus copy is already available in the original checkout.

For continuation: no more local weight tuning based on the observed holdout. Acquire source-disjoint independently annotated C++ Big-O examples before external evaluation; missing problem IDs and six/seven-row validation minority classes limit confidence. CPU-only measured study work totaled 333.18 seconds across validation/final-refit/stability stages. No cloud or paid run.

## Resumed mechanics round

Prespecified grouped OOF confirmation retained v2. Baseline OOF/validation macro F1 is 0.558688/0.588250; mechanics leaf3 is 0.562338/0.676847 and fails the 0.005 OOF gain gate. Leaf1 improves OOF but worsens validation. No holdout rescore. See `docs/mechanics-v3.md` and aggregate manifests. 23 tests and actual saved artifact/retained CLI checks pass. Runtime 93.99 seconds; data/models remain ignored.
