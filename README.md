# Code Complexity Predictor

An installable research pipeline that classifies **C++ source code into coarse time- and space-complexity classes** using lexical and structural representations. Includes a trained TF-IDF/logistic regression baseline, a trainable PyTorch token CNN, transitive problem/near-duplicate grouping, evaluation manifests, and prediction CLI. Source is never compiled or executed.

**Current status: functional research prototype evaluated on the real TASTY C++ corpus, plus a clearly separate synthetic smoke benchmark.** Time and space labels are learned separately. This is a statistical classifier, not a proof-producing complexity analyzer. Algorithm-paradigm prediction is not claimed.

## Run

```sh
python -m venv .venv
# Windows PowerShell: activate this repository's environment
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m src.data.tasty
python -m src.models.train_model
python -m src.models.predict_model examples/linear.cpp
python -m pytest -q
```

Deep learning (CPU works; CUDA can be provisioned separately):

```sh
python -m pip install -e ".[neural]"
python -m src.models.train_model --backend cnn --output models/tasty-cnn --epochs 20
python -m src.models.predict_model examples/linear.cpp --model models/tasty-cnn
```

The default CLI/Makefile workflow uses real TASTY C++ data. Synthetic smoke training requires explicit `make smoke`. Training artifacts stay local; checked-in `models/*/evaluation.json` and `loss.json` record actual runs. Reproduce them by running the commands. Load joblib artifacts only from trusted training runs because pickle deserialization can execute code.

## Reproduce the real TASTY C++ experiment

```sh
python -m src.data.tasty
python -m src.models.train_model --data data/processed/tasty-time_complexity.jsonl --output models/tasty-linear
python -m src.models.train_model --data data/processed/tasty-time_complexity.jsonl --backend cnn --output models/tasty-cnn --epochs 20
python -m src.data.tasty --target space_complexity
python -m src.models.train_model --data data/processed/tasty-space_complexity.jsonl --output models/tasty-space-linear
python -m src.models.predict_model examples/linear.cpp --model models/tasty-linear
python -m src.models.predict_model examples/linear.cpp --model models/tasty-space-linear
```

The public CSV is pinned by revision and SHA256. Raw and processed code remain ignored. Authors declare CC-BY-3.0; attribution, source limitations, and exact preprocessing counts are committed in `models/tasty-*-provenance.json` and [dataset research](docs/datasets.md). Every comment is stripped, including complexity annotations; ambiguous `np` labels are excluded. Source problem URLs are absent, so title-signature and normalized clone grouping cannot prove all semantic relatives stay together.

Actual results and class balance are documented in [evaluation](docs/evaluation.md). No CUDA/Kaggle job was needed for these CPU runs.

## Your labeled corpus

Use UTF-8 JSONL, one object per solution:

```json
{"code":"int f(int n,int*a){return a[0];}","label":"constant","problem_id":"judge/problem-1","source":"corpus citation"}
```

Classes: `constant`, `logn`, `linear`, `nlogn`, `quadratic`, `cubic`, `exponential`. Labels require expert annotation with the input-size variable and operational assumptions recorded in your corpus documentation. Do not equate judge difficulty/runtime with Big-O.

```sh
python -m src.data.make_dataset --input data/raw/annotated.jsonl --output data/processed/labeled.jsonl
python -m src.models.train_model --data data/processed/labeled.jsonl --output models/public-corpus
```

An HTTPS fetch adapter accepts a user-selected licensed JSONL URL, mandatory SHA256, and license identifier:

```sh
python -m src.data.fetch_dataset URL --sha256 EXPECTED_SHA256 --license SPDX_OR_LICENSE_URL
```

No automatic scraping of judges or unlicensed dataset redistribution. See [dataset research](docs/datasets.md) and [evaluation protocol](docs/evaluation.md).

## Repository map

`src/data` handles ingestion and grouping; `src/features` tokenizes C++; `src/models` trains and predicts; `src/visualization` prints confusion matrices; `notebooks` contains EDA only; `tests` and `.github/workflows` validate the pipeline. `data/{raw,external,interim,processed}` and `.env` are ignored. Model binaries are ignored; numerical summaries are committed.

## Boundaries

Lexical normalization handles ordinary comments and string literals, but is not a complete C++ parser or preprocessor. Macros, templates, undefined behavior, multiple input dimensions, library cost assumptions, recursion and adversarial source can invalidate predictions. `exponential` is a coarse class, not a precise base or polynomial factor. Similarity clustering is quadratic and can conservatively group different labels. Use a scalable token MinHash index for large corpora, retaining the same transitive split constraints.

## Validation-selected optimization study

The initial single train/test experiment is followed by a frozen three-way, clone-grouped TASTY study. It reuses a previously explored corpus: its retrospective holdout is **not an independently untouched external test set**. See [prespecified plan](docs/optimization-plan.md).

Actual search: 24 logistic pipelines across token/character/combined/structural features, regularization and class balancing; four CNN ablations across512/1024-token context and frequency-based class weights,20epochs each. Selection used validation macroF1 only, with stated tie rules. Best linear validation macroF1=.5450; best CNN=.5306. The selected linear model scored .3790 macroF1 on the retrospective holdout, versus .3883 for the fixed word baseline. This negative generalization result is retained; no test-guided replacement was made.

Reproduce the declared study in a new output directory (existing frozen results cannot be overwritten):

```sh
python -m src.models.optimize freeze --output models/reproduction
python -m src.models.optimize search --output models/reproduction
python -m src.models.neural_search --output models/reproduction --epochs 20
python -m src.models.optimize evaluate --output models/reproduction
python -m src.models.predict_model examples/linear.cpp --model models/tasty-optimized
```

The selected model is refit on training+validation after configuration selection. Aggregate reports, class-level metrics, histories, frozen split IDs and configuration hashes are committed; per-row predictions are saved under ignored `data/interim/`, and binaries remain ignored. No claim of globally optimal models or unseen-source generalization is made.

Use a separate virtual environment for each repository. These research projects use the requested generic `src` package layout; installing several together in one environment is unsupported. On Linux/macOS, activate with `source .venv/bin/activate` instead of the PowerShell command. All `python -m ...` and installed CLI commands assume the current repository directory and its activated environment. On Windows, direct `.venv\Scripts\python.exe -m ...` commands can also run without activation.

## Controlled structural follow-up

The 2026 follow-up compares lexical, structural, calibrated, and fixed ensemble models on the existing grouped validation split. Read [the study report](docs/structure-v2.md) before interpreting its results: the corpus and holdout have previously been observed. Scores are estimates rather than complexity proofs.

Run from this repository in its own environment; the five research repositories share a generic `src` package and must not be installed into one environment.

```powershell
Set-Location D:\CodexData\portfolio\code-complexity-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m src.data.tasty
python -m src.models.predict_model examples/linear.cpp --model models/structure-v2-weights
```

Model binaries remain local. To reproduce studies in a fresh clone, run `python -m src.models.structure_v2 --output models/structure-v2`, then the same command with `models/structure-v2-followup` and `models/structure-v2-weights`. Committed selection/results refuse overwrite: preserve those manifests and use a copied output directory with its plan only for reruns. `--release` refits the frozen selected model on train plus validation and performs the descriptive observed-holdout comparison once.

The later [group-confirmed mechanics study](docs/mechanics-v3.md) retained this model: new loop/recursion features did not improve both grouped out-of-fold and validation macro F1 by the prespecified margin. No new holdout scoring was performed.

See [Report.md](Report.md) for the consolidated baseline/candidate table. Root `train.py` runs validation-only studies into fresh directories; `evaluation_scripts/` separates aggregate reporting and explicit observed-cohort evaluation. The later [family-balanced study](docs/family-weight-v4.md) also retained v2.
