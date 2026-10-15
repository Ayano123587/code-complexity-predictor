# Code Complexity Predictor

An installable research pipeline that classifies **C++ source code into coarse time- and space-complexity classes** using normalized code tokens. Includes a trained TF-IDF/logistic regression baseline, a trainable PyTorch token CNN, transitive problem/near-duplicate grouping, evaluation manifests, and prediction CLI. Source is never compiled or executed.

**Current status: functional research prototype evaluated on the real TASTY C++ corpus, plus a clearly separate synthetic smoke benchmark.** Time and space labels are learned separately. This is a statistical classifier, not a proof-producing complexity analyzer. Algorithm-paradigm prediction is not claimed.

## Run

```sh
python -m pip install -e ".[dev]"
python -m src.data.make_dataset
python -m src.models.train_model
python -m src.models.predict_model examples/linear.cpp
python -m pytest -q
```

Deep learning (CPU works; CUDA can be provisioned separately):

```sh
python -m pip install -e ".[neural]"
python -m src.models.train_model --backend cnn --output models/cnn --epochs 60
python -m src.models.predict_model examples/linear.cpp --model models/cnn
```

Training artifacts stay local; checked-in `models/*/evaluation.json` and `loss.json` record actual runs. Reproduce them by running the commands. Load joblib artifacts only from trusted training runs because pickle deserialization can execute code.

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
