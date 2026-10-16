"""Reproduce the fixed lexical/structural ensemble without implicit test scoring."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.data.make_dataset import load
from src.models.ensemble_v2 import EqualEnsemble
from src.models.structure_v2 import word, structure, extra, sha, scores


def model():
    lexical = Pipeline([('features', word('legacy')), ('classifier', LogisticRegression(C=16, class_weight='balanced', max_iter=3000, random_state=42))])
    structural = Pipeline([('features', structure()), ('classifier', extra(3))])
    return EqualEnsemble([lexical, structural], weights=[.25, .75])


def cohort(data, split_path):
    split = json.loads(Path(split_path).read_text(encoding='utf-8'))
    if sha(data) != split['data_sha256']:
        raise ValueError('Frozen dataset fingerprint mismatch')
    rows = load(data)
    parts = split['partitions']
    indexes = [i for key in ['train', 'validation', 'holdout'] for i in parts[key]]
    if sorted(indexes) != list(range(len(rows))) or len(split['groups']) != len(rows):
        raise ValueError('Frozen split must cover each row exactly once')
    group_sets = [set(split['groups'][i] for i in parts[k]) for k in ['train', 'validation', 'holdout']]
    if any(group_sets[i] & group_sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Frozen split groups overlap')
    return np.asarray([r['code'] for r in rows]), np.asarray([r['label'] for r in rows]), split


def train(data, output, split_path='models/tasty-optimized/split.json'):
    output = Path(output)
    if output.exists():
        raise ValueError('Fresh output directory required')
    x, y, split = cohort(data, split_path)
    trained = model()
    with threadpool_limits(limits=2):
        trained.fit(x[split['partitions']['train']], y[split['partitions']['train']])
    val = split['partitions']['validation']
    report = {'input_representation': 'raw_cpp_v2', 'selected': 'legacy_structure_weight25',
              'data_sha256': sha(data), 'split_sha256': sha(split_path),
              'validation': scores(trained, x[val], y[val], split['label_order']),
              'test_evaluated': False, 'probability_warning': 'Uncalibrated model scores'}
    combined = split['partitions']['train'] + val
    with threadpool_limits(limits=2):
        trained = model().fit(x[combined], y[combined])
    report['final_training_rows'] = len(combined)
    output.mkdir(parents=True)
    with (output/'model.joblib').open('xb') as stream:
        joblib.dump(trained, stream)
    report['checkpoint_sha256'] = sha(output/'model.joblib')
    with (output/'evaluation.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report


def evaluate(data, checkpoint_directory, report, *, acknowledge_observed=False, split_path='models/tasty-optimized/split.json'):
    if not acknowledge_observed:
        raise ValueError('Explicit acknowledgement of previously observed test required')
    report = Path(report)
    if report.exists():
        raise ValueError('Fresh report path required')
    x, y, split = cohort(data, split_path)
    checkpoint_directory = Path(checkpoint_directory)
    metadata = json.loads((checkpoint_directory/'evaluation.json').read_text(encoding='utf-8'))
    required = {'data_sha256': sha(data), 'split_sha256': sha(split_path),
                'checkpoint_sha256': sha(checkpoint_directory/'model.joblib')}
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError('Checkpoint provenance mismatch or missing binding; reproduce with selected training before evaluation')
    trained = joblib.load(checkpoint_directory/'model.joblib')
    indexes = split['partitions']['holdout']
    result = {'test': scores(trained, x[indexes], y[indexes], split['label_order']),
              'rows': len(indexes), 'observed_retrospective': True,
              'checkpoint_sha256': sha(checkpoint_directory/'model.joblib'), 'split_sha256': sha(split_path)}
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    return result
