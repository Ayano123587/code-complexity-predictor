"""Bounded validation-only code representation/classifier experiment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from src.data.make_dataset import load
from src.features.code_v2 import CodeText, CodeStructure
from src.models.ensemble_v2 import EqualEnsemble, DenseArray


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def word(representation='canonical'):
    return Pipeline([('code', CodeText(representation)), ('tfidf', TfidfVectorizer(
        tokenizer=str.split, token_pattern=None, lowercase=False, ngram_range=(1, 3),
        min_df=2, sublinear_tf=True, max_features=75000))])


def structure():
    return Pipeline([('code', CodeStructure()), ('vectorize', DictVectorizer()), ('scale', MaxAbsScaler())])


def extra(leaf):
    return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=leaf, max_features=.8,
                                class_weight='balanced', random_state=42, n_jobs=2)


def grouped_cv(x, y, groups):
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42).split(x, y, groups))
    expected = set(y)
    for train, val in folds:
        if set(y[train]) != expected or set(y[val]) != expected:
            raise ValueError('Insufficient independent class groups for calibration')
        if set(groups[train]) & set(groups[val]):
            raise ValueError('Calibration groups overlap')
    return folds


def candidates(x, y, groups, followup=False):
    if followup:
        base = Pipeline([('features', structure()), ('classifier', extra(1))])
        smooth = Pipeline([('features', structure()), ('classifier', extra(3))])
        models = [('structure_extra_leaf1', base)]
        for leaf in [2, 5]:
            models.append((f'structure_extra_leaf{leaf}', Pipeline([('features', structure()), ('classifier', extra(leaf))])))
        unweighted = clone(base).set_params(classifier__class_weight=None)
        models.append(('structure_extra_unweighted', unweighted))
        models.append(('structure_calibrated_sigmoid', CalibratedClassifierCV(base, cv=grouped_cv(x,y,groups), method='sigmoid', n_jobs=1)))
        legacy = Pipeline([('features', word('legacy')), ('classifier', LogisticRegression(C=16,class_weight='balanced',max_iter=3000,random_state=42))])
        models.append(('legacy_structure_equal', EqualEnsemble([legacy, smooth])))
        if followup == 'weights':
            return [models[-1]] + [(f'legacy_structure_weight{int(w*100)}', EqualEnsemble([legacy, smooth], weights=[w,1-w])) for w in [.25,.75]]
        return models
    logistic = lambda c: LogisticRegression(C=c, class_weight='balanced', max_iter=3000, random_state=42)
    canonical = Pipeline([('features', word()), ('classifier', logistic(16))])
    calibrated = CalibratedClassifierCV(Pipeline([('features', word()), ('classifier',
        LinearSVC(C=1., class_weight='balanced', random_state=42, max_iter=5000))]),
        cv=grouped_cv(x, y, groups), method='sigmoid', n_jobs=1)
    structural = Pipeline([('features', structure()), ('classifier', extra(3))])
    char = Pipeline([('code', CodeText()), ('tfidf', TfidfVectorizer(
        analyzer='char', ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=75000))])
    latent = lambda n: Pipeline([('text', word()), ('svd', TruncatedSVD(n_components=n, random_state=42))])
    return [
        ('legacy_balanced_C16', Pipeline([('features', word('legacy')), ('classifier', logistic(16))])),
        ('legacy_unweighted_C4', Pipeline([('features', word('legacy')), ('classifier', LogisticRegression(C=4, max_iter=3000, random_state=42))])),
        ('canonical_balanced_C4', Pipeline([('features', word()), ('classifier', logistic(4))])),
        ('canonical_balanced_C16', canonical),
        ('canonical_wordchar_C16', Pipeline([('features', FeatureUnion([('word', word()), ('char', char)])), ('classifier', logistic(16))])),
        ('canonical_calibrated_SVC', calibrated),
        ('structure_extra_leaf1', Pipeline([('features', structure()), ('classifier', extra(1))])),
        ('structure_extra_leaf3', structural),
        ('latent128_structure_extra', Pipeline([('features', FeatureUnion([('latent', latent(128)), ('structure', structure())])), ('classifier', extra(2))])),
        ('latent64_structure_hist', Pipeline([('features', FeatureUnion([('latent', latent(64)), ('structure', structure())])), ('dense', DenseArray()),
            ('classifier', HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=15, l2_regularization=10, class_weight='balanced', random_state=42))])),
        ('fixed_equal_ensemble', EqualEnsemble([canonical, calibrated, structural]))]


def scores(model, x, y, labels):
    p = model.predict_proba(x)
    p = p[:, [list(model.classes_).index(label) for label in labels]]
    pred = np.asarray(labels)[p.argmax(axis=1)]
    gold = np.eye(len(labels))[[labels.index(label) for label in y]]
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=labels, average='macro', zero_division=0)),
            'log_loss': float(log_loss(y, p, labels=labels)),
            'multiclass_brier': float(np.mean(np.sum((p-gold)**2, axis=1)))}


def run(data, output='models/structure-v2', release=False):
    output = Path(output)
    if release:return release_existing(data,output)
    if (output/'selection.json').exists():raise ValueError('Validation selection already exists; use a declared follow-up study')
    if (output/'results.json').exists():
        raise ValueError('Study results already exist; preserve them')
    plan = json.loads((output/'plan.json').read_text())
    split_path = Path('models/tasty-optimized/split.json')
    split = json.loads(split_path.read_text())
    if sha(data) != plan['dataset_sha256'] or sha(split_path) != plan['split_sha256']:
        raise ValueError('Dataset or frozen split changed')
    rows = load(data)
    x = np.asarray([row['code'] for row in rows])
    y = np.asarray([row['label'] for row in rows])
    groups = np.asarray(split['groups'])
    train, val, test = [np.asarray(split['partitions'][key]) for key in ['train', 'validation', 'holdout']]
    labels = split['label_order']
    started = time.monotonic()
    runs, failures, fitted = [], [], {}
    with threadpool_limits(limits=2):
        for name, model in candidates(x[train], y[train], groups[train], plan.get('followup',False)):
            if time.monotonic()-started > plan['maximum_cpu_wall_seconds']:
                failures.append({'candidate': name, 'reason': 'Declared wall budget exhausted before fit'})
                break
            begin = time.monotonic()
            try:
                model.fit(x[train], y[train])
                result = {'candidate': name, 'validation': scores(model, x[val], y[val], labels),
                          'runtime_seconds': time.monotonic()-begin}
                runs.append(result)
                fitted[name] = model
                print(json.dumps(result), flush=True)
            except Exception as error:
                failures.append({'candidate': name, 'error_type': type(error).__name__, 'reason': str(error)})
                print(json.dumps(failures[-1]), flush=True)
        if not runs:
            raise ValueError('Every candidate failed')
        winner = max(runs, key=lambda row: (row['validation']['macro_f1'], row['validation']['accuracy']))
        selection = {'plan_sha256': sha(output/'plan.json'), 'winner': winner,
                     'runs': runs, 'failures': failures, 'test_used_for_selection': False}
        (output/'selection.json').write_text(json.dumps(selection, indent=2))
        if not release:
            joblib.dump(fitted[winner['candidate']],output/'validation-model.joblib')
            local=Path('data/interim')/(output.name+'-validation-predictions.json')
            local.parent.mkdir(parents=True,exist_ok=True)
            prediction=fitted[winner['candidate']].predict(x[val])
            local.write_text(json.dumps([{'row':int(i),'true':str(y[i]),'prediction':str(p)} for i,p in zip(val,prediction)]))
            report={'selection':selection,'runtime_seconds':time.monotonic()-started,'test_evaluated':False}
            (output/'validation-results.json').write_text(json.dumps(report,indent=2))
            return report


def release_existing(data,output):
    output=Path(output)
    if (output/'results.json').exists():raise ValueError('Final descriptive comparison already exists')
    plan=json.loads((output/'plan.json').read_text());selection=json.loads((output/'selection.json').read_text())
    split_path=Path('models/tasty-optimized/split.json');split=json.loads(split_path.read_text())
    if sha(data)!=plan['dataset_sha256'] or sha(split_path)!=plan['split_sha256'] or sha(output/'plan.json')!=selection['plan_sha256']:raise ValueError('Frozen study changed')
    rows=load(data);x=np.asarray([row['code'] for row in rows]);y=np.asarray([row['label'] for row in rows]);groups=np.asarray(split['groups'])
    train,val,test=[np.asarray(split['partitions'][key]) for key in ['train','validation','holdout']];labels=split['label_order']
    winner=selection['winner'];started=time.monotonic()
    # Validation-selected policy is now serialized. Test is already observed;
    # this is a single descriptive comparison, never new independent evidence.
    combined = np.r_[train, val]
    templates = dict(candidates(x[combined], y[combined], groups[combined]))
    if plan.get('followup',False):
        templates.update(candidates(x[combined], y[combined], groups[combined], plan['followup']))
    comparison = list(dict.fromkeys([winner['candidate'], 'legacy_balanced_C16', 'legacy_unweighted_C4']))
    test_reports = {}
    for name in comparison:
        with threadpool_limits(limits=2):
            model = templates[name].fit(x[combined], y[combined])
        test_reports[name] = scores(model, x[test], y[test], labels)
        if name == winner['candidate']:
            joblib.dump(model, output/'model.joblib')
            pred = model.predict(x[test])
            (output/'evaluation.json').write_text(json.dumps({
                'target': 'time_complexity', 'input_representation': 'raw_cpp_v2',
                'label_order': labels, 'selected_candidate': name,
                'calibration': 'training-only grouped sigmoid' if 'calibrated' in name else 'uncalibrated model scores',
                'checkpoint_sha256': sha(output/'model.joblib'),
                'classification_report': classification_report(y[test], pred, labels=labels, output_dict=True, zero_division=0)}, indent=2))
            local = Path('data/interim')/(output.name+'-test-predictions.json')
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps([{'row': int(i), 'true': str(y[i]), 'prediction': str(p)} for i, p in zip(test, pred)]))
    report = {'selection': selection, 'compatible_observed_test_comparison': test_reports,
              'train_rows': len(train), 'validation_rows': len(val), 'test_rows': len(test),
              'group_overlap': len(set(groups[combined]) & set(groups[test])),
              'runtime_seconds': time.monotonic()-started,
              'environment': {'python': platform.python_version(), **{name: importlib.metadata.version(name)
                              for name in ['numpy', 'scikit-learn', 'rapidfuzz']}},
              'limitation': plan['split_status']}
    (output/'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/processed/tasty-time_complexity.jsonl')
    parser.add_argument('--output', default='models/structure-v2')
    parser.add_argument('--release',action='store_true',help='Perform the final descriptive observed-test comparison only after validation study is complete')
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.output,args.release), indent=2))


if __name__ == '__main__':
    main()
