"""Prespecified validation-only random-seed diagnostic; never changes deployment."""
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.data.make_dataset import load
from src.models.structure_v2 import word, structure, extra, sha, scores
from src.models.ensemble_v2 import EqualEnsemble


def run(output='models/structure-v2-stability', data='data/processed/tasty-time_complexity.jsonl'):
    output=Path(output)
    if (output/'results.json').exists():
        raise ValueError('Preserve existing seed diagnostic')
    plan=json.loads((output/'plan.json').read_text())
    split_path=Path('models/tasty-optimized/split.json')
    if sha(data)!=plan['dataset_sha256'] or sha(split_path)!=plan['split_sha256']:
        raise ValueError('Frozen corpus or split changed')
    if plan['deployment_changes_allowed'] or plan['holdout_evaluation_allowed']:
        raise ValueError('This diagnostic cannot alter deployment or score holdout')
    split=json.loads(split_path.read_text());rows=load(data)
    tr=split['partitions']['train'];va=split['partitions']['validation']
    x=np.array([r['code'] for r in rows]);y=np.array([r['label'] for r in rows])
    fitted=[];runs=[];started=time.monotonic()
    with threadpool_limits(limits=2):
        for seed in plan['seeds']:
            if time.monotonic()-started>plan['maximum_cpu_wall_seconds']:
                raise TimeoutError('Seed diagnostic wall budget exhausted')
            lexical=Pipeline([('features',word('legacy')),('classifier',LogisticRegression(C=16,class_weight='balanced',max_iter=3000,random_state=42))])
            structural=Pipeline([('features',structure()),('classifier',extra(3).set_params(random_state=seed))])
            model=EqualEnsemble([lexical,structural],weights=[.25,.75]).fit(x[tr],y[tr])
            fitted.append(model)
            runs.append({'seed':seed,'validation':scores(model,x[va],y[va],split['label_order'])})
        ensemble=EqualEnsemble([])
        ensemble.classes_=fitted[0].classes_;ensemble.fitted_=fitted
        averaged=scores(ensemble,x[va],y[va],split['label_order'])
        joblib.dump(ensemble,output/'validation-model.joblib')
    report={'plan_sha256':sha(output/'plan.json'),'runs':runs,'three_seed_average_validation':averaged,'seed_macro_f1_standard_deviation':float(np.std([r['validation']['macro_f1'] for r in runs])),'runtime_seconds':time.monotonic()-started,'deployment_changed':False,'holdout_evaluated':False}
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=2))
