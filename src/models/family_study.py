"""Grouped OOF and validation confirmation for training-fold family weighting, never test-selected."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score,accuracy_score,classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from threadpoolctl import threadpool_limits
from src.features.mechanics_v3 import CodeMechanics
from src.models.structure_v2 import word,structure,extra,grouped_cv,scores,sha
from src.models.ensemble_v2 import EqualEnsemble
from sklearn.linear_model import LogisticRegression
from src.data.make_dataset import load


def candidate(name):
    from src.models.mechanics_v3 import candidate as base
    from src.models.family_v4 import FamilyEnsemble
    if name=='v2_baseline':return base(name)
    if name not in ['family_v2','family_class_v2','family_mechanics_leaf3']:raise ValueError('Unknown family candidate')
    return FamilyEnsemble(mode='family_class' if name=='family_class_v2' else 'family',mechanics=name=='family_mechanics_leaf3')


def trained(name,x,y,groups):
    model=candidate(name)
    return model.fit(x,y) if name=='v2_baseline' else model.fit(x,y,groups)


def gate(runs,minimum=.005):
    baseline=next(r for r in runs if r['candidate']=='v2_baseline')
    eligible=[r for r in runs if r['oof']['macro_f1']>=baseline['oof']['macro_f1']+minimum and r['validation']['macro_f1']>=baseline['validation']['macro_f1']+minimum]
    return max(eligible,key=lambda r:r['oof']['macro_f1'])['candidate'] if eligible else 'v2_baseline'


def run(output='models/family-weight-v4'):
    output=Path(output)
    if (output/'selection.json').exists():raise ValueError('Preserve completed grouped study')
    plan=json.loads((output/'plan.json').read_text(encoding='utf-8'))
    data=Path('data/processed/tasty-time_complexity.jsonl');split_path=Path('models/tasty-optimized/split.json')
    if sha(data)!=plan['dataset_sha256'] or sha(split_path)!=plan['split_sha256']:raise ValueError('Frozen corpus/split changed')
    split=json.loads(split_path.read_text());rows=load(data)
    x=np.asarray([r['code'] for r in rows]);y=np.asarray([r['label'] for r in rows]);groups=np.asarray(split['groups']);labels=split['label_order']
    tr=np.asarray(split['partitions']['train']);va=np.asarray(split['partitions']['validation'])
    folds=grouped_cv(x[tr],y[tr],groups[tr]);local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True)
    runs=[];failures=[];started=time.monotonic()
    with threadpool_limits(limits=2):
        for name in plan['candidates']:
            if time.monotonic()-started>plan['maximum_cpu_wall_seconds']:
                failures.append({'candidate':name,'reason':'Declared budget exhausted before fit'});break
            begin=time.monotonic()
            try:
                prediction=np.empty(len(tr),dtype=object);fold_reports=[]
                for train,val in folds:
                    model=trained(name,x[tr[train]],y[tr[train]],groups[tr[train]])
                    prediction[val]=model.predict(x[tr[val]])
                    fold_reports.append({'rows':len(val),'macro_f1':float(f1_score(y[tr[val]],prediction[val],labels=labels,average='macro',zero_division=0)),
                                         'group_overlap':len(set(groups[tr[train]])&set(groups[tr[val]]))})
                model=trained(name,x[tr],y[tr],groups[tr])
                result={'candidate':name,'oof':{'macro_f1':float(f1_score(y[tr],prediction,labels=labels,average='macro',zero_division=0)),
                                              'accuracy':float(accuracy_score(y[tr],prediction)),
                                              'classification_report':classification_report(y[tr],prediction,labels=labels,output_dict=True,zero_division=0)},
                        'folds':fold_reports,'validation':scores(model,x[va],y[va],labels),'runtime_seconds':time.monotonic()-begin}
                runs.append(result)
                joblib.dump(model,output/(name+'-validation.joblib'))
                (local/(name+'-oof-predictions.json')).write_text(json.dumps([{'row':int(i),'true':str(y[i]),'prediction':str(p)} for i,p in zip(tr,prediction)]))
                print(json.dumps(result),flush=True)
            except Exception as error:
                failures.append({'candidate':name,'error':type(error).__name__,'reason':str(error)})
    if not any(r['candidate']=='v2_baseline' for r in runs):raise ValueError('Baseline failed; comparison not valid')
    selected=gate(runs)
    report={'plan_sha256':sha(output/'plan.json'),'implementation_sha256':sha(__file__),
            'weighting_sha256':sha('src/models/family_v4.py'),'runs':runs,'failures':failures,'selected':selected,
            'deployment_changed':selected!='v2_baseline','test_evaluated':False,'runtime_seconds':time.monotonic()-started,
            'disclosure':plan['disclosure']}
    (output/'selection.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    if selected=='v2_baseline':
        report['retained_checkpoint_sha256']=sha('models/structure-v2-weights/model.joblib')
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


def release(output='models/family-weight-v4'):
    output=Path(output)
    if (output/'results.json').exists():raise ValueError('Final report already exists')
    selection=json.loads((output/'selection.json').read_text());plan=json.loads((output/'plan.json').read_text())
    if sha(output/'plan.json')!=selection['plan_sha256'] or sha('data/processed/tasty-time_complexity.jsonl')!=plan['dataset_sha256'] or sha('models/tasty-optimized/split.json')!=plan['split_sha256']:raise ValueError('Frozen study changed')
    selected=selection['selected']
    if selected=='v2_baseline':raise ValueError('Baseline retained; no new holdout scoring needed')
    split=json.loads(Path('models/tasty-optimized/split.json').read_text());rows=load('data/processed/tasty-time_complexity.jsonl')
    tr=split['partitions']['train']+split['partitions']['validation'];te=split['partitions']['holdout']
    x=np.asarray([r['code'] for r in rows]);y=np.asarray([r['label'] for r in rows]);labels=split['label_order']
    with threadpool_limits(limits=2):model=trained(selected,x[tr],y[tr],np.asarray(split['groups'])[tr])
    joblib.dump(model,output/'model.joblib');pred=model.predict(x[te])
    report={'selection_sha256':sha(output/'selection.json'),'selected':selected,'test_status':'Previously observed, retrospective descriptive comparison only',
            'observed_holdout':scores(model,x[te],y[te],labels),'checkpoint_sha256':sha(output/'model.joblib')}
    (output/'evaluation.json').write_text(json.dumps({'input_representation':'raw_cpp_v2','target':'time_complexity','label_order':labels,'classification_report':classification_report(y[te],pred,labels=labels,output_dict=True,zero_division=0)},indent=2))
    local=Path('data/interim')/output.name;local.mkdir(parents=True,exist_ok=True)
    (local/'holdout-predictions.json').write_text(json.dumps([{'row':int(i),'true':str(y[i]),'prediction':str(p)} for i,p in zip(te,pred)]))
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',default='models/family-weight-v4');parser.add_argument('--release',action='store_true');args=parser.parse_args()
    print(json.dumps(release(args.output) if args.release else run(args.output),indent=2))


if __name__=='__main__':main()
