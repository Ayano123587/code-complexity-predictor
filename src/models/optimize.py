"""Train/validation search plus one preserved retrospective group holdout."""
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from src.data.make_dataset import load,groups
from src.features.build_features import normalized
from src.features.structure import StructureFeatures

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def freeze(data,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'split.json').exists():raise ValueError('Split already frozen')
    rows=load(data);g=np.asarray(groups(rows));y=np.asarray([r['label'] for r in rows]);fold=np.empty(len(rows),dtype=int)
    for i,(_,te) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261016).split(rows,y,g)):fold[te]=i
    partitions={'train':np.flatnonzero(fold>=2).tolist(),'validation':np.flatnonzero(fold==1).tolist(),'holdout':np.flatnonzero(fold==0).tolist()}
    expected=set(y)
    for key,indices in partitions.items():
        if set(y[indices])!=expected:raise ValueError(f'{key} lacks observed class; add independent data')
    manifest={'data_sha256':sha(data),'seed':20261016,'partitions':partitions,'groups':g.tolist(),'rows':len(rows),'limitation':'Retrospective holdout on a previously explored corpus; not an untouched external dataset','label_order':sorted(expected)}
    (output/'split.json').write_text(json.dumps(manifest,indent=2));return manifest

def build(spec):
    word=TfidfVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,ngram_range=(1,3),sublinear_tf=True,min_df=2)
    char=TfidfVectorizer(analyzer='char',ngram_range=(3,5),sublinear_tf=True,min_df=2,max_features=100000)
    structure=Pipeline([('counts',StructureFeatures()),('vectorize',DictVectorizer()),('scale',MaxAbsScaler())])
    choices={'word':word,'char':char,'wordchar':FeatureUnion([('word',word),('char',char)]),'wordstructure':FeatureUnion([('word',word),('structure',structure)])}
    return Pipeline([('features',choices[spec['features']]),('classifier',LogisticRegression(C=spec['C'],class_weight=spec['class_weight'],max_iter=3000,random_state=42))])

def score(y,pred,labels):return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0))}

def search(data,output):
    output=Path(output)
    if (output/'selection.json').exists() or (output/'evaluation.json').exists():raise ValueError('Selection already frozen; create a new declared study rather than overwriting')
    manifest=json.loads((output/'split.json').read_text())
    if sha(data)!=manifest['data_sha256']:raise ValueError('Frozen data changed')
    rows=load(data);x=np.asarray([normalized(r['code']) for r in rows]);y=np.asarray([r['label'] for r in rows]);tr=manifest['partitions']['train'];va=manifest['partitions']['validation'];runs=[]
    for features in ['word','char','wordchar','wordstructure']:
        for weight in [None,'balanced']:
            for c in [1.,4.,16.]:
                spec={'features':features,'class_weight':weight,'C':c};model=build(spec);model.fit(x[tr],y[tr]);result={'spec':spec,'validation':score(y[va],model.predict(x[va]),manifest['label_order'])};runs.append(result);print(json.dumps(result),flush=True)
    winner=max(runs,key=lambda r:(r['validation']['macro_f1'],r['validation']['accuracy'],-['word','char','wordchar','wordstructure'].index(r['spec']['features'])))
    report={'split_sha256':sha(output/'split.json'),'selection_metric':'validation macro F1 then accuracy then simpler feature family','winner':winner,'runs':runs,'candidates':len(runs),'holdout_evaluated':False}
    (output/'selection.json').write_text(json.dumps(report,indent=2))

def evaluate_once(data,output):
    output=Path(output)
    if (output/'evaluation.json').exists():raise ValueError('Retrospective holdout already evaluated; preserve result')
    manifest=json.loads((output/'split.json').read_text());selection=json.loads((output/'selection.json').read_text())
    if sha(data)!=manifest['data_sha256'] or sha(output/'split.json')!=selection['split_sha256']:raise ValueError('Frozen protocol changed')
    neural=json.loads((output/'neural-selection.json').read_text())
    neural_best=max(neural['runs'],key=lambda r:(r['validation_macro_f1'],r['validation_accuracy']))
    neural_wins=(neural_best['validation_macro_f1'],neural_best['validation_accuracy'])>(selection['winner']['validation']['macro_f1'],selection['winner']['validation']['accuracy'])
    rows=load(data);x=np.asarray([normalized(r['code']) for r in rows]);y=np.asarray([r['label'] for r in rows]);tr=manifest['partitions']['train']+manifest['partitions']['validation'];te=manifest['partitions']['holdout'];labels=manifest['label_order']
    if neural_wins:
        from src.models.neural_search import refit_selected
        pred=refit_selected(x,y,tr,te,neural_best,output)
    else:
        model=build(selection['winner']['spec']);model.fit(x[tr],y[tr]);pred=model.predict(x[te]);joblib.dump(model,output/'model.joblib')
    from collections import Counter
    majority=Counter(y[tr]).most_common(1)[0][0];baseline=build({'features':'word','class_weight':None,'C':4.});baseline.fit(x[tr],y[tr])
    local_predictions=Path('data/interim')/f'{output.name}-holdout-predictions.json'
    local_predictions.parent.mkdir(parents=True,exist_ok=True)
    local_predictions.write_text(json.dumps([{'original_row':rows[i].get('original_row',i),'true_label':str(y[i]),'prediction':str(p),'group':manifest['groups'][i]} for i,p in zip(te,pred)],indent=2))
    report={'backend':'validation_selected_cnn' if neural_wins else 'validation_selected_logistic','context_tokens':neural_best['context_tokens'] if neural_wins else None,'target':rows[0].get('target','time_complexity'),'data_sha256':sha(data),'split_sha256':sha(output/'split.json'),'selection_sha256':sha(output/'selection.json'),'selected_spec':{'model':neural_best['name'],'epochs':neural_best['best_epoch'],'class_weight':neural_best['class_weight']} if neural_wins else selection['winner']['spec'],'neural_comparison':{'name':neural_best['name'],'validation_macro_f1':neural_best['validation_macro_f1'],'validation_accuracy':neural_best['validation_accuracy'],'selected':neural_wins},'validation_scores':{'accuracy':neural_best['validation_accuracy'],'macro_f1':neural_best['validation_macro_f1']} if neural_wins else selection['winner']['validation'],'holdout_scores':score(y[te],pred,labels),'baseline_holdout_scores':score(y[te],baseline.predict(x[te]),labels),'majority_holdout_scores':score(y[te],[majority]*len(te),labels),'train_validation_rows':len(tr),'holdout_rows':len(te),'group_overlap':len(set(manifest['groups'][i] for i in tr)&set(manifest['groups'][i] for i in te)),'local_prediction_rows':len(te),'local_prediction_table_sha256':sha(local_predictions),'classification_report':classification_report(y[te],pred,labels=labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix(y[te],pred,labels=labels).tolist(),'label_order':labels,'limitation':manifest['limitation']}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','search','evaluate']);p.add_argument('--data',default='data/processed/tasty-time_complexity.jsonl');p.add_argument('--output',default='models/tasty-optimized');a=p.parse_args()
    {'freeze':freeze,'search':search,'evaluate':evaluate_once}[a.action](a.data,a.output)
if __name__=='__main__':main()
