import argparse, hashlib, json, platform
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from src.config import LABELS, SEED
from src.data.make_dataset import load, groups
from src.features.build_features import normalized

def split(rows):
    g=groups(rows)
    # Select a fixed seed split with class coverage; never select on model performance.
    for seed in range(SEED,SEED+1000):
        tr,te=next(GroupShuffleSplit(n_splits=1,test_size=.3,random_state=seed).split(rows,groups=g))
        if set(rows[i]['label'] for i in tr)==set(rows[i]['label'] for i in te)==set(r['label'] for r in rows): return tr,te,g,seed
    raise ValueError('Cannot create grouped split with every class in both partitions; add independent problems')

def train(path, output, backend='linear', epochs=20):
    rows=load(path);tr,te,g,seed=split(rows)
    x=[normalized(r['code']) for r in rows];y=[r['label'] for r in rows]
    active_labels=[label for label in LABELS if label in set(y)]
    from collections import Counter
    majority=Counter(y[i] for i in tr).most_common(1)[0][0]
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if backend=='linear':
        model=Pipeline([('tfidf',TfidfVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,ngram_range=(1,3))),('classifier',LogisticRegression(max_iter=2000,C=4,random_state=SEED))])
        model.fit([x[i] for i in tr],[y[i] for i in tr]);pred=model.predict([x[i] for i in te]);joblib.dump(model,output/'model.joblib')
    else:
        from src.models.neural import fit
        pred=fit(x,y,tr,te,output,epochs)
    report={'backend':backend,'target':rows[0].get('target','time_complexity'),'majority_accuracy':sum(y[i]==majority for i in te)/len(te),'majority_class':majority,'epochs':epochs if backend=='cnn' else None,'seed':SEED,'dataset_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'rows':len(rows),'train_rows':len(tr),'test_rows':len(te),'train_groups':len(set(g[i] for i in tr)),'test_groups':len(set(g[i] for i in te)),'group_overlap':len(set(g[i] for i in tr)&set(g[i] for i in te)),'split_seed':seed,'accuracy':accuracy_score([y[i] for i in te],pred),'macro_f1':f1_score([y[i] for i in te],pred,labels=active_labels,average='macro',zero_division=0),'classification_report':classification_report([y[i] for i in te],pred,labels=active_labels,output_dict=True,zero_division=0),'confusion_matrix':confusion_matrix([y[i] for i in te],pred,labels=active_labels).tolist(),'label_order':active_labels,'test_problem_ids':[rows[i]['problem_id'] for i in te],'synthetic':all(r.get('source','').startswith('original synthetic') for r in rows),'python':platform.python_version()}
    (output/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/processed/tasty-time_complexity.jsonl'));p.add_argument('--output',default='models/tasty-linear');p.add_argument('--backend',choices=['linear','cnn'],default='linear');p.add_argument('--epochs',type=int,default=20);a=p.parse_args();print(json.dumps(train(a.data,a.output,a.backend,a.epochs),indent=2))
if __name__=='__main__':main()
