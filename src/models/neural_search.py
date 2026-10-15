"""Class-weight/context ablation on frozen training/validation partitions."""
import argparse,copy,json,hashlib
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from src.config import LABELS
from src.data.make_dataset import load
from src.features.build_features import normalized
from src.models.neural import TokenCNN

def encode(texts,vocab,maxlen):return torch.tensor([[vocab.get(t,1) for t in text.split()[:maxlen]]+[0]*(maxlen-min(len(text.split()),maxlen)) for text in texts])
def run(data,output,epochs=20):
    torch.set_num_threads(2);output=Path(output)
    if (output/'neural-selection.json').exists() or (output/'evaluation.json').exists():raise ValueError('Study already evaluated or neural selection frozen')
    split=json.loads((output/'split.json').read_text())
    if hashlib.sha256(Path(data).read_bytes()).hexdigest()!=split['data_sha256']:raise ValueError('Frozen data changed')
    rows=load(data);x=[normalized(r['code']) for r in rows];tr=split['partitions']['train'];va=split['partitions']['validation'];y=torch.tensor([LABELS.index(r['label']) for r in rows]);active=[LABELS.index(label) for label in split['label_order']]
    vocab={t:i+2 for i,t in enumerate(sorted(set(' '.join(x[i] for i in tr).split())))};counts=torch.bincount(y[tr],minlength=len(LABELS)).float();runs=[]
    for maxlen in [512,1024]:
        inputs=encode(x,vocab,maxlen)
        for mode in ['sqrt_inverse','inverse']:
            torch.manual_seed(42);model=TokenCNN(len(vocab)+2);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
            weights=1/counts.clamp(min=1).pow(.5 if mode=='sqrt_inverse' else 1);weights/=weights.mean();history=[];best=(-1.,-1.);best_state=None;best_epoch=None
            generator=torch.Generator().manual_seed(42)
            for epoch in range(1,epochs+1):
                model.train();order=torch.tensor(tr)[torch.randperm(len(tr),generator=generator)];loss_sum=0.
                for batch in order.split(32):
                    optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(inputs[batch]),y[batch],weight=weights);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*len(batch)
                model.eval()
                with torch.no_grad():pred=torch.cat([model(inputs[batch]).argmax(1) for batch in torch.tensor(va).split(32)]).numpy()
                scores=(f1_score(y[va],pred,labels=active,average='macro',zero_division=0),accuracy_score(y[va],pred));history.append({'epoch':epoch,'train_loss':loss_sum/len(tr),'validation_macro_f1':float(scores[0]),'validation_accuracy':float(scores[1])})
                if scores>best:best=scores;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch
            name=f'cnn-{maxlen}-{mode}';folder=output/name;folder.mkdir(exist_ok=True);torch.save(best_state,folder/'weights.pt');(folder/'vocab.json').write_text(json.dumps(vocab));report={'name':name,'context_tokens':maxlen,'class_weight':mode,'best_epoch':best_epoch,'validation_macro_f1':float(best[0]),'validation_accuracy':float(best[1]),'history':history,'train_rows':len(tr),'validation_rows':len(va),'seed':42,'model':'token embedding48/conv64/maxpool; AdamW.002; batch32; clipgrad1; weightdecay.01'};(folder/'validation.json').write_text(json.dumps(report,indent=2));runs.append(report);print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    result={'selection':'validation macro F1 then accuracy; per-epoch checkpoint selection uses validation only','split_sha256':hashlib.sha256((output/'split.json').read_bytes()).hexdigest(),'runs':runs,'winner':max(runs,key=lambda r:(r['validation_macro_f1'],r['validation_accuracy']))['name'],'holdout_evaluated':False};(output/'neural-selection.json').write_text(json.dumps(result,indent=2))
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/processed/tasty-time_complexity.jsonl');p.add_argument('--output',default='models/tasty-optimized');p.add_argument('--epochs',type=int,default=20);a=p.parse_args();run(a.data,a.output,a.epochs)
if __name__=='__main__':main()


def refit_selected(x,y,tr,te,spec,output):
    """Refit a validation-selected CNN for a fixed epoch count on train+validation."""
    torch.set_num_threads(2);torch.manual_seed(42)
    vocab={t:i+2 for i,t in enumerate(sorted(set(' '.join(x[i] for i in tr).split())))}
    inputs=encode(x,vocab,spec['context_tokens']);targets=torch.tensor([LABELS.index(str(label)) for label in y]);counts=torch.bincount(targets[tr],minlength=len(LABELS)).float();weights=1/counts.clamp(min=1).pow(.5 if spec['class_weight']=='sqrt_inverse' else 1);weights/=weights.mean()
    model=TokenCNN(len(vocab)+2);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01);generator=torch.Generator().manual_seed(42)
    for _ in range(spec['best_epoch']):
        model.train();order=torch.tensor(tr)[torch.randperm(len(tr),generator=generator)]
        for batch in order.split(32):
            optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(inputs[batch]),targets[batch],weight=weights);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval()
    with torch.no_grad():pred=torch.cat([model(inputs[b]).argmax(1) for b in torch.tensor(te).split(32)]).tolist()
    torch.save(model.state_dict(),output/'weights.pt');(output/'vocab.json').write_text(json.dumps(vocab))
    return [LABELS[i] for i in pred]
