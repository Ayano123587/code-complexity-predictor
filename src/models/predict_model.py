import argparse,json
from pathlib import Path
import joblib
from src.features.build_features import normalized
from src.config import LABELS

def predict(code,model_dir):
    p=Path(model_dir);text=normalized(code)
    metadata=json.loads((p/'evaluation.json').read_text()) if (p/'evaluation.json').exists() else {}
    if not text:raise ValueError('Source has no tokens')
    if (p/'model.joblib').exists():
        model=joblib.load(p/'model.joblib');prob=model.predict_proba([text])[0];labels=model.classes_
    else:
        import torch
        from src.models.neural import TokenCNN,encode
        vocab=json.loads((p/'vocab.json').read_text());model=TokenCNN(len(vocab)+2);model.load_state_dict(torch.load(p/'weights.pt',weights_only=True,map_location='cpu'));model.eval()
        with torch.no_grad():prob=torch.softmax(model(encode([text],vocab,metadata.get('context_tokens',512))),dim=1)[0].numpy()
        labels=LABELS
    scores={str(k):float(v) for k,v in zip(labels,prob)}
    target=metadata.get('target','time_complexity')
    if metadata.get('label_order'):
        scores={k:v for k,v in scores.items() if k in metadata['label_order']}
        total=sum(scores.values());scores={k:v/total for k,v in scores.items()}
    return {target+'_class':max(scores,key=scores.get),'scores':scores,'warning':'Statistical estimate, not a proof. Scores are uncalibrated. Input-size and execution assumptions require human review.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--model',default='models/tasty-linear');a=p.parse_args();print(json.dumps(predict(a.source.read_text(encoding='utf-8'),a.model),indent=2))
if __name__=='__main__':main()
