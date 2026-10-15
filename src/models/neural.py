"""Trainable token embedding + convolution + max pooling neural baseline."""
import json
from collections import Counter
import numpy as np
import torch
from torch import nn
from src.config import LABELS, SEED

class TokenCNN(nn.Module):
    def __init__(self,vocab_size):
        super().__init__();self.embedding=nn.Embedding(vocab_size,48,padding_idx=0);self.conv=nn.Conv1d(48,64,3,padding=1);self.head=nn.Linear(64,len(LABELS))
    def forward(self,x):
        h=torch.relu(self.conv(self.embedding(x).transpose(1,2)))
        h=h.masked_fill((x==0).unsqueeze(1),-1e9)
        return self.head(h.max(dim=2).values)

def encode(texts,vocab): return torch.tensor([[vocab.get(t,1) for t in text.split()[:512]]+[0]*(512-min(len(text.split()),512)) for text in texts])

def fit(x,y,tr,te,output,epochs):
    if epochs<1:raise ValueError('epochs must be positive')
    torch.manual_seed(SEED);torch.set_num_threads(2)
    vocab={t:i+2 for i,t in enumerate(sorted(set(' '.join(x[i] for i in tr).split())))}
    model=TokenCNN(len(vocab)+2);opt=torch.optim.Adam(model.parameters(),lr=.003)
    inputs=encode([x[i] for i in tr],vocab);target=torch.tensor([LABELS.index(y[i]) for i in tr]);losses=[]
    for _ in range(epochs):
        model.train();opt.zero_grad();loss=nn.functional.cross_entropy(model(inputs),target);loss.backward();opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():pred=model(encode([x[i] for i in te],vocab)).argmax(1).tolist()
    torch.save(model.state_dict(),output/'weights.pt');(output/'vocab.json').write_text(json.dumps(vocab));(output/'loss.json').write_text(json.dumps(losses))
    return [LABELS[i] for i in pred]
