"""Import labeled JSONL or build an explicitly synthetic pipeline benchmark."""
import argparse, hashlib, json
from pathlib import Path
from rapidfuzz.fuzz import ratio
from src.config import LABELS
from src.features.build_features import normalized

def validate(rows):
    if not rows: raise ValueError('Dataset is empty')
    for row in rows:
        if not isinstance(row.get('code'), str) or not row['code'].strip(): raise ValueError('Nonempty code required')
        if row.get('label') not in LABELS: raise ValueError('Unknown label')
        if not row.get('problem_id'): raise ValueError('problem_id required for grouped evaluation')
    return rows

def load(path): return validate([json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()])

def groups(rows, threshold=.88):
    # Union by problem OR normalized near-duplicate, transitively. O(n^2), for research corpora.
    parents=list(range(len(rows)))
    def root(i):
        while parents[i]!=i: parents[i]=parents[parents[i]];i=parents[i]
        return i
    texts=[normalized(row['code']) for row in rows]
    for i in range(len(rows)):
        for j in range(i):
            if rows[i]['problem_id']==rows[j]['problem_id'] or ratio(texts[i],texts[j],score_cutoff=threshold*100)>=threshold*100:
                parents[root(i)]=root(j)
    return [root(i) for i in range(len(rows))]

def synthetic():
    # Distinct algorithm families; labels assume n is the single input size,
    # unit-cost arithmetic, and worst-case execution. These are teaching examples.
    bodies={
    'constant':['return a[0];','return n*(n+1)/2;','int x=n; x=x^7; return x;','return a[n-1]+a[0];'],
    'logn':['while(n>1){n/=2;} return n;','int k=1; while(k<n){k*=2;} return k;','int l=0,r=n; while(l<r){int m=(l+r)/2; if(a[m]<7)l=m+1;else r=m;}return l;','int s=0; for(int i=n;i>0;i>>=1){s+=i;}return s;'],
    'linear':['int s=0;for(int i=0;i<n;i++){s+=a[i];}return s;','int s=0;while(n--){s+=a[n];}return s;','int s=a[0];for(int i=1;i<n;i++){if(a[i]>s)s=a[i];}return s;','int i=0,j=n-1;while(i<j){int t=a[i];a[i++]=a[j];a[j--]=t;}return a[0];'],
    'nlogn':['sort(a,a+n);return a[0];','int s=0;for(int i=0;i<n;i++){int j=n;while(j>1){j/=2;s++;}}return s;','int s=0;for(int i=1;i<n;i*=2){for(int j=0;j<n;j++){s+=a[j];}}return s;','int s=0;for(int i=0;i<n;i++){s+=lower_bound(a,a+n,i)-a;}return s;'],
    'quadratic':['int s=0;for(int i=0;i<n;i++){for(int j=0;j<n;j++){s+=a[i]*a[j];}}return s;','int s=0;for(int i=0;i<n;i++){for(int j=i;j<n;j++){s++;}}return s;','for(int i=0;i<n;i++){for(int j=0;j<n-i-1;j++){if(a[j]>a[j+1]){int t=a[j];a[j]=a[j+1];a[j+1]=t;}}}return a[0];','int s=0,i=n;while(i--){int j=n;while(j--){s+=j;}}return s;'],
    'cubic':['int s=0;for(int i=0;i<n;i++){for(int j=0;j<n;j++){for(int k=0;k<n;k++){s+=a[k];}}}return s;','int s=0;for(int i=0;i<n;i++){for(int j=i;j<n;j++){for(int k=j;k<n;k++){s++;}}}return s;','int s=0,i=n;while(i--){int j=n;while(j--){int k=n;while(k--){s++;}}}return s;','int s=0;for(int i=n;i>0;i--){for(int j=n;j>0;j--){for(int k=n;k>0;k--){s+=i+j+k;}}}return s;'],
    'exponential':['if(n<=1)return 1;return f(n-1,a)+f(n-2,a);','if(n==0)return 1;return f(n-1,a)+f(n-1,a);','int s=0;for(long long mask=0;mask<(1LL<<n);mask++){s++;}return s;','if(n<1)return 0;int x=f(n-1,a);int y=f(n-1,a);int z=f(n-1,a);return x+y+z;']}
    return [{'code':'long long f(int n,int* a){'+body+'}', 'label':label,'problem_id':f'synthetic-{label}-{i}','source':'original synthetic teaching example; NOT public benchmark'} for label,bs in bodies.items() for i,body in enumerate(bs)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path);p.add_argument('--output',type=Path,default=Path('data/processed/smoke.jsonl'));a=p.parse_args()
    rows=load(a.input) if a.input else synthetic();a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'synthetic':not bool(a.input)}))
if __name__=='__main__':main()
