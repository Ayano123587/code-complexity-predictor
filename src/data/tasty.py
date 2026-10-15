"""Pinned TASTY C++ ingestion. Never republishes downloaded snippets."""
import argparse,csv,hashlib,io,json,re,urllib.request
from collections import Counter
from pathlib import Path
from src.config import LABELS
from src.features.build_features import TOKEN
REVISION='8c273d0476910f9b08b8cf01c2e31fd843ee2d7f'
SHA256='ff746b37b2d7adf8e7436c772732c62a2385b18b4e78b5ad1ef518e96f70b61c'
URL=f'https://huggingface.co/datasets/Banana-Leopard/TASTY/resolve/{REVISION}/cpp_data_clean.csv'

def strip_comments(code):
    return ' '.join(t for t in TOKEN.findall(code) if not t.startswith(('//','/*')))

def prepare(content,target='time_complexity'):
    rows=list(csv.DictReader(io.StringIO(content.decode('utf-8'))));clean=[];excluded=Counter()
    for i,row in enumerate(rows):
        label=row[target].strip()
        if label not in LABELS:excluded[label]+=1;continue
        code=row['code']
        # No original problem URLs are provided. Repeated descriptive leading
        # comment signatures are a best-effort additional family grouping.
        leading=re.match(r'\s*(//[^\n]*|/\*.*?\*/)',code,re.S)
        title=re.sub(r'\W+',' ',leading.group(1).lower()).strip() if leading else ''
        informative=len(title)>25 and not re.match(r'(c program|c code|cpp program|cpp code|this code|this program)',title)
        problem='title-'+hashlib.sha256(title.encode()).hexdigest()[:20] if informative else f'tasty-row-{i}'
        clean.append({'code':strip_comments(code),'label':label,'problem_id':problem,'source':f'TASTY/{REVISION}/cpp_data_clean.csv#{i}','original_row':i,'target':target})
    return clean,{'raw_rows':len(rows),'retained_rows':len(clean),'excluded_labels':dict(excluded),'class_counts':dict(Counter(r['label'] for r in clean)),'target':target,'problem_id_limitation':'No source URLs/problem IDs supplied; descriptive leading-comment signature when available, otherwise row ID, plus normalized near-duplicate transitive grouping at train time.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['time_complexity','space_complexity'],default='time_complexity');p.add_argument('--output',type=Path);a=p.parse_args()
    with urllib.request.urlopen(URL,timeout=60) as f:content=f.read()
    if hashlib.sha256(content).hexdigest()!=SHA256:raise ValueError('Pinned TASTY checksum mismatch')
    raw=Path('data/external/tasty.csv');raw.parent.mkdir(parents=True,exist_ok=True);raw.write_bytes(content)
    rows,report=prepare(content,a.target);out=a.output or Path(f'data/processed/tasty-{a.target}.jsonl');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    report.update({'url':URL,'revision':REVISION,'sha256':SHA256,'declared_license':'CC-BY-3.0 (dataset card metadata; scraped third-party provenance remains a caveat)','attribution':'Kaushik Moudgalya, Ankit Ramakrishnan, Vamsikrishna Chemudupati, Xing Han Lu; TASTY, arXiv:2305.05379; Banana-Leopard/TASTY; underlying GeeksforGeeks snippets','output_sha256':hashlib.sha256(out.read_bytes()).hexdigest()})
    manifest=Path('models')/f'tasty-{a.target}-provenance.json';manifest.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
