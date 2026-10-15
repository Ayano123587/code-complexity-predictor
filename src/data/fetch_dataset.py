"""Fetch a licensed JSONL corpus with integrity verification (no judge scraping)."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from urllib.parse import urlparse
from src.data.make_dataset import validate

def fetch(url,sha256,license_id,output):
    if urlparse(url).scheme!='https':raise ValueError('HTTPS required')
    if len(sha256)!=64:raise ValueError('Expected SHA256 required')
    if not license_id.strip():raise ValueError('Record the corpus license first')
    with urllib.request.urlopen(url,timeout=60) as response:
        if urlparse(response.url).scheme!='https':raise ValueError('Insecure redirect')
        content=response.read(100_000_001)
    if len(content)>100_000_000:raise ValueError('Corpus exceeds 100 MB adapter limit')
    if hashlib.sha256(content).hexdigest()!=sha256:raise ValueError('Checksum mismatch')
    rows=validate([json.loads(x) for x in content.decode('utf-8').splitlines() if x.strip()])
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(content)
    output.with_suffix('.provenance.json').write_text(json.dumps({'url':url,'sha256':sha256,'license':license_id,'rows':len(rows)},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('--sha256',required=True);p.add_argument('--license',required=True);p.add_argument('--output',type=Path,default=Path('data/external/labeled.jsonl'));a=p.parse_args();fetch(a.url,a.sha256,a.license,a.output)
if __name__=='__main__':main()
