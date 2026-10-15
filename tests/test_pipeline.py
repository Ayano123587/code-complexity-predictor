import json
import pytest
from src.features.build_features import normalized,tokenize
from src.data.make_dataset import synthetic,groups,validate
from src.models.train_model import split,train
from src.models.predict_model import predict

def test_comments_strings_and_renaming():
    assert normalized('int x=42; // while(n)')==normalized('int y=9;')
    assert 'for' not in tokenize('"for while" /* for */')
    assert 'STR' in tokenize('"hello // still string"')

def test_duplicate_and_problem_union():
    rows=[{'code':'int x=1;', 'problem_id':'a'},{'code':'int y=2;', 'problem_id':'b'},{'code':'while(n)n/=2;', 'problem_id':'b'}]
    assert len(set(groups(rows)))==1

def test_split_no_overlap():
    rows=synthetic();tr,te,g,_=split(rows)
    assert not set(g[i] for i in tr)&set(g[i] for i in te)
    assert set(tr).isdisjoint(te)
    assert len(tr)+len(te)==len(rows)

def test_validation():
    with pytest.raises(ValueError):validate([])
    with pytest.raises(ValueError):validate([{'code':'x','label':'O(n)','problem_id':'x'}])

def test_actual_training_roundtrip(tmp_path):
    path=tmp_path/'data.jsonl';path.write_text('\n'.join(json.dumps(x) for x in synthetic()))
    result=train(path,tmp_path/'model');assert result['group_overlap']==0
    assert 0<=result['macro_f1']<=1
    pred=predict('int f(int n,int*a){for(int i=0;i<n;i++)a[i]++;return 0;}',tmp_path/'model')
    assert abs(sum(pred['scores'].values())-1)<1e-6
    assert (tmp_path/'model'/'evaluation.json').exists()


def test_fetch_requires_integrity_and_transport(tmp_path):
    from src.data.fetch_dataset import fetch
    with pytest.raises(ValueError, match='HTTPS'):
        fetch('http://example.com/data', 'a'*64, 'MIT', tmp_path/'data')
    with pytest.raises(ValueError, match='SHA256'):
        fetch('https://example.com/data', 'short', 'MIT', tmp_path/'data')


def test_fetch_verified_content(monkeypatch,tmp_path):
    import hashlib
    from src.data.fetch_dataset import fetch
    content=json.dumps(synthetic()[0]).encode()
    class Response:
        url='https://example.com/corpus.jsonl'
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return content
    monkeypatch.setattr('urllib.request.urlopen',lambda *args,**kwargs:Response())
    path=tmp_path/'corpus.jsonl'
    fetch(Response.url, hashlib.sha256(content).hexdigest(), 'MIT', path)
    assert path.read_bytes()==content
    with pytest.raises(ValueError,match='Checksum'):
        fetch(Response.url,'0'*64,'MIT',path)


def test_tasty_strips_complexity_comments_and_excludes_np():
    import csv,io
    from src.data.tasty import prepare
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['code','time_complexity','space_complexity']);writer.writeheader()
    writer.writerow({'code':'// Time O(n)\nint x=0; /* quadratic */', 'time_complexity':'linear','space_complexity':'constant'})
    writer.writerow({'code':'int n;', 'time_complexity':'np','space_complexity':'np'})
    rows,report=prepare(stream.getvalue().encode())
    assert len(rows)==1 and report['excluded_labels']=={'np':1}
    assert 'Time' not in rows[0]['code'] and 'quadratic' not in rows[0]['code']
    assert 'int' in rows[0]['code']


def test_tasty_source_family_grouping():
    import csv,io
    from src.data.tasty import prepare
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=['code','time_complexity','space_complexity']);writer.writeheader()
    for body in ['int a=0;', 'while(n)n/=2;']:
        writer.writerow({'code':'// Segment tree range query implementation\n'+body,'time_complexity':'linear','space_complexity':'constant'})
    rows,_=prepare(stream.getvalue().encode())
    assert rows[0]['problem_id']==rows[1]['problem_id']
