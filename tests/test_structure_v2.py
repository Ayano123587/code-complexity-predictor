import json
import joblib
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.features.code_v2 import canonical, structural
from src.models.structure_v2 import structure, extra, grouped_cv
from src.models.predict_model import predict


def test_canonical_names_constants_and_hidden_text():
    assert canonical('int x=2; x=x+1;') == canonical('int renamed=2; renamed=renamed+1;')
    assert canonical('int x=2;') != canonical('int x=3;')
    assert canonical('int x=x;') != canonical('int x=y;')
    assert 'quadratic' not in canonical('// quadratic O(n)\nint x=1;')
    assert 'exponential' not in canonical('puts("exponential");')


def test_loop_structure_and_direct_recursion():
    nested=structural('void f(){for(int i=0;i<n;i++)for(int j=0;j<n;j++)x++;}')
    sequential=structural('void f(){for(int i=0;i<n;i++)x++;for(int j=0;j<n;j++)x++;}')
    assert nested['max_loop_nesting']==2
    assert sequential['max_loop_nesting']==1
    assert structural('int f(int n){if(n<2)return n;return f(n-1)+f(n-2);}')['maximum_self_calls']==2


def test_grouped_calibration_does_not_mix_groups():
    x=np.arange(36);y=np.array(['a','b']*18);groups=np.repeat(np.arange(18),2)
    for tr,va in grouped_cv(x,y,groups):
        assert not set(groups[tr])&set(groups[va])
        assert set(y[tr])==set(y[va])=={'a','b'}


def test_raw_representation_checkpoint_roundtrip(tmp_path):
    x=['int f(){return 1;}','void f(){for(int i=0;i<n;i++)x++;}']*3
    y=['constant','linear']*3
    model=Pipeline([('features',structure()),('classifier',extra(1))]).fit(x,y)
    joblib.dump(model,tmp_path/'model.joblib')
    (tmp_path/'evaluation.json').write_text(json.dumps({'input_representation':'raw_cpp_v2','label_order':list(model.classes_)}))
    result=predict(x[1],tmp_path)
    np.testing.assert_allclose(list(result['scores'].values()),model.predict_proba([x[1]])[0])
    with pytest.raises(ValueError,match='text'):predict(None,tmp_path)
    with pytest.raises(ValueError,match='no tokens'):predict('// only comment',tmp_path)


def test_weighted_ensemble_aligns_probabilities():
    from src.models.structure_v2 import EqualEnsemble
    from sklearn.dummy import DummyClassifier
    model=EqualEnsemble([DummyClassifier(strategy='constant',constant='a'),DummyClassifier(strategy='constant',constant='b')],weights=[.25,.75]).fit(['x','y'],['a','b'])
    np.testing.assert_allclose(model.predict_proba(['z']),[[.25,.75]])
    assert model.predict(['z'])[0]=='b'


def test_cli_estimator_has_importable_pickle_identity():
    import subprocess, sys
    from src.models.structure_v2 import EqualEnsemble, DenseArray
    assert EqualEnsemble.__module__=='src.models.ensemble_v2'
    assert DenseArray.__module__=='src.models.ensemble_v2'
    subprocess.run([sys.executable,'-m','src.models.structure_v2','--help'],check=True,capture_output=True)


def test_seed_diagnostic_preserves_existing_results(tmp_path):
    from src.models.stability_v2 import run
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(ValueError,match='Preserve'):run(tmp_path,'missing-data')
