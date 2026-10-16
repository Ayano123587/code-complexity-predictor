import joblib
import numpy as np
import pytest
from src.models.family_v4 import sample_weights,FamilyEnsemble


def test_equal_family_mass_and_class_mass_are_training_local():
    labels=['a','a','b','b'];groups=['one','one','two','three']
    w=sample_weights(labels,groups)
    assert w[:2].sum()==pytest.approx(w[2]) and w[2]==pytest.approx(w[3])
    assert w.mean()==pytest.approx(1.)
    balanced=sample_weights(labels,groups,True)
    assert balanced[:2].sum()==pytest.approx(balanced[2:].sum())
    duplicate=sample_weights(labels+['a','a'],groups+['one','one'])
    assert duplicate[[0,1,4,5]].sum()==pytest.approx(duplicate[2])


def test_weighted_model_persistence_and_finite_inference(tmp_path):
    x=np.array(['int f(){return 1;}','void f(){for(int i=0;i<n;i++)x++;}']*6)
    y=np.array(['constant','linear']*6);groups=np.repeat(np.arange(6),2)
    model=FamilyEnsemble('family_class').fit(x,y,groups)
    joblib.dump(model,tmp_path/'model.joblib');loaded=joblib.load(tmp_path/'model.joblib')
    np.testing.assert_array_equal(model.predict_proba(x),loaded.predict_proba(x))
    p=loaded.predict_proba(x);assert np.isfinite(p).all();np.testing.assert_allclose(p.sum(axis=1),1.)


def test_weighting_rejects_misaligned_groups():
    with pytest.raises(ValueError,match='Aligned'):sample_weights(['a','b'],['one'])
    with pytest.raises(ValueError,match='Aligned'):sample_weights([],[])
