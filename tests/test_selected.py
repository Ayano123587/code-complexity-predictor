import pytest
from unittest.mock import Mock
from src.models import selected


def test_train_rejects_existing_before_reading(tmp_path, monkeypatch):
    reader = Mock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(selected, 'cohort', reader)
    with pytest.raises(ValueError, match='Fresh'):
        selected.train('missing', tmp_path)
    reader.assert_not_called()


def test_evaluate_requires_ack_before_loading(tmp_path, monkeypatch):
    reader = Mock(side_effect=AssertionError('must not read'))
    monkeypatch.setattr(selected, 'cohort', reader)
    with pytest.raises(ValueError, match='acknowledgement'):
        selected.evaluate('missing', 'missing', tmp_path/'report.json')
    reader.assert_not_called()


def test_evaluate_preserves_existing_report(tmp_path):
    report = tmp_path/'report.json'
    report.write_text('preserve', encoding='utf-8')
    with pytest.raises(ValueError, match='Fresh'):
        selected.evaluate('missing', 'missing', report, acknowledge_observed=True)
    assert report.read_text() == 'preserve'


def test_fixed_selected_parameters():
    model = selected.model()
    assert model.weights == [.25, .75]
    assert model.estimators[0].named_steps['classifier'].C == 16
    assert model.estimators[1].named_steps['classifier'].min_samples_leaf == 3
    assert model.estimators[1].named_steps['classifier'].n_estimators == 400


def test_validation_fit_then_final_development_refit(tmp_path, monkeypatch):
    import numpy as np
    x=np.asarray(['a','b','c','d','heldout-one','heldout-two'])
    y=np.asarray(['linear']*6)
    split={'partitions':{'train':[0,1],'validation':[2,3],'holdout':[4,5]},'label_order':['linear']}
    monkeypatch.setattr(selected,'cohort',lambda *_:(x,y,split))
    fits=[]
    class Estimator:
        def fit(self,features,labels):
            fits.append(list(features))
            return self
    monkeypatch.setattr(selected,'model',Estimator)
    monkeypatch.setattr(selected,'scores',lambda *_:{'macro_f1':.5})
    monkeypatch.setattr(selected,'sha',lambda _: 'bound')
    monkeypatch.setattr(selected.joblib,'dump',lambda _,stream:stream.write(b'checkpoint'))
    report=selected.train('fixture',tmp_path/'fresh')
    assert fits==[['a','b'],['a','b','c','d']]
    assert report['final_training_rows']==4 and report['test_evaluated'] is False


def test_missing_checkpoint_binding_rejected_before_model_load(tmp_path,monkeypatch):
    import numpy as np
    modeldir=tmp_path/'model';modeldir.mkdir()
    (modeldir/'evaluation.json').write_text('{}',encoding='utf-8')
    monkeypatch.setattr(selected,'cohort',lambda *_:(np.asarray(['a']),np.asarray(['linear']),{'partitions':{'holdout':[0]},'label_order':['linear']}))
    monkeypatch.setattr(selected,'sha',lambda _:'bound')
    load=Mock(side_effect=AssertionError('must not deserialize'))
    monkeypatch.setattr(selected.joblib,'load',load)
    with pytest.raises(ValueError,match='provenance'):
        selected.evaluate('fixture',modeldir,tmp_path/'report.json',acknowledge_observed=True)
    load.assert_not_called()
