import pytest
from src.features.mechanics_v3 import mechanics
from src.models.mechanics_v3 import gate,run


def test_name_invariant_multiplicative_and_constant_bounds():
    a='for(int i=1;i<n;i*=2){x++;}'
    b='for(int renamed=1;renamed<length;renamed*=2){value++;}'
    assert mechanics(a)==mechanics(b)
    assert mechanics(a)['mechanics:for_multiplicative_updates']==1
    assert mechanics('for(int i=0;i<8;i++)x++;')['mechanics:for_constant_bounds']==1
    assert mechanics(a)['mechanics:for_constant_bounds']==0


def test_dependent_bounds_with_and_without_braces_and_comments():
    for code in ['for(int i=0;i<n;i++){for(int j=0;j<i;j++)x++;}',
                 'for(int i=0;i<n;i++)for(int j=0;j<i;j++)x++;']:
        assert mechanics(code)['mechanics:for_dependent_nested_bounds']==1
    assert all(v==0 for v in mechanics('// for(i*=2) quadratic\n"while(n/=2)"').values())


def test_direct_recursive_argument_mechanics():
    assert mechanics('int f(int n){return f(n/2);}')['mechanics:recursive_division_calls']==1
    assert mechanics('int f(int n){return f(n-1)+f(n-2);}')['mechanics:recursive_subtraction_calls']==2
    assert mechanics('int f(int n){return g(n/2);}')['mechanics:recursive_division_calls']==0


def test_deployment_gate_requires_both_confirmations():
    runs=[{'candidate':'v2_baseline','oof':{'macro_f1':.5},'validation':{'macro_f1':.6}},
          {'candidate':'a','oof':{'macro_f1':.51},'validation':{'macro_f1':.601}}]
    assert gate(runs)=='v2_baseline'
    runs[-1]['validation']['macro_f1']=.61
    assert gate(runs)=='a'


def test_completed_study_is_preserved(tmp_path):
    (tmp_path/'selection.json').write_text('{}')
    with pytest.raises(ValueError,match='Preserve'):run(tmp_path)
