"""Analytically check dependence handling; these are not benchmark results."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from paper4_kbvqa.evaluation.statistics import (
    _resample_indices, paired_bootstrap_accuracy, paired_cluster_swap_test,
    paired_selective_bootstrap,
)


def test_resampling_keeps_every_question_in_selected_image():
    for index in _resample_indices(5,30,2026,['a','a','b','b','b']):
        counts=np.bincount(index,minlength=5)
        assert counts[0]==counts[1]
        assert counts[2]==counts[3]==counts[4]
        assert counts[0]+counts[2]==2


def test_cluster_intervals_do_not_count_duplicate_questions_as_independent():
    a=[1]*10+[0]*10; b=[0]*10+[1]*10
    clustered=paired_bootstrap_accuracy(a,b,groups=['a']*10+['b']*10,seed=2026)
    iid=paired_bootstrap_accuracy(a,b,seed=2026)
    assert clustered['ci95_low']==-1 and clustered['ci95_high']==1
    assert iid['ci95_low']>clustered['ci95_low']
    assert iid['ci95_high']<clustered['ci95_high']
    assert clustered['n_clusters']==2


def test_cluster_swap_matches_exact_two_image_enumeration():
    # Four equiprobable image-label assignments: two are equally extreme.
    result=paired_cluster_swap_test([1,1,1,1],[0,0,0,0],['a','a','b','b'])
    assert result['p_value']==.5 and result['n_assignments']==4
    assert result['mode']=='exact_enumeration'
    duplicated=paired_cluster_swap_test([1]*8,[0]*8,['a']*4+['b']*4)
    assert duplicated['p_value']==result['p_value']


def test_cluster_swap_monte_carlo_is_seeded_and_nonzero():
    args=([1]*17,[0]*17,list(range(17)))
    a=paired_cluster_swap_test(*args,n_permutations=127,seed=4)
    assert a==paired_cluster_swap_test(*args,n_permutations=127,seed=4)
    assert a['mode']=='monte_carlo_plus_one' and a['p_value']>=1/128


@pytest.mark.parametrize('groups',[['a'],['a','a'],[None,'b']])
def test_cluster_inference_rejects_missing_or_unreplicated_units(groups):
    with pytest.raises(ValueError):
        paired_bootstrap_accuracy([0,1],[1,0],groups=groups,n_boot=10)


def test_cluster_selective_intervals_preserve_undefined_risk():
    r=paired_selective_bootstrap([.5]*4,[0,0,1,1],1.1,[.5]*4,[0,0,1,1],0,
                                groups=['a','a','b','b'],n_boot=40)
    assert r['coverage_difference']['difference']==-1
    assert r['selective_risk_difference']['ci95_low'] is None
    assert r['selective_risk_difference']['valid_bootstrap_replicates']==0


def test_report_rejects_missing_or_mismatched_image_identity():
    path=Path(__file__).resolve().parents[1]/'scripts/report_study.py'
    spec=importlib.util.spec_from_file_location('report_cluster_test',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with pytest.raises(ValueError): module.aligned_image_groups([{}],[{}])
    with pytest.raises(ValueError):
        module.aligned_image_groups([{'image_group':'a'}],[{'image_group':'b'}])
