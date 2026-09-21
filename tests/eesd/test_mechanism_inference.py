import hashlib
import numpy as np
import pytest

from pbpf.eesd.mechanism_inference import (
    PairedMechanismData, bootstrap_gain, family_report, stable_seed, unsupported,
)


def make(labels, probabilities, *, sources=None, seeds=None, comparators=None, query_ids=None):
    n=len(labels)
    return PairedMechanismData(labels=labels, proposed=probabilities,
        comparators=comparators or {'baseline': probabilities},
        sources=sources or [str(i) for i in range(n)], seeds=seeds or [1701]*n,
        query_ids=query_ids or [str(i) for i in range(n)], expected_seeds=(1701,))


def test_metrics_source_equal_weight_not_query_weight():
    data=make([0,0,0], [[.8,.2],[.8,.2],[.2,.8]], sources=['a','a','b'])
    metrics=data.metrics()['proposed']
    assert metrics['accuracy']==.5
    assert metrics['nll']==pytest.approx((-np.log(.8)-np.log(.2))/2)
    assert metrics['brier']==pytest.approx((.08+1.28)/2)
    assert metrics['ece']==pytest.approx(.3) # confidence .8, source-weighted acc .5


def test_ece_recomputed_after_resampling_not_mean_source_ece():
    data=make([0,1], [[.75,.25],[.75,.25]])
    assert data.metrics()['proposed']['ece']==pytest.approx(.25)
    assert data.metrics([0,0])['proposed']['ece']==pytest.approx(.25)
    assert data.metrics([1,1])['proposed']['ece']==pytest.approx(.75)
    # Averaging individual-source ECEs would incorrectly give .5.
    assert data.metrics()['proposed']['ece'] != pytest.approx(.5)


def test_permuted_comparator_means_metrics_not_probabilities():
    data=make([0],[[.5,.5]], comparators={'p1':[[.9,.1]],'p2':[[.1,.9]]})
    metric=data.metrics()
    assert metric['comparator']['nll']==pytest.approx((-np.log(.9)-np.log(.1))/2)
    assert metric['comparator']['nll'] > -np.log(.5)
    assert metric['comparator']['brier']==pytest.approx((.02+1.62)/2)


def test_cross_seed_blocks_stay_paired_and_seed_equal_weight():
    # Opposite gains across seeds cancel within EACH source. Independent seed
    # resampling would incorrectly create nonzero bootstrap variance.
    data=PairedMechanismData(labels=[0]*4, proposed=[[.8,.2],[.2,.8]]*2,
        comparators={'baseline':[[.2,.8],[.8,.2]]*2}, sources=['a','a','b','b'],
        seeds=[1701,1702]*2, query_ids=['a1','a2','b1','b2'], expected_seeds=(1701,1702))
    report=bootstrap_gain(data,domain='toy',model='toy')
    for result in report['metrics'].values():
        assert result['gain']==pytest.approx(0)
        assert result['ci95']==pytest.approx([0,0])
        assert result['p_raw']==1
    assert report['replicates']==10000


def test_zero_difference_and_stable_seed():
    report=bootstrap_gain(make([0,1],[[.6,.4],[.1,.9]]),domain='d',model='m')
    assert stable_seed('d','m','all')==int.from_bytes(hashlib.sha256(b'314159|d|m|all').digest()[:8],'big')
    assert all(r['p_raw']==1 and r['gain']==0 for r in report['metrics'].values())
    assert report==bootstrap_gain(make([0,1],[[.6,.4],[.1,.9]]),domain='d',model='m')


def test_nonzero_constant_gain_uses_plus_one_p_floor():
    data=make([0,0],[[.9,.1],[.9,.1]],comparators={'baseline':[[.6,.4],[.6,.4]]})
    result=bootstrap_gain(data,domain='d',model='m')['metrics']['nll']
    assert result['gain']==pytest.approx(np.log(.9/.6))
    assert result['p_raw']==1/10001
    assert result['ci95']==pytest.approx([result['gain']]*2)


@pytest.mark.parametrize('change', ['missing_seed','duplicate_query','bad_probabilities'])
def test_rejects_invalid_pairing(change):
    args=dict(labels=[0,0],proposed=[[.8,.2],[.6,.4]],comparators={'b':[[.5,.5]]*2},
              sources=['s','s'],seeds=[1701,1702],query_ids=['q','q'],expected_seeds=(1701,1702))
    if change=='missing_seed':args['sources']=['s','t']
    elif change=='duplicate_query':args['seeds']=[1701,1701]
    else:args['proposed']=[[.8,.8],[.6,.4]]
    with pytest.raises(ValueError):PairedMechanismData(**args)


def test_holm_complete_family_and_missing_refusal():
    rows={key:{'status':'supported','p_raw':p} for key,p in [('a',.01),('b',.04),('c',.03)]}
    report=family_report(rows,family=['a','b','c'])
    assert report['adjusted_p']==pytest.approx({'a':.03,'b':.06,'c':.06})
    assert report['reject']=={'a':True,'b':False,'c':False}
    with pytest.raises(ValueError,match='complete'):
        family_report({'a':rows['a']},family=['a','b'])


def test_unsupported_is_not_null_and_blocks_family_conclusions():
    result=bootstrap_gain(make([0],[[.7,.3]]),domain='d',model='m')
    assert result['status']=='unsupported' and 'metrics' not in result
    report=family_report({'a':{'status':'supported','p_raw':.0001},
                          'b':unsupported('history leaves no target')},family=['a','b'])
    assert report['complete'] is False
    assert report['adjusted_p'] is None and report['reject'] is None
    assert report['unsupported']['b']=='history leaves no target'


def test_centered_nonlinear_ece_p_matches_enumerated_two_source_draws():
    data=make([0,1],[[.75,.25],[.75,.25]], comparators={'perfect':[[1,0],[0,1]]})
    report=bootstrap_gain(data,domain='enumerated',model='ece')
    result=report['metrics']['ece']
    # Four equally likely ordered draws: aa, ab, ba give ECE .25, bb gives .75.
    # Observed gain -.25; ONLY bb has centered distance .5 >= .25.
    rng=np.random.Generator(np.random.PCG64(stable_seed('enumerated','ece','all')))
    draws=rng.integers(0,2,size=(10000,2))
    expected=(1+sum(np.all(draws==1,axis=1)))/10001
    assert result['p_raw']==expected
    assert result['ci95']==pytest.approx([-.75,-.25])


def test_full_secondary_family_cannot_reject_at_10k_resolution():
    family=[f'h{i}' for i in range(1560)]
    report=family_report({key:{'status':'supported','p_raw':1/10001} for key in family},family=family)
    assert report['complete']
    assert not any(report['reject'].values())
    assert min(report['adjusted_p'].values())==pytest.approx(1560/10001)


def test_identical_three_permutations_are_exact_null_despite_average_roundoff():
    probabilities=[[.63,.37],[.63,.37]]
    data=make([0,1],probabilities,comparators={str(i):probabilities for i in range(3)})
    result=bootstrap_gain(data,domain='identical',model='three')
    for value in result['metrics'].values():
        assert value['gain']==0
        assert value['ci95']==[0,0]
        assert value['p_raw']==1


@pytest.mark.parametrize('p',[0,1/10002])
def test_locked_family_rejects_p_below_10k_resolution(p):
    family=[str(i) for i in range(1560)]
    with pytest.raises(ValueError,match='resolution'):
        family_report({key:{'status':'supported','p_raw':p} for key in family},family=family)
