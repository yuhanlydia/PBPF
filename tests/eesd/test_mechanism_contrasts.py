import json
import shutil
import sys
import numpy as np
import pytest
import yaml
from test_mechanism_artifacts import artifacts,reseal
from test_mechanism_runner import load_script,ROOT


@pytest.fixture
def complete_artifacts(artifacts,monkeypatch):
    cfg=yaml.safe_load((ROOT/'configs/experiments/eesd_iclr2027.yaml').read_text())
    cfg['evidence']['bootstrap_draws']=10
    artifacts['config'].write_text(yaml.safe_dump(cfg))
    shutil.rmtree(artifacts['report_dir'])
    dataset='runbugrun' if artifacts['domain']=='rbr' else 'codearc'
    monkeypatch.setattr(sys,'argv',['runner','--cache',str(artifacts['cache']),'--config',str(artifacts['config']),
        '--dataset',dataset,'--model',artifacts['model'],'--seed',str(artifacts['seed']),
        '--output',str(artifacts['report_dir'])])
    load_script('run_eesd_evidence_matrix').main()
    return artifacts


def test_all_49_slots_history_binary_and_metric_permutations(complete_artifacts):
    from pbpf.eesd.mechanism_contrasts import reconstruct_contrasts
    contrasts,receipt=reconstruct_contrasts(**complete_artifacts)
    assert len(contrasts)==49
    assert all(x['status']=='available' for x in contrasts.values())
    assert contrasts['A8/n4']['alias_of']=='core/tuned'
    assert len(contrasts['A7/permuted']['comparators'])==3
    assert contrasts['A7/permuted']['comparator_aggregation']=='mean-metrics-not-probabilities'
    assert contrasts['A10/binary']['proposed'].shape[1]==2
    assert set(contrasts['A10/binary']['labels'])=={0,1}
    for n in (1,2,4,8):
        arm=contrasts[f'A8/n{n}']
        assert len(arm['labels'])==2*(10-n)
        assert json.loads(arm['query_ids'][0])['test_index']==n
        assert all(type(s) is int for s in arm['seeds'])
    for arm in (v for k,v in contrasts.items() if k.startswith('A9/')):
        assert len(arm['mask'])==12 and arm['population']=='pending-cross-seed-common-source-support'
    assert receipt['reserved_slots']==49


def test_same_kernel_mass_uses_only_development(complete_artifacts,monkeypatch):
    import pbpf.eesd.mechanism_contrasts as module
    observed=[];original=module.select_same_kernel_mass
    def trace(runner,examples,selection,masses,ece_bins):
        observed.append({e['cluster'] for e in examples})
        return original(runner,examples,selection,masses,ece_bins)
    monkeypatch.setattr(module,'select_same_kernel_mass',trace)
    module.reconstruct_contrasts(**complete_artifacts)
    assert len(observed)==1 and all('development' in x for x in observed[0])


def test_saved_scalar_tamper_is_rejected(complete_artifacts):
    from pbpf.eesd.mechanism_contrasts import reconstruct_contrasts
    out=complete_artifacts['report_dir'];p=out/'report.json';r=json.loads(p.read_text())
    r['factorial'][0]['fixed_nll']+=1; p.write_text(json.dumps(r));reseal(out)
    with pytest.raises(ValueError,match='factorial'):
        reconstruct_contrasts(**complete_artifacts)


def test_absent_report_slot_remains_with_reason(complete_artifacts):
    from pbpf.eesd.mechanism_contrasts import reconstruct_contrasts
    out=complete_artifacts['report_dir'];p=out/'report.json';r=json.loads(p.read_text())
    r['history_sweep']=[x for x in r['history_sweep'] if x['history_size']!=8]
    p.write_text(json.dumps(r));reseal(out)
    contrasts,_=reconstruct_contrasts(**complete_artifacts)
    assert len(contrasts)==49
    assert contrasts['A8/n8']['status']=='unsupported' and contrasts['A8/n8']['reason']


def test_a9_uses_prediction_mass_boundary_without_changing_a6_a7(complete_artifacts,monkeypatch):
    import pbpf.eesd.mechanism_contrasts as module
    from pbpf.eesd.evidence import effective_mass,effective_evidence_weights
    weights=np.array([0.9999999901087866,0.9999999963221335,1.0000000128792528,1.0000000019397441])
    raw_mass=effective_mass(weights)
    predicted_mass=float(effective_evidence_weights(weights).sum())
    assert raw_mass>4.0 and predicted_mass==4.0
    original=module.runpy.run_path
    def patched(path,*args,**kwargs):
        runner=original(path,*args,**kwargs)
        if 'build_examples' in runner:
            build=runner['build_examples']
            def boundary_examples(rows,history_size,strength,*,binary):
                examples=build(rows,history_size,strength,binary=binary)
                if history_size==4:
                    for example in examples:example['mass']=raw_mass
                return examples
            runner['build_examples']=boundary_examples
        return runner
    monkeypatch.setattr(module.runpy,'run_path',patched)
    contrasts,_=module.reconstruct_contrasts(**complete_artifacts)
    assert contrasts['A9/[3,4]']['public_mask'].all()
    assert np.all(contrasts['A9/[3,4]']['public_mass']==4.0)
    assert contrasts['A6/assessment_mean']['public_mean_mass']>4.0
    assert len(contrasts['A7/permuted']['comparators'])==3


def test_public_support_ignores_assessment_outcomes_and_matches_full(complete_artifacts):
    from pbpf.eesd.mechanism_contrasts import derive_public_support,reconstruct_contrasts,BIN_IDS
    from pbpf.apbpf.codearc_bank import file_sha
    first,receipt=derive_public_support(**complete_artifacts)
    full,_=reconstruct_contrasts(**complete_artifacts)
    for key in BIN_IDS:
        assert first[key]['query_ids']==full[key]['query_ids']
        np.testing.assert_array_equal(first[key]['public_mask'],full[key]['public_mask'])
        np.testing.assert_array_equal(first[key]['public_mass'],full[key]['public_mass'])
    cache=complete_artifacts['cache'];value=json.loads(cache.read_text())
    for row in value['records']:
        if row['split']=='primary':row['outcomes']={'DO_NOT_ACCESS':'opaque deliberately invalid labels'}
    cache.write_text(json.dumps(value))
    seal=cache.with_suffix('.binding.json');s=json.loads(seal.read_text());s['cache_sha256']=file_sha(cache);seal.write_text(json.dumps(s))
    out=complete_artifacts['report_dir'];rp=out/'report.json';r=json.loads(rp.read_text());r['cache_sha256']=file_sha(cache);rp.write_text(json.dumps(r));reseal(out)
    second,_=derive_public_support(**complete_artifacts)
    for key in BIN_IDS:
        assert first[key]['query_ids']==second[key]['query_ids']
        np.testing.assert_array_equal(first[key]['public_mask'],second[key]['public_mask'])
    assert receipt['assessment_outcomes_accessed'] is False


def test_public_support_verifies_development_selected_strength(complete_artifacts):
    from pbpf.eesd.mechanism_contrasts import derive_public_support
    out=complete_artifacts['report_dir'];rp=out/'report.json';r=json.loads(rp.read_text())
    r['selected']['effective']['strength']=123.;rp.write_text(json.dumps(r));reseal(out)
    with pytest.raises(ValueError,match='development'):
        derive_public_support(**complete_artifacts)


def test_public_support_does_not_even_access_assessment_label_field(complete_artifacts,monkeypatch):
    import pbpf.eesd.mechanism_contrasts as module
    class Guarded(dict):
        def __getitem__(self,key):
            if key=='outcomes':raise AssertionError('assessment outcomes accessed before support seal')
            return super().__getitem__(key)
        def get(self,key,*args):
            if key=='outcomes':raise AssertionError('assessment outcomes accessed before support seal')
            return super().get(key,*args)
    read=module.read
    def guarded_read(path):
        value=read(path)
        if str(path)==str(complete_artifacts['cache']):
            value['records']=[Guarded(row) if row['split']=='primary' else row for row in value['records']]
        return value
    def forbidden_loader(**kwargs):raise AssertionError('full artifact loader invoked before support seal')
    monkeypatch.setattr(module,'read',guarded_read)
    monkeypatch.setattr(module,'load_mechanism_artifacts',forbidden_loader)
    support,_=module.derive_public_support(**complete_artifacts)
    assert len(support)==3
