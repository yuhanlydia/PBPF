"""Public-only, non-executing Cartesian Replay launch planning."""
import importlib.util
import json
from pathlib import Path
import pytest

PATH=Path(__file__).resolve().parents[2]/'scripts/plan_eesd_replay_generation.py'
spec=importlib.util.spec_from_file_location('replay_planner',PATH)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

@pytest.fixture
def setup(tmp_path):
    root=tmp_path/'repo';root.mkdir()
    for name in ('scripts/generate_eesd_replay_bank.py','scripts/plan_eesd_replay_generation.py',
                 'src/pbpf/eesd/replay_generation.py','src/pbpf/eesd/replay_prompt.py','src/pbpf/apbpf/rbr_prompt.py'):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('# fixture '+name)
    bundle=root/'bundle';bundle.mkdir();(bundle/'admission.json').write_text('{}')
    calls=[]
    def loader(bundle_arg,sha,domain,split,family):
        assert bundle_arg==bundle and sha==m.sha(bundle/'admission.json')
        calls.append((domain,split,family))
        count=200 if split=='development' else 500
        rows=[dict(domain=domain,split=split,task_id=f'{domain}/{split}/{i}',source_component_id=f'source/{domain}/{split}/{i}') for i in range(count)]
        return dict(rows=rows,audits={},model={'model_id':family,'revision':'fixed'},bindings={'public_tasks_sha256':'1'*64})
    return root,bundle,loader,calls

def test_exact_matrix_public_only_and_same_population(setup,tmp_path):
    root,bundle,loader,calls=setup
    original=root/'old-manifest.json';original.write_text('untouched')
    result=m.build_plan(root,bundle,m.sha(bundle/'admission.json'),tmp_path/'runs',loader=loader)
    assert len(calls)==16
    assert len(result['cells'])==24
    assert len({(c['domain'],c['family'],c['seed']) for c in result['cells']})==24
    commands=[]
    for cell in result['cells']:
        assert set(cell['splits'])=={'development','primary'}
        for split,job in cell['splits'].items():
            assert job['components']==(200 if split=='development' else 500)
            assert job['population_sha256']==result['populations'][cell['domain']][split]['sha256']
            argv=job['argv'];assert isinstance(argv,list)
            assert argv[1].endswith('/scripts/generate_eesd_replay_bank.py')
            assert '--bundle' in argv and '--admission-sha256' in argv
            assert all('evaluator' not in arg and 'private' not in arg for arg in argv)
            assert Path(argv[argv.index('--output')+1])==tmp_path/'runs'/'replay-mechanism-banks'/cell['domain']/cell['family']/f"seed{cell['seed']}"/split
            commands.append(argv)
    assert len(commands)==48
    assert original.read_text()=='untouched'
    assert not (tmp_path/'runs').exists()
    assert result['status']=='planned-generation-only-not-launched'
    assert set(result['sources'])==set(m.SOURCE_FILES)

@pytest.mark.parametrize('defect',['missing','different_population','cross_split_source'])
def test_incomplete_or_changed_population_fails(setup,tmp_path,defect):
    root,bundle,loader,_=setup
    def broken(*args):
        data=loader(*args)
        if defect=='missing': data['rows'].pop()
        elif defect=='different_population' and args[-1]=='seed_coder_8b': data['rows'][0]['task_id']='replacement'
        elif defect=='cross_split_source' and args[-2]=='primary': data['rows'][0]['source_component_id']=f'source/{args[-3]}/development/0'
        return data
    with pytest.raises(ValueError):m.build_plan(root,bundle,m.sha(bundle/'admission.json'),tmp_path/'runs',loader=broken)

def test_manifest_create_once_preserves_existing(tmp_path):
    out=tmp_path/'manifest.json'
    m.write_manifest_once(out,{'status':'planned'})
    assert json.loads(out.read_text())=={'status':'planned'}
    with pytest.raises(FileExistsError):m.write_manifest_once(out,{'status':'overwrite'})
    assert json.loads(out.read_text())=={'status':'planned'}

def test_wrong_admission_rejected_before_public_loading(setup,tmp_path):
    root,bundle,loader,calls=setup
    with pytest.raises(ValueError,match='admission checksum'):
        m.build_plan(root,bundle,'0'*64,tmp_path/'runs',loader=loader)
    assert calls==[]

def test_source_mutation_during_planning_rejected(setup,tmp_path):
    root,bundle,loader,_=setup
    def mutating(*args):
        data=loader(*args)
        (root/'scripts/generate_eesd_replay_bank.py').write_text('changed')
        return data
    with pytest.raises(ValueError,match='changed during planning'):
        m.build_plan(root,bundle,m.sha(bundle/'admission.json'),tmp_path/'runs',loader=mutating)
