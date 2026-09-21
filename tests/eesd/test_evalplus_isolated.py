import json
from pathlib import Path
import pytest
from test_mechanism_runner import load_script
from test_evalplus_public import receipt, sha
from pbpf.eesd.evalplus_public import load_public_dataset


@pytest.fixture
def inputs(receipt, tmp_path):
    load_script('materialize_eesd_evalplus').materialize(receipt, tmp_path / 'data')
    public = tmp_path / 'data/public'
    lock = sha(public / 'manifest.json')
    rows, binding = load_public_dataset(public, 'humaneval', lock)
    transfer = tmp_path / 'transfer'
    transfer.mkdir()
    samples = transfer / 'samples.jsonl'
    samples.write_text(''.join(json.dumps({'task_id': r['task_id'], 'solution': 'def f(): pass'})+'\n' for r in rows))
    report = dict(schema='eesd-evalplus-transfer-v1',dataset='humaneval',tasks=2,
                  samples_sha256=sha(samples),data_lock=binding,
                  evaluation_status='pending-isolated-official-evaluation')
    (transfer / 'report.json').write_text(json.dumps(report))
    return dict(transfer_root=transfer, transfer_report_sha256=sha(transfer/'report.json'),
                public_root=public, public_manifest_sha256=lock,private_root=tmp_path/'data/private')


def test_mounts_environment_and_commands_are_allowlisted(tmp_path):
    m = load_script('evaluate_eesd_evalplus_isolated')
    runtime = m.runtime_info()
    command = m.sandbox_command(runtime, tmp_path, 'probe')
    assert '--unshare-all' in command and '--die-with-parent' in command and '--clearenv' in command
    sources = [command[i+1] for i,v in enumerate(command) if v in ('--ro-bind','--bind')]
    assert '/root' not in sources and '/root/PBPF' not in sources
    assert str(tmp_path) in sources
    assert '--dataset' not in command
    assert not any('TOKEN' in part for part in command)
    with pytest.raises(ValueError):
        m.sandbox_command(runtime,tmp_path,'bash')
    sanitize = m.sandbox_command(runtime,tmp_path,'sanitize',dataset='humaneval',private_root=tmp_path)
    assert '--dataset' not in sanitize  # EvalPlus 0.3.1 sanitize has no dataset argument.
    assert any(x.endswith('/evalplus.sanitize') for x in sanitize)


@pytest.mark.parametrize('change',['report','samples','raw','private_manifest','duplicate'])
def test_integrity_refused_before_subprocess(inputs, change):
    m = load_script('evaluate_eesd_evalplus_isolated')
    if change == 'report':
        (inputs['transfer_root']/'report.json').write_text('{}')
    elif change == 'samples':
        (inputs['transfer_root']/'samples.jsonl').write_text('{}\n')
    elif change == 'raw':
        (inputs['private_root']/'mbpp.jsonl').write_text('{}\n')
    elif change == 'private_manifest':
        (inputs['private_root']/'manifest.json').write_text('{}')
    else:
        path = inputs['transfer_root']/'samples.jsonl'
        path.write_text(path.read_text().splitlines()[0]+'\n')
        rp = inputs['transfer_root']/'report.json'
        r=json.loads(rp.read_text()); r['samples_sha256']=sha(path);rp.write_text(json.dumps(r))
        inputs['transfer_report_sha256']=sha(rp)
    with pytest.raises(ValueError): m.validate_inputs(**inputs)


def test_probe_failure_never_executes_candidates(inputs,tmp_path,monkeypatch):
    m=load_script('evaluate_eesd_evalplus_isolated'); calls=[]
    def run(command, log, timeout):
        calls.append(command)
        assert not (tmp_path/'out/work/samples.jsonl').exists()
        return 1
    monkeypatch.setattr(m,'run_command',run)
    assert m.evaluate(output=tmp_path/'out',**inputs) == 3
    assert len(calls)==1
    r=json.loads((tmp_path/'out/status.json').read_text())
    assert r['status']=='infrastructure_failure' and 'metrics' not in r


def test_mock_official_pipeline_seals_base_plus_and_refuses_reuse(inputs,tmp_path,monkeypatch):
    m=load_script('evaluate_eesd_evalplus_isolated'); calls=[];out=tmp_path/'out'
    def run(command,log,timeout):
        calls.append(command)
        if any(x.endswith('/evalplus.sanitize') for x in command):
            (out/'work/samples-sanitized.jsonl').write_bytes((out/'work/samples.jsonl').read_bytes())
        if any(x.endswith('/evalplus.evaluate') for x in command):
            (out/'work/samples-sanitized_eval_results.json').write_text(json.dumps({'eval':{
                'HumanEval/0':[{'task_id':'HumanEval/0','base_status':'pass','plus_status':'fail'}],
                'HumanEval/1':[{'task_id':'HumanEval/1','base_status':'pass','plus_status':'pass'}]}}))
        return 0
    monkeypatch.setattr(m,'run_command',run)
    assert m.evaluate(output=out,**inputs)==0
    assert len(calls)==3
    report=json.loads((out/'report.json').read_text())
    assert report['metrics']['base_pass_at_1']==1 and report['metrics']['plus_pass_at_1']==.5
    complete=json.loads((out/'complete.json').read_text())
    assert complete['report_sha256']==sha(out/'report.json')
    with pytest.raises(FileExistsError): m.evaluate(output=out,**inputs)


@pytest.mark.parametrize('value',[{}, {'HumanEval/0':[]}, {'HumanEval/0':[{'base_status':'pass','plus_status':None}]}])
def test_result_coverage_or_missing_status_rejected(value):
    m=load_script('evaluate_eesd_evalplus_isolated')
    with pytest.raises(ValueError): m.summarize({'eval':value},['HumanEval/0'])


def test_cli_requires_external_locks_before_probe(tmp_path):
    import subprocess,sys
    from test_mechanism_runner import ROOT
    p=subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_eesd_evalplus_isolated.py'),
                      '--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert p.returncode==2 and 'external report/manifest hashes required' in p.stderr
    assert not (tmp_path/'out').exists()


def test_output_symlinks_are_rejected(tmp_path):
    m=load_script('evaluate_eesd_evalplus_isolated')
    secret=tmp_path/'outside';secret.write_text('private')
    work=tmp_path/'work';work.mkdir();(work/'result.json').symlink_to(secret)
    with pytest.raises(ValueError):m.output_file(work,'result.json')
