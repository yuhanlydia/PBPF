"""Receipt and create-once boundary checks, independent of population selection."""
import hashlib
import json
from pathlib import Path
import pytest
from pbpf.eesd.replay_admission import verify_file, publish_bundle


def test_bound_file_rejects_modified_bytes(tmp_path):
    p=tmp_path/'input'; p.write_bytes(b'first')
    seal={'sha256':hashlib.sha256(b'first').hexdigest(),'bytes':5}
    verify_file(p,seal)
    p.write_bytes(b'other')
    with pytest.raises(ValueError,match='checksum'):
        verify_file(p,seal)


def test_publish_refuses_existing_destination(tmp_path):
    out=tmp_path/'bundle'; out.mkdir(); (out/'keep').write_text('original')
    with pytest.raises(FileExistsError):
        publish_bundle(out,[],[],[],{})
    assert (out/'keep').read_text()=='original'


def test_publish_validation_failure_leaves_no_bundle(tmp_path):
    out=tmp_path/'bundle'
    with pytest.raises(ValueError,match='200 development and 500 primary'):
        publish_bundle(out,[],[],[],{})
    assert not out.exists()


def test_publish_all_views_and_bind_hashes(tmp_path):
    rows=[]
    for domain in ('apps_replay','codecontests_replay'):
        for i in range(700):
            rows.append(dict(domain=domain,task_id=f'{domain}/{i}',source_component_id=f'{domain}/{i}',split='development' if i<200 else 'primary'))
    out=tmp_path/'bundle'
    from pbpf.eesd.replay_materialization import MODEL_KEYS
    tokens=[dict(domain=r['domain'],task_id=r['task_id'],source_id=r['source_component_id'],models={k:{'input_tokens':100} for k in MODEL_KEYS}) for r in rows]
    publish_bundle(out,rows,rows,tokens,{'verification':'fixture'})
    receipt=json.loads((out/'admission.json').read_text())
    for name,seal in receipt['outputs'].items():
        verify_file(out/name,seal)
    assert receipt['status']=='data-admitted-not-generated-not-scored'
    assert len(list(out.glob('*/public/tasks.jsonl')))==2

@pytest.fixture
def bound_inputs(tmp_path):
    from pbpf.eesd.replay_admission import digest, compact
    root=tmp_path; joint=root/'joint'; joint.mkdir()
    spec=root/'docs/EESD_REPLAY_EXTENSION_SPEC_20260920.md'; spec.parent.mkdir(); spec.write_text('fixed spec')
    probe=root/'runs/eesd-setup/replay-token-probe/probe.py'; probe.parent.mkdir(parents=True)
    keys=('qwen25_7b','deepseek_6p7b','seed_coder_8b','starcoder2_15b')
    models=[(k,k,'rev') for k in keys]
    template='{statement}\n{observations}'
    probe.write_text(f'TEMPLATE={template!r}\nMODELS={models!r}\nPOLICY="policy"\n')
    (joint/'build_inventory.py').write_text('# provenance')
    row=dict(domain='apps_replay',source_id='s',task_id='t',statement='task',visible_tests=[{'input':str(i),'output':'ok'} for i in range(4)])
    files={'joint-public-candidates.jsonl':compact(row)+b'\n','joint-evaluator-candidates.jsonl':compact(row)+b'\n',
        'components.jsonl':compact(dict(source_id='s',quarantined=False,eligible_member_ids={'apps_replay':['t']}))+b'\n',
        'joint-graph.json':b'{}','candidate-rows.jsonl':b'','joint-edges.jsonl':b''}
    seals={}
    for name,data in files.items():
        p=joint/name;p.write_bytes(data);seals[name]={'sha256':digest(p),'bytes':len(data)}
    jr=joint/'receipt.json';jr.write_text(json.dumps(dict(schema='eesd-replay-joint-inventory-receipt-v1',status='candidate-preparation-only-not-split-not-queued',outputs=seals,input_sha256={str(spec.relative_to(root)):digest(spec)},source_sha256=digest(joint/'build_inventory.py'))))
    obs=[f'Observation {i}\nInput: '+json.dumps(t['input'],ensure_ascii=False)+'\nOutput: '+json.dumps(t['output'],ensure_ascii=False) for i,t in enumerate(row['visible_tests'],1)]
    message=[{'role':'user','content':template.format(statement='task',observations='\n\n'.join(obs))}]
    audit={k:row[k] for k in ('domain','source_id','task_id')}
    audit.update(message_sha256=hashlib.sha256(compact(message)).hexdigest(),models={k:dict(input_tokens=100,rendered_prompt_sha256='a'*64,token_ids_sha256='b'*64) for k in keys})
    results=root/'rows.jsonl';results.write_bytes(compact(audit)+b'\n')
    tr=root/'token.json'; tr.write_text(json.dumps(dict(schema='eesd-replay-token-probe-v1',status='cpu-probe-not-admitted-not-generated',joint_receipt=dict(path=str(jr),sha256=digest(jr),graph_sha256=seals['joint-graph.json']['sha256'],verified_complete_public_members=1),input=str(joint/'joint-public-candidates.jsonl'),input_sha256=seals['joint-public-candidates.jsonl']['sha256'],rows_sha256=digest(results),tool_sha256=digest(probe),template_sha256=hashlib.sha256(template.encode()).hexdigest(),policy='policy',input_cap=4096,no_truncation=True,proposed_max_new_tokens=1024,input_rows=1,models=[dict(key=k,model_id=k,revision='rev',chat_template='x',chat_template_sha256=hashlib.sha256(b'x').hexdigest(),tokenizer_files_sha256={'tokenizer.json':'a'*64}) for k in keys])))
    return root,jr,joint/'joint-public-candidates.jsonl',joint/'joint-evaluator-candidates.jsonl',tr,results


def test_receipts_bind_complete_inventory(bound_inputs):
    from pbpf.eesd.replay_admission import verify_inputs
    public,private,tokens,quarantine,evidence=verify_inputs(*bound_inputs)
    assert len(public)==len(private)==len(tokens)==1
    assert quarantine==set()


def test_changed_public_file_rejected(bound_inputs):
    from pbpf.eesd.replay_admission import verify_inputs
    bound_inputs[2].write_text('{}\n')
    with pytest.raises(ValueError,match='checksum'):
        verify_inputs(*bound_inputs)


def test_changed_prompt_in_self_consistent_audit_rejected(bound_inputs):
    from pbpf.eesd.replay_admission import verify_inputs,digest
    audit=json.loads(bound_inputs[5].read_text());audit['message_sha256']='a'*64
    bound_inputs[5].write_text(json.dumps(audit)+'\n')
    receipt=json.loads(bound_inputs[4].read_text());receipt['rows_sha256']=digest(bound_inputs[5])
    bound_inputs[4].write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='tokenized message'):
        verify_inputs(*bound_inputs)


def test_rejected_diagnostic_receipt_cannot_admit(bound_inputs):
    from pbpf.eesd.replay_admission import verify_inputs
    receipt=json.loads(bound_inputs[4].read_text());receipt['status']='rejected-normalized-io-diagnostic-only-not-for-selection'
    bound_inputs[4].write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='not complete/admissible'):
        verify_inputs(*bound_inputs)


def test_publish_rejects_missing_selected_token_audit(tmp_path):
    rows=[dict(domain=d,task_id=f'{d}/{i}',source_component_id=f'{d}/{i}',split='development' if i<200 else 'primary') for d in ('apps_replay','codecontests_replay') for i in range(700)]
    with pytest.raises(ValueError,match='selected token'):
        publish_bundle(tmp_path/'bundle',rows,rows,[],{})


def test_atomic_publish_cannot_replace_empty_directory(tmp_path):
    from pbpf.eesd.replay_admission import rename_new_directory
    source=tmp_path/'staging';source.mkdir();(source/'file').write_text('new')
    target=tmp_path/'existing';target.mkdir()
    with pytest.raises(FileExistsError):
        rename_new_directory(source,target)
    assert list(target.iterdir())==[] and source.exists()
