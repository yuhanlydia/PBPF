import hashlib
import json

import pytest

from pbpf.apbpf.candidate_import import import_domain_banks
from pbpf.apbpf.codearc_bank import file_sha


def bank_fixture(tmp_path, *, duplicate_public=False):
    public = tmp_path/'public'; public.mkdir()
    tasks = [{'task_id': f'RBR/{s}', 'source_component_id': f'p-{s}', 'split': s}
             for s in ('train', 'development', 'primary')]
    public_tasks = tasks + ([{**tasks[0], 'task_id': 'RBR/duplicate-alias'}] if duplicate_public else [])
    (public/'tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in public_tasks))
    (public/'manifest.json').write_text(json.dumps({'public_tasks_sha256': file_sha(public/'tasks.jsonl')}))
    model = {'id': 'pinned', 'revision': 'a'*40}
    entries = []
    for task in tasks:
        root = tmp_path/task['split']; root.mkdir()
        run = {'schema': 'apbpf-rbr-generation-v1', 'model': model['id'], 'revision': model['revision'],
               'seed': 1701, 'candidates': 8, 'temperature': .8, 'top_p': .95, 'max_new_tokens': 1024,
               'public_tasks_sha256': file_sha(public/'tasks.jsonl'), 'source_sha256': 'b'*64,
               'split': task['split'], 'task_ids': [task['task_id']],
               'source_component_ids': [task['source_component_id']]}
        (root/'run.json').write_text(json.dumps(run))
        row = {**task, 'candidates': [{'candidate_id': f'{task["task_id"]}/{i}',
                 'code': 'print(1)', 'raw_completion': 'print(1)'} for i in range(8)]}
        name = task['task_id'].replace('/', '-')+'.json'; (root/name).write_text(json.dumps(row))
        (root/'complete.json').write_text(json.dumps({'run_sha256': file_sha(root/'run.json'),
                                                   'files': {name: file_sha(root/name)}}))
        (root/'old-hidden-results.json').write_text('PRIVATE_CANARY')
        entries.append({'path': str(root), 'complete_sha256': file_sha(root/'complete.json')})
    return public, model, entries


def test_full_import_preserves_candidate_bytes_and_excludes_side_evaluations(tmp_path):
    public, model, entries = bank_fixture(tmp_path)
    output = tmp_path/'imported'
    imported = import_domain_banks(entries, public, output, domain='rbr', model=model, seed=1701)
    assert sum(r['groups'] for r in imported) == 3
    for source, receipt in zip(entries, imported, strict=True):
        assert file_sha(output/receipt['directory']/'complete.json') == source['complete_sha256']
    assert not any('PRIVATE_CANARY' in f.read_text() for f in output.rglob('*') if f.is_file())


def test_import_matches_first_representative_of_duplicate_public_sources(tmp_path):
    public, model, entries = bank_fixture(tmp_path, duplicate_public=True)
    imported = import_domain_banks(entries, public, tmp_path/'imported', domain='rbr', model=model, seed=1701)
    assert sum(row['groups'] for row in imported) == 3


def test_import_rejects_missing_sources_before_copying(tmp_path):
    public, model, entries = bank_fixture(tmp_path)
    output = tmp_path/'imported'
    with pytest.raises(ValueError, match='every public'):
        import_domain_banks(entries[:-1], public, output, domain='rbr', model=model, seed=1701)
    assert not output.exists()


def test_import_rejects_duplicate_sources_and_wrong_model(tmp_path):
    public, model, entries = bank_fixture(tmp_path)
    with pytest.raises(ValueError, match='source inventory'):
        import_domain_banks(entries + entries[:1], public, tmp_path/'one', domain='rbr', model=model, seed=1701)
    with pytest.raises(ValueError, match='model'):
        import_domain_banks(entries, public, tmp_path/'two', domain='rbr', model={**model, 'id': 'other'}, seed=1701)


def test_import_rejects_hidden_fields_even_with_resealed_bytes(tmp_path):
    public, model, entries = bank_fixture(tmp_path)
    from pathlib import Path
    root = Path(entries[0]['path']); name = 'RBR-train.json'
    row = json.loads((root/name).read_text()); row['hidden_labels'] = [1]*8
    (root/name).write_text(json.dumps(row))
    complete = json.loads((root/'complete.json').read_text()); complete['files'][name] = file_sha(root/name)
    (root/'complete.json').write_text(json.dumps(complete)); entries[0]['complete_sha256'] = file_sha(root/'complete.json')
    with pytest.raises(ValueError, match='source inventory'):
        import_domain_banks(entries, public, tmp_path/'imported', domain='rbr', model=model, seed=1701)
