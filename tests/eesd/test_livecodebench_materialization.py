import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / 'scripts/materialize_eesd_livecodebench.py'
    spec = importlib.util.spec_from_file_location('lcb_materializer', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def fixture(tmp_path, *, duplicate=False):
    m = module()
    entries, files = [], []
    for index, name in enumerate(m.FILES):
        row = dict(question_id='0' if duplicate else str(index), platform='atcoder',
                   question_title='Title', question_content='Problem', starter_code='',
                   contest_date='2025-01-01T00:00:00', difficulty='easy',
                   private_test_cases='PRIVATE_SENTINEL', public_test_cases='TEST_SENTINEL',
                   metadata='REFERENCE_SENTINEL')
        path = tmp_path / name
        path.write_text(json.dumps(row) + '\n')
        digest = m.sha(path)
        entries.append(dict(path=name, size=path.stat().st_size,
                            lfs=dict(sha256=digest, size=path.stat().st_size)))
        files.append(dict(filename=name, path=str(path), bytes=path.stat().st_size,
                          sha256=digest, official_lfs_sha256=digest, verified=True,
                          statistics={'rows': 1}))
    inventory = tmp_path / 'inventory.json'
    inventory.write_text(json.dumps(dict(repo_id=m.REPO, revision=m.REVISION, files=entries)))
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(schema='eesd-livecodebench-download-v1',
        status='complete', repo_id=m.REPO, revision=m.REVISION, release='release_v6',
        inventory_sha256=m.sha(inventory), files=files, statistics={'rows': 6})))
    return m, receipt, inventory


def test_public_projection_excludes_all_evaluator_fields_and_preserves_every_task(tmp_path):
    m, receipt, inventory = fixture(tmp_path)
    output = tmp_path / 'out'
    result = m.materialize(receipt, inventory, output, m.sha(receipt))
    public = (output / 'public/tasks.jsonl').read_text()
    assert all(value not in public for value in ['PRIVATE_SENTINEL', 'TEST_SENTINEL', 'REFERENCE_SENTINEL'])
    rows = [json.loads(line) for line in public.splitlines()]
    assert len(rows) == 6 and len({r['task_id'] for r in rows}) == 6
    assert result['tasks'] == 6
    assert (output / 'private/manifest.json').is_file()
    with pytest.raises(FileExistsError):
        m.materialize(receipt, inventory, output, m.sha(receipt))


def test_cross_file_duplicate_is_rejected_without_publishing(tmp_path):
    m, receipt, inventory = fixture(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match='duplicate'):
        m.materialize(receipt, inventory, tmp_path / 'out', m.sha(receipt))
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('corruption', ['raw', 'receipt', 'incomplete'])
def test_chain_tampering_and_incomplete_download_rejected(tmp_path, corruption):
    m, receipt, inventory = fixture(tmp_path)
    expected = m.sha(receipt)
    if corruption == 'raw':
        (tmp_path / m.FILES[0]).write_text('{}\n')
    elif corruption == 'receipt':
        receipt.write_text(receipt.read_text() + ' ')
    else:
        data = json.loads(receipt.read_text()); data['status'] = 'running'
        receipt.write_text(json.dumps(data)); expected = m.sha(receipt)
    with pytest.raises(ValueError):
        m.materialize(receipt, inventory, tmp_path / 'out', expected)
    assert not (tmp_path / 'out').exists()
