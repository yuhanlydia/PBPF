"""Count sealed generation work without reading evaluator outcomes."""
import hashlib
import json
from pathlib import Path


def summarize_bank(directory: Path) -> dict:
    directory = Path(directory)
    raw = (directory/'run.json').read_bytes()
    run = json.loads(raw)
    run_sha = hashlib.sha256(raw).hexdigest()
    expected = dict(zip(run['task_ids'], run['source_component_ids'], strict=True))
    if len(expected) != run['components'] or len(set(expected.values())) != len(expected):
        raise ValueError('invalid source inventory')
    complete_path = directory/'complete.json'
    complete = json.loads(complete_path.read_text()) if complete_path.exists() else None
    if complete and complete.get('run_sha256') != run_sha:
        raise ValueError('run checksum mismatch')
    verified, unsealed, file_hashes = [], [], {}
    for path in sorted(directory.glob('*.json')):
        if path.name in {'run.json','complete.json'}:
            continue
        checksum = path.with_suffix('.sha256')
        expected_sha = (complete.get('files', {}).get(path.name) if complete is not None
                        else checksum.read_text().strip() if checksum.exists() else None)
        if expected_sha is None:
            unsealed.append(path.name)
            continue
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if expected_sha != digest:
            raise ValueError(f'candidate checksum mismatch: {path}')
        row = json.loads(content)
        if (row['task_id'] not in expected
            or expected[row['task_id']] != row['source_component_id']
            or row['split'] != run['split']
            or len(row['candidates']) != run['candidates']):
            raise ValueError('candidate inventory mismatch')
        if path.name != row['task_id'].replace('/','-')+'.json':
            raise ValueError('candidate filename mismatch')
        tokens = [row['prompt_metadata']['input_tokens']]+[c['generated_tokens'] for c in row['candidates']]
        if any(type(n) is not int or n < 0 for n in tokens):
            raise ValueError('invalid token ledger value')
        verified.append(row)
        file_hashes[path.name] = digest
    if len({r['task_id'] for r in verified}) != len(verified):
        raise ValueError('duplicate task inventory')
    if complete is not None:
        if (len(verified) != len(expected) or unsealed
            or complete.get('files') != file_hashes):
            raise ValueError('complete inventory or checksum mismatch')
    timing_rows = [r for r in verified if 'elapsed_seconds' in r]
    return {'bank':str(directory.resolve()), 'model':run['model'], 'revision':run['revision'],
        'seed':run['seed'], 'split':run['split'], 'run_sha256':run_sha,
        'status':'complete' if complete is not None else 'partial',
        'expected_sources':len(expected), 'verified_sources':len(verified),
        'verified_candidates':sum(len(r['candidates']) for r in verified),
        'input_tokens':sum(r['prompt_metadata']['input_tokens'] for r in verified),
        'generated_tokens':sum(c['generated_tokens'] for r in verified for c in r['candidates']),
        'token_cap_hits':sum(bool(c['hit_token_cap']) for r in verified for c in r['candidates']),
        'clipped_prompts':sum(bool(r['prompt_metadata'].get('clipped_fields')) for r in verified),
        'measured_candidate_seconds':sum(r['elapsed_seconds'] for r in timing_rows) if timing_rows else None,
        'timed_candidates':len(timing_rows), 'unsealed_records':unsealed,
        'cost_usd':None, 'scientific_scoring_complete':False,
        'limitations':['Input tokens counted once per task prompt; not billed API tokens.',
                      'Candidate timings, where available, exclude model loading and overlap across concurrent jobs.',
                      'No GPU-hour price supplied; dollar cost unknown.',
                      'Completion concerns generation checksums only, not scientific evaluation.']}
