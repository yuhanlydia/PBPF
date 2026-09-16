"""Import declared public generation bytes, never another run's evaluations."""
import hashlib
import json
from pathlib import Path

from .codearc_bank import file_sha, load_bank


ROW_FIELDS = {'task_id', 'source_component_id', 'split', 'seed', 'prompt_sha256',
              'candidates', 'elapsed_seconds', 'prompt_metadata', 'input_tokens', 'public_prompt_bounded'}
CANDIDATE_FIELDS = {'candidate_id', 'code', 'raw_completion', 'generated_tokens',
                    'hit_token_cap', 'extraction_changed'}
RUN_FIELDS = {'schema', 'family', 'model', 'revision', 'source_sha256', 'prompt_source_sha256',
              'inventory_source_sha256', 'public_tasks_sha256', 'public_manifest_sha256', 'split',
              'seed', 'offset', 'candidates', 'components', 'task_ids', 'source_component_ids',
              'max_input_tokens', 'max_new_tokens', 'temperature', 'top_p', 'prompt_policy',
              'population_selection', 'claim_status', 'code_reextraction', 'model_proof_sha256'}


def import_domain_banks(entries, public_root, output, *, domain, model, seed):
    """Bind full train/dev/primary candidate inventories to public task bytes.

    Every candidate file is copied byte-for-byte; excluded side files may include
    old execution results. The import does not assert that generation is fresh.
    """
    public_root, output = Path(public_root), Path(output)
    public_hash = file_sha(public_root/'tasks.jsonl')
    manifest = json.loads((public_root/'manifest.json').read_text())
    if public_hash != manifest['public_tasks_sha256']:
        raise ValueError('public task checksum mismatch')
    # Match generation's first representative in materialized source order.
    # CodeARC has two additional task IDs sharing already represented AST sources.
    grouped = {}
    with (public_root/'tasks.jsonl').open() as stream:
        for row in map(json.loads, stream):
            previous = grouped.setdefault(row['source_component_id'], row)
            if previous['split'] != row['split']:
                raise ValueError('one public source cannot cross splits')
    tasks = {r['task_id']: r for r in grouped.values()}
    if not entries:
        raise ValueError('complete generation inventory required')
    seen_tasks, seen_sources, seen_candidates = set(), set(), set()
    plans = []
    for i, entry in enumerate(entries):
        if set(entry) != {'path', 'complete_sha256'}:
            raise ValueError('cache entry requires only a path and frozen complete checksum')
        source = Path(entry['path'])
        if not source.is_absolute() or file_sha(source/'complete.json') != entry['complete_sha256']:
            raise ValueError('generation cache must have an absolute path and exact checksum')
        run, rows, checksum = load_bank(source)
        if (set(run) - RUN_FIELDS or run['schema'] != f'apbpf-{domain}-generation-v1'
                or run['model'] != model['id'] or run['revision'] != model['revision']
                or run['seed'] != seed or run['candidates'] != 8
                or run['temperature'] != .8 or run['top_p'] != .95
                or run['max_new_tokens'] != (1024 if domain == 'rbr' else 512)
                or run['public_tasks_sha256'] != public_hash):
            raise ValueError('cache protocol/model/public-data identity mismatch')
        if domain == 'codearc':
            extraction = run.get('code_reextraction', {})
            if extraction.get('source_sha256') != file_sha(Path(__file__).with_name('code_extraction.py')):
                raise ValueError('CodeARC requires uniform declared syntax-based extraction')
        for row in rows:
            task = tasks.get(row['task_id'])
            if (set(row) - ROW_FIELDS or task is None or row['task_id'] in seen_tasks
                    or row['source_component_id'] in seen_sources
                    or task['source_component_id'] != row['source_component_id']
                    or task['split'] != row['split']):
                raise ValueError('cache source inventory differs from full public population')
            seen_tasks.add(row['task_id']); seen_sources.add(row['source_component_id'])
            for candidate in row['candidates']:
                if (set(candidate) - CANDIDATE_FIELDS or candidate['candidate_id'] in seen_candidates
                        or not isinstance(candidate.get('code'), str)
                        or not isinstance(candidate.get('raw_completion'), str)):
                    raise ValueError('only public generation fields and unique candidates are allowed')
                seen_candidates.add(candidate['candidate_id'])
        complete = json.loads((source/'complete.json').read_text())
        files = {'run.json': complete['run_sha256'], 'complete.json': checksum, **complete['files']}
        plans.append((source, f'{domain}-bank-{i:03d}', run, checksum, files))
    if seen_tasks != set(tasks):
        raise ValueError('cache must cover every public train/development/primary source exactly once')
    imported = []
    for source, name, run, checksum, files in plans:
        target = output/name
        target.mkdir(parents=True, exist_ok=False)
        for filename, expected in files.items():
            path = source/filename
            if path.is_symlink():
                raise ValueError('candidate cache files cannot be symlinks')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError('candidate cache changed during import')
            (target/filename).write_bytes(data)
        load_bank(target)
        imported.append({'domain': domain, 'directory': name, 'split': run['split'],
                         'complete_sha256': checksum, 'groups': len(run['task_ids']),
                         'source_generation_sha256': run['source_sha256']})
    return imported


def import_candidate_manifest(path, expected_sha256, outputs, *, config):
    path, outputs = Path(path), Path(outputs)
    manifest_bytes = path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != expected_sha256:
        raise ValueError('declared candidate manifest checksum mismatch')
    manifest = json.loads(manifest_bytes)
    if (set(manifest) != {'schema', 'domains'} or manifest['schema'] != 'apbpf-public-generation-cache-v1'
            or set(manifest['domains']) != {'rbr', 'codearc'}):
        raise ValueError('candidate manifest must declare both full domain inventories')
    primary = [m for m in config['models'].values() if m['role'] == 'primary']
    if len(primary) != 1:
        raise ValueError('one pinned primary model required')
    imported = []
    for domain in ('rbr', 'codearc'):
        imported += import_domain_banks(manifest['domains'][domain], outputs/f'{domain}-public',
            outputs/'candidate-banks', domain=domain, model=primary[0], seed=config['protocol']['seeds'][0])
    receipt = {'schema': 'apbpf-public-generation-import-v1', 'manifest_sha256': expected_sha256,
               'fresh_generation': False, 'evaluation_results_imported': False,
               'scope': 'exploratory cache reuse; full frozen public population; no candidate selection',
               'banks': imported}
    (outputs/'candidate-import.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (outputs/'candidate-cache-manifest.json').write_bytes(manifest_bytes)
    return receipt
