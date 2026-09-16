"""Pinned RBR generated-bank inputs with explicit exploratory source splits."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

from pbpf.real_gate import html_to_text


PINS = {
    'python_train0.jsonl.gz': 'de4df37fe33753acc6310bd0c48d5e41',
    'python_train1.jsonl.gz': 'fbafa0786576e3b49bc795de2b304b4c',
    'python_train2.jsonl.gz': '57c614050be4745c2c2337e78e410a2a',
    'python_test0.jsonl.gz': '375f056565af619146c12e0e63e3974c',
    'tests_all.jsonl.gz': '6eaac2b6a535aa8b5213489c6511cb26',
}
DESCRIPTION_SHA256 = '8b631ae168ba84dce69c7d8e1b6c632256e0155858c2664c6493a5f001b45fdd'


def digest(path, algorithm='sha256'):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def rank(seed, kind, key):
    return hashlib.sha256(f'{seed}\0{kind}\0{key}'.encode()).digest()


def build_records(bugs, tests, descriptions, *, seed, development_components, primary_components):
    """Choose sources/tests/base bugs without execution or fixed-code screening."""
    cases = defaultdict(list)
    for row in tests:
        if len(row['input']) <= 8192 and len(row['output']) <= 4096:
            cases[row['problem_id']].append(row)
    grouped = defaultdict(list)
    seen = set()
    for row in bugs:
        key = str(row['id'])
        if key in seen:
            raise ValueError('duplicate official training bug identity')
        seen.add(key)
        source = row['problem_id']
        if (source in descriptions and descriptions[source].strip() and len(cases[source]) >= 10
                and isinstance(row['buggy_code'], str) and len(row['buggy_code']) <= 8192):
            grouped[source].append(row)
    ordered = sorted(grouped, key=lambda s: rank(seed, 'generated-bank-source', s))
    if min(development_components, primary_components) < 1 or development_components+primary_components >= len(ordered):
        raise ValueError('split counts must leave nonempty training sources')
    public, private, inventory = [], [], []
    for index, source in enumerate(ordered):
        split = ('primary' if index < primary_components else 'development'
                 if index < primary_components+development_components else 'train')
        base = min(grouped[source], key=lambda row: rank(seed, 'base-bug', row['id']))
        selected = sorted(cases[source], key=lambda row: rank(seed, 'test', row['id']))[:10]
        if len({str(t['id']) for t in selected}) != 10:
            raise ValueError('duplicate selected test identities')
        identity = {'task_id': 'RBR/'+source, 'source_component_id': source, 'split': split,
                    'protocol': 'RBR-generated-repair', 'base_bug_id': str(base['id'])}
        visible = [{'id': str(i), 'input': row['input'], 'expected': row['output']}
                   for i, row in enumerate(selected[:4])]
        public.append({**identity, 'task_text': descriptions[source][:6000],
                       'buggy_code': base['buggy_code'], 'visible_tests': visible})
        private.append({**identity, 'reference_code': base['fixed_code'],
                        'tests': [{'id': str(i), 'original_id': str(row['id']), 'input': row['input'],
                                   'expected': row['output'], 'hidden': i >= 4}
                                  for i, row in enumerate(selected)]})
        inventory.append({**identity, 'base_bug_candidates': len(grouped[source])})
    return public, private, {'source_components': len(ordered), 'source_inventory': inventory,
                             'component_counts': dict(Counter(r['split'] for r in public))}


def materialize(source_root, descriptions_archive, public_root, evaluator_root, *, seed=1701,
                development_components=400, primary_components=500):
    source_root, archive, public_root, evaluator_root = map(Path, (source_root, descriptions_archive, public_root, evaluator_root))
    a, b = public_root.resolve(), evaluator_root.resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('public/evaluator roots must be separate and nonnested')
    if public_root.exists() or evaluator_root.exists():
        raise FileExistsError('materializations are create-once')
    hashes = {}
    for name, checksum in PINS.items():
        path = source_root/name
        if digest(path, 'md5') != checksum:
            raise ValueError('official RBR checksum mismatch: '+name)
        hashes[name] = digest(path)
    if digest(archive) != DESCRIPTION_SHA256:
        raise ValueError('CodeNet description archive checksum mismatch')
    descriptions = {}
    # Read directly from the pinned archive; never trust mutable extracted HTML.
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            path = Path(member.name)
            if member.isfile() and path.suffix == '.html' and path.stem.startswith('p'):
                if path.stem in descriptions:
                    raise ValueError('duplicate archived problem description')
                descriptions[path.stem] = html_to_text(tar.extractfile(member).read().decode(errors='replace'))[:6000]
    def read(name):
        with gzip.open(source_root/name, 'rt') as stream:
            return [json.loads(line) for line in stream]
    bugs = [row for name in PINS if name.startswith('python_train') for row in read(name)]
    public, private, inventory = build_records(bugs, read('tests_all.jsonl.gz'), descriptions,
        seed=seed, development_components=development_components, primary_components=primary_components)
    for folder, rows in ((public_root, public), (evaluator_root, private)):
        folder.mkdir(parents=True, mode=0o700)
        with (folder/'tasks.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
    manifest = {'schema': 'apbpf-rbr-generated-materialization-v1', **inventory, 'seed': seed,
                'source_sha256': hashes, 'description_archive_sha256': DESCRIPTION_SHA256,
                'split_policy': 'hashed official-training source IDs; official test-only sources excluded',
                'filter_policy': 'raw length/test-count/description availability only; no correctness filtering',
                'claim_status': 'exploratory-predeclared', 'fresh_belief_training_required': True,
                'prior_exposure': 'source problems participated in earlier exploratory runs; no confirmatory claim',
                'public_tests': 4, 'evaluator_hidden_tests': 6,
                'public_tasks_sha256': digest(public_root/'tasks.jsonl'),
                'isolation': 'generator must mount only public_root; fixed code and future calls are evaluator-only'}
    (public_root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (evaluator_root/'manifest.json').write_text(json.dumps({**manifest,
        'evaluator_tasks_sha256': digest(evaluator_root/'tasks.jsonl')}, indent=2)+'\n')
    return manifest
