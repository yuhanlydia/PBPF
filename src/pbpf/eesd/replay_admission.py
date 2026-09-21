"""Hash-bound admission and atomic publication for the separate Replay study."""
from __future__ import annotations
import ast
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter
from .replay_materialization import DOMAINS, MODEL_KEYS


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_file(path, seal):
    path = Path(path)
    expected = seal if isinstance(seal, str) else seal['sha256']
    if digest(path) != expected or (isinstance(seal, dict) and path.stat().st_size != seal['bytes']):
        raise ValueError(f'checksum/size mismatch: {path}')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def verify_inputs(root, joint_path, public_path, private_path, token_path, results_path):
    """Check the completed inventories, full graph membership and prompt audit.

    This checks provenance receipts; it does not independently repeat upstream
    overlap detection or tokenizer execution. Those remain separately reviewed.
    """
    root = Path(root).resolve()
    joint_path, public_path, private_path, token_path, results_path = map(
        lambda p: Path(p).resolve(), (joint_path, public_path, private_path, token_path, results_path))
    joint = json.loads(joint_path.read_text())
    token = json.loads(token_path.read_text())
    if joint.get('schema') != 'eesd-replay-joint-inventory-receipt-v1' or joint.get('status') != 'candidate-preparation-only-not-split-not-queued':
        raise ValueError('joint receipt is not complete/admissible')
    if token.get('schema') != 'eesd-replay-token-probe-v1' or token.get('status') != 'cpu-probe-not-admitted-not-generated':
        raise ValueError('token receipt is not complete/admissible')
    base = joint_path.parent
    if public_path != base/'joint-public-candidates.jsonl' or private_path != base/'joint-evaluator-candidates.jsonl':
        raise ValueError('candidate paths differ from bound inventory')
    required = {'joint-public-candidates.jsonl', 'joint-evaluator-candidates.jsonl', 'components.jsonl', 'joint-graph.json', 'candidate-rows.jsonl', 'joint-edges.jsonl'}
    if not required <= set(joint['outputs']):
        raise ValueError('incomplete joint output inventory')
    for name, seal in joint['outputs'].items():
        if Path(name).name != name:
            raise ValueError('joint output must be a basename')
        verify_file(base/name, seal)
    spec = 'docs/EESD_REPLAY_EXTENSION_SPEC_20260920.md'
    if spec not in joint['input_sha256']:
        raise ValueError('missing prospective spec binding')
    for name, seal in joint['input_sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('input path outside repository')
        verify_file(path, seal)
    verify_file(base/'build_inventory.py', joint['source_sha256'])
    binding = token['joint_receipt']
    if Path(binding['path']).resolve() != joint_path:
        raise ValueError('token audit belongs to another joint receipt')
    verify_file(joint_path, binding['sha256'])
    if binding['graph_sha256'] != joint['outputs']['joint-graph.json']['sha256']:
        raise ValueError('token graph mismatch')
    if Path(token['input']).resolve() != public_path:
        raise ValueError('token public input mismatch')
    verify_file(public_path, token['input_sha256'])
    verify_file(results_path, token['rows_sha256'])
    probe = root/'runs/eesd-setup/replay-token-probe/probe.py'
    verify_file(probe, token['tool_sha256'])
    constants = {}
    for node in ast.parse(probe.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {'TEMPLATE','MODELS','POLICY'}:
            constants[node.targets[0].id] = ast.literal_eval(node.value)
    template = constants['TEMPLATE']
    if hashlib.sha256(template.encode()).hexdigest() != token['template_sha256'] or token['policy'] != constants['POLICY']:
        raise ValueError('prompt policy/template mismatch')
    if token.get('input_cap') != 4096 or token.get('no_truncation') is not True or token.get('proposed_max_new_tokens') != 1024:
        raise ValueError('generation limits differ from prospective policy')
    expected_models = {key:(model,revision) for key,model,revision in constants['MODELS']}
    actual_models = {m['key']:(m['model_id'],m['revision']) for m in token['models']}
    if set(expected_models) != MODEL_KEYS or actual_models != expected_models or len(token['models']) != 4:
        raise ValueError('tokenizer model identity mismatch')
    for model in token['models']:
        if hashlib.sha256(model['chat_template'].encode()).hexdigest() != model['chat_template_sha256'] or not model['tokenizer_files_sha256']:
            raise ValueError('missing/inconsistent tokenizer provenance')
    public, private, tokens = map(read_rows, (public_path, private_path, results_path))
    identity = lambda row:(row['domain'],row['source_id'],row['task_id'])
    components = read_rows(base/'components.jsonl')
    if len({c['source_id'] for c in components}) != len(components):
        raise ValueError('duplicate graph source')
    expected = set()
    quarantine = set()
    for component in components:
        if component['quarantined']:
            quarantine.add(component['source_id']); continue
        for domain, tasks in component['eligible_member_ids'].items():
            for task in tasks:
                expected.add((domain,component['source_id'],task))
    for rows in (public,private,tokens):
        actual = {identity(row) for row in rows}
        if actual != expected or len(actual) != len(rows):
            raise ValueError('joint public/private/token population mismatch')
    if token['input_rows'] != len(public) or binding['verified_complete_public_members'] != len(public):
        raise ValueError('token audit row count mismatch')
    by_id = {row['task_id']:row for row in tokens}
    for row in public:
        observations = [f'Observation {i}\nInput: '+json.dumps(t['input'],ensure_ascii=False)+'\nOutput: '+json.dumps(t['output'],ensure_ascii=False) for i,t in enumerate(row['visible_tests'],1)]
        messages = [{'role':'user','content':template.format(statement=row['statement'], observations='\n\n'.join(observations))}]
        audit = by_id[row['task_id']]
        if hashlib.sha256(compact(messages)).hexdigest() != audit['message_sha256']:
            raise ValueError('public prompt differs from tokenized message')
        for model in audit['models'].values():
            for key in ('rendered_prompt_sha256','token_ids_sha256'):
                value = model.get(key)
                if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                    raise ValueError('missing token/prompt digest')
    evidence = {'joint_receipt':{'path':str(joint_path),'sha256':digest(joint_path)},
        'token_receipt':{'path':str(token_path),'sha256':digest(token_path)},
        'spec_sha256':joint['input_sha256'][spec], 'prompt_template_sha256':token['template_sha256'],
        'models':token['models'], 'input_population':len(public),
        'limitations':['Data admission only; no program generation/execution or experimental scores.',
            'Public-example extraction cannot certify all original examples were identified.',
            'Uses common Replay executor semantics, not official benchmark accuracy.']}
    return public, private, tokens, quarantine, evidence


def rename_new_directory(source, target):
    """Linux atomic RENAME_NOREPLACE; unsupported hosts fail closed."""
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(target))


def publish_bundle(output, public, private, token_rows, evidence):
    """Publish complete 2-domain 200/500 views in one directory rename."""
    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(output)
    expected = {(domain,split):count for domain in DOMAINS for split,count in [('development',200),('primary',500)]}
    for rows in (public,private):
        if Counter((r['domain'],r['split']) for r in rows) != expected:
            raise ValueError('requires exactly 200 development and 500 primary per domain')
        if len({r['task_id'] for r in rows}) != 1400 or len({r['source_component_id'] for r in rows}) != 1400:
            raise ValueError('duplicate selected task/source across domains')
    identity = lambda rows:{(r['domain'],r['split'],r['task_id'],r['source_component_id']) for r in rows}
    if identity(public) != identity(private):
        raise ValueError('public/evaluator identities differ')
    expected_tokens = {(r['domain'],r['source_component_id'],r['task_id']) for r in public}
    actual_tokens = {(r['domain'],r['source_id'],r['task_id']) for r in token_rows}
    if actual_tokens != expected_tokens or len(token_rows) != len(expected_tokens):
        raise ValueError('selected token audit population mismatch')
    for row in token_rows:
        if set(row.get('models', {})) != MODEL_KEYS:
            raise ValueError('selected token audit requires all four models')
        for model in row['models'].values():
            count = model.get('input_tokens')
            if type(count) is not int or not 1 <= count <= 4096:
                raise ValueError('selected token audit input exceeds limits')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name+'.staging-',dir=output.parent))
    seals = {}
    def write(name, data):
        path=staging/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        seals[name]={'sha256':digest(path),'bytes':len(data)}
    try:
        for domain in DOMAINS:
            for view, rows in [('public',public),('evaluator',private)]:
                selected = [r for r in rows if r['domain']==domain]
                name=f'{domain}/{view}/tasks.jsonl'
                write(name,b''.join(compact(r)+b'\n' for r in selected))
                manifest={'schema':f'eesd-replay-{view}-v1','domain':domain,'task_kind':'stdin_synthesis',
                    'counts':{'development':200,'primary':500},'tasks':seals[name]}
                write(f'{domain}/{view}/manifest.json',compact(manifest)+b'\n')
        write('selected-token-audit.jsonl',b''.join(compact(r)+b'\n' for r in token_rows))
        receipt={'schema':'eesd-replay-admission-v1','status':'data-admitted-not-generated-not-scored',
            'evidence':evidence,'outputs':seals.copy()}
        (staging/'admission.json').write_bytes(compact(receipt)+b'\n')
        if output.exists():
            raise FileExistsError(output)
        rename_new_directory(staging,output)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
