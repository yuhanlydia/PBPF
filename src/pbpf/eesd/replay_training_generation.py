"""Validate the disjoint Replay training reserve before candidate generation."""
import hashlib
import json
from pathlib import Path

from . import replay_prompt
from . import replay_generation as base


def load_training_split(bundle, admission_sha256, domain, family):
    bundle = Path(bundle).resolve()
    if domain not in base.DOMAINS or family not in ('qwen25_7b', 'deepseek_6p7b'):
        raise ValueError('unsupported training domain/family')
    admission_path = bundle / 'admission.json'
    if not base._digest(admission_sha256) or base.sha(admission_path) != admission_sha256:
        raise ValueError('training admission checksum mismatch')
    admission = base._json(admission_path)
    if (admission.get('schema') != 'eesd-replay-training-reserve-admission-v1'
            or admission.get('status') != 'training-data-admitted-not-generated-not-scored'
            or admission.get('seed') != 1701
            or admission.get('assessment_source_overlap') != 0):
        raise ValueError('training admission status/scope mismatch')
    assessment = admission['assessment_admission']
    assessment_path = Path(assessment['path']).resolve()
    if base.sha(assessment_path) != assessment['sha256']:
        raise ValueError('assessment admission changed')
    evidence, outputs = admission['evidence'], admission['outputs']
    token_path = Path(evidence['token_receipt']['path']).resolve()
    if base.sha(token_path) != evidence['token_receipt']['sha256']:
        raise ValueError('token receipt changed')
    token = base._json(token_path)
    if (token.get('schema') != 'eesd-replay-token-probe-v1'
            or token.get('status') != 'cpu-probe-not-admitted-not-generated'
            or token.get('policy') != replay_prompt.POLICY):
        raise ValueError('token policy differs')
    template_path = token_path.parent / 'prompt-template.txt'
    template_sha = hashlib.sha256(replay_prompt.TEMPLATE.encode()).hexdigest()
    if not (base.sha(template_path) == token['template_sha256']
            == evidence['prompt_template_sha256'] == template_sha):
        raise ValueError('prompt template differs')
    models = token['models']
    if models != evidence['models'] or {m['key'] for m in models} != set(base.MODELS):
        raise ValueError('model inventory differs')
    model = next(m for m in models if m['key'] == family)
    if (model['model_id'], model['revision']) != base.MODELS[family]:
        raise ValueError('model revision differs')
    proof_path = base.MODEL_PROOF_ROOT / f'{family}-verified.json'
    proof = base._json(proof_path)
    if (proof.get('status') != 'downloaded-and-sha256-verified'
            or (proof.get('model_id'), proof.get('revision')) != base.MODELS[family]):
        raise ValueError('model weight proof differs')
    for file in proof['files']:
        if file['file'].endswith('.safetensors'):
            path = Path(proof['snapshot']) / file['file']
            if path.stat().st_size != file['bytes']:
                raise ValueError('model weight size differs')
    token_audit = bundle / 'selected-token-audit.jsonl'
    base._verify(token_audit, outputs[token_audit.name])
    audits = {}
    for row in base._rows(token_audit):
        task_id = row['task_id']
        if task_id in audits or row['models'][family]['input_tokens'] > 4096:
            raise ValueError('duplicate or overlength training token audit')
        audits[task_id] = row
    all_rows, sources = {}, set()
    public_bindings = {}
    assessment_root = assessment_path.parent
    assessment_receipt = base._json(assessment_path)
    for current_domain in base.DOMAINS:
        manifest_path = bundle / current_domain / 'public/manifest.json'
        tasks_path = bundle / current_domain / 'public/tasks.jsonl'
        for path in (manifest_path, tasks_path):
            base._verify(path, outputs[path.relative_to(bundle).as_posix()])
        manifest = base._json(manifest_path)
        if (manifest.get('schema') != 'eesd-replay-public-train-v1'
                or manifest.get('domain') != current_domain
                or manifest.get('counts') != {'train': 200}
                or manifest.get('tasks') != outputs[tasks_path.relative_to(bundle).as_posix()]):
            raise ValueError('training public manifest mismatch')
        rows = base._rows(tasks_path)
        if len(rows) != 200:
            raise ValueError('training population differs')
        old_path = assessment_root / current_domain / 'public/tasks.jsonl'
        if base.sha(old_path) != assessment_receipt['outputs'][f'{current_domain}/public/tasks.jsonl']['sha256']:
            raise ValueError('assessment public population changed')
        old_sources = {row['source_component_id'] for row in base._rows(old_path)}
        for row in rows:
            if (set(row) != {'task_id', 'source_component_id', 'split', 'domain', 'statement', 'visible_tests'}
                    or row['domain'] != current_domain or row['split'] != 'train'
                    or row['source_component_id'] in sources or row['source_component_id'] in old_sources
                    or row['task_id'] in all_rows):
                raise ValueError('training row identity/assessment overlap')
            if not isinstance(row['statement'], str) or not row['statement'].strip():
                raise ValueError('empty training statement')
            if not isinstance(row['visible_tests'], list) or len(row['visible_tests']) != 4:
                raise ValueError('training row requires four public tests')
            sources.add(row['source_component_id'])
            all_rows[row['task_id']] = row
        public_bindings[current_domain] = {
            'public_tasks_sha256': base.sha(tasks_path),
            'public_manifest_sha256': base.sha(manifest_path),
        }
    if len(all_rows) != 400 or set(audits) != set(all_rows):
        raise ValueError('training public/token population mismatch')
    for task_id, row in all_rows.items():
        audit = audits[task_id]
        if audit['domain'] != row['domain'] or audit['source_id'] != row['source_component_id']:
            raise ValueError('training token audit row mismatch')
    selected = [row for row in all_rows.values() if row['domain'] == domain]
    return {'rows': selected, 'audits': {row['task_id']: audits[row['task_id']] for row in selected},
            'model': model, 'domain': domain, 'split': 'train', 'family': family,
            'bindings': {'admission_sha256': admission_sha256,
                         'assessment_admission_sha256': assessment['sha256'],
                         'token_receipt_sha256': base.sha(token_path),
                         'selected_token_audit_sha256': base.sha(token_audit),
                         'template_sha256': template_sha,
                         'spec_sha256': evidence['spec_sha256'],
                         'model_weight_proof_sha256': base.sha(proof_path),
                         **public_bindings[domain]}}
