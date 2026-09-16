"""Forward evaluator caches and checkpoints through exact direct dependencies."""
import json
from pathlib import Path
import shutil

from .codearc_bank import file_sha
from .stage_prediction import DOMAINS


def create_bundle(io, training):
    target = io.outputs/'assessment-bundle'; target.mkdir()
    index = json.loads(io.artifact('execution_cache', 'execution-index.json').read_text())
    manifest = {'schema': 'apbpf-assessment-bundle-v1', 'fingerprint': io.request['fingerprint'],
                'origin_dependencies': io.request['dependencies'], 'domains': {},
                'visibility': 'evaluator-only; never mount in generator sandbox'}
    for domain in DOMAINS:
        entry = index['training_caches'][domain]['full']
        source = io.artifact('execution_cache', entry['path'])
        if file_sha(source) != entry['sha256']:
            raise ValueError('cache binding mismatch while forwarding')
        cache = target/f'{domain}-cache.json'; shutil.copyfile(source, cache)
        if file_sha(cache) != entry['sha256']:
            raise ValueError('copied cache digest differs')
        cells = []
        for seed in io.config['protocol']['seeds']:
            cell = training[domain, seed]
            if cell['entry']['cache_sha256'] != entry['sha256']:
                raise ValueError('checkpoint training cache differs from forwarded cache')
            source = cell['checkpoints']['belief']
            checkpoint = target/f'{domain}-seed{seed}.pt'; shutil.copyfile(source, checkpoint)
            checksum = cell['report']['arms']['belief']['checkpoint_sha256']
            if file_sha(checkpoint) != checksum:
                raise ValueError('copied belief checkpoint differs')
            report = target/f'{domain}-seed{seed}-training.json'
            original = io.artifact('train_belief', cell['entry']['directory']+'/training.json')
            shutil.copyfile(original, report)
            if file_sha(report) != cell['training_report_sha256']:
                raise ValueError('copied training report differs')
            cells.append({'seed': seed, 'checkpoint': checkpoint.name, 'checkpoint_sha256': checksum,
                          'training_report': report.name, 'training_report_sha256': file_sha(report)})
        manifest['domains'][domain] = {'cache': cache.name, 'cache_sha256': file_sha(cache), 'models': cells}
    (target/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def forward_bundle(io, dependency):
    source = io.directory(dependency, 'assessment-bundle')
    value = json.loads((source/'manifest.json').read_text())
    if value['schema'] != 'apbpf-assessment-bundle-v1' or value['fingerprint'] != io.request['fingerprint']:
        raise ValueError('assessment bundle belongs to another run')
    target = io.outputs/'assessment-bundle'
    shutil.copytree(source, target)
    for path in source.rglob('*'):
        if path.is_file() and file_sha(path) != file_sha(target/path.relative_to(source)):
            raise ValueError('assessment bundle changed while forwarding')
    return value


def load_bundle(io, dependency):
    directory = io.directory(dependency, 'assessment-bundle')
    manifest = json.loads((directory/'manifest.json').read_text())
    if (manifest['schema'] != 'apbpf-assessment-bundle-v1' or manifest['fingerprint'] != io.request['fingerprint']
            or set(manifest['domains']) != set(DOMAINS)):
        raise ValueError('assessment bundle domain or run identity differs')
    def artifact(name, checksum):
        if Path(name).name != name:
            raise ValueError('bundle artifacts must be direct declared files')
        path = io.artifact(dependency, 'assessment-bundle/'+name)
        if file_sha(path) != checksum:
            raise ValueError('bundle artifact digest mismatch')
        return path
    result = {}
    for domain, entry in manifest['domains'].items():
        cache = artifact(entry['cache'], entry['cache_sha256'])
        models = {}
        for cell in entry['models']:
            seed = cell['seed']
            if seed in models:
                raise ValueError('duplicate seed in assessment bundle')
            checkpoint = artifact(cell['checkpoint'], cell['checkpoint_sha256'])
            report = json.loads(artifact(cell['training_report'], cell['training_report_sha256']).read_text())
            if (report['config']['seed'] != seed or report['config']['protocol'] != io.config['protocol']
                    or report['dataset'] != DOMAINS[domain]
                    or report['arms']['belief']['checkpoint_sha256'] != cell['checkpoint_sha256']):
                raise ValueError('bundle model report identity mismatch')
            models[seed] = {'checkpoint': checkpoint, 'report': report}
        if set(models) != set(io.config['protocol']['seeds']):
            raise ValueError('all fixed seeds required in assessment bundle')
        result[domain] = {'cache': cache, 'models': models}
    return result
