"""Declared-stage I/O that never opens unrelated evaluator artifacts."""
import json
import os
from pathlib import Path
import subprocess

from .codearc_bank import file_sha
from .config import digest
from .stages import DEPENDENCIES


class WorkerIO:
    def __init__(self, stage, sources):
        self.request_path = Path(os.environ['APBPF_REQUEST']).resolve()
        self.request = json.loads(self.request_path.read_text())
        request = self.request
        if (request['stage'] != stage or request['fingerprint'] != digest(request['config'])
                or set(request['dependencies']) != set(DEPENDENCIES[stage])
                or set(request['inputs']) != set(DEPENDENCIES[stage])):
            raise ValueError('invalid stage request or dependency set')
        if request['confirmatory'] or request['claim_status'] != 'exploratory-predeclared':
            raise ValueError('these cache-reuse workers require permanently exploratory execution')
        self.config = request['config']
        self.root = Path(self.config['site']['working_directory']).resolve()
        self.outputs = Path(request['outputs_directory']).resolve()
        if (self.outputs != Path(os.environ['APBPF_OUTPUTS']).resolve()
                or self.outputs != self.request_path.parent/'outputs'):
            raise ValueError('outputs must belong to this exact stage attempt')
        self.sources = sources + ['src/pbpf/apbpf/worker_io.py', 'src/pbpf/apbpf/codearc_bank.py']
        self.check_sources()
        self.dependencies = {}
        run_root = self.request_path.parent.parents[2]
        for name, item in request['inputs'].items():
            path = Path(item['result']).resolve()
            if (path.name != 'result.json' or not path.is_relative_to(run_root/'stages')
                    or item['checksum'] != request['dependencies'][name]):
                raise ValueError('dependency must be an exact same-run input')
            complete = json.loads((path.parent/'complete.json').read_text())
            if digest(complete) != item['checksum'] or file_sha(path) != complete['files']['result.json']:
                raise ValueError('dependency completion/result checksum mismatch')
            value = json.loads(path.read_text())
            if value['stage'] != name or value['fingerprint'] != request['fingerprint']:
                raise ValueError('dependency stage/fingerprint mismatch')
            artifacts = {a['path']: a['sha256'] for a in value['artifacts']}
            if any(complete['files'].get(p) != h for p, h in artifacts.items()):
                raise ValueError('dependency artifact inventory differs from completion receipt')
            self.dependencies[name] = (path.parent, artifacts)

    def check_sources(self):
        for name in self.sources:
            if file_sha(self.root/name) != self.config['source_hashes'][name]:
                raise ValueError('worker source differs from frozen request: '+name)

    def artifact(self, dependency, relative):
        base, inventory = self.dependencies[dependency]
        relative = Path('outputs')/relative
        path = base/relative
        if ('..' in relative.parts or relative.as_posix() not in inventory or path.is_symlink()
                or not path.resolve().is_relative_to((base/'outputs').resolve())
                or file_sha(path) != inventory[relative.as_posix()]):
            raise ValueError('requested artifact is not an intact declared input')
        return path

    def directory(self, dependency, relative):
        base, inventory = self.dependencies[dependency]
        prefix = (Path('outputs')/relative).as_posix()+'/'
        names = {name for name in inventory if name.startswith(prefix)}
        path = base/'outputs'/relative
        if not names or path.is_symlink():
            raise ValueError('requested input directory is not declared')
        actual = {f.relative_to(base).as_posix() for f in path.rglob('*') if f.is_file() or f.is_symlink()}
        if actual != names:
            raise ValueError('input directory has missing or undeclared files')
        for name in names:
            self.artifact(dependency, str(Path(name).relative_to('outputs')))
        return path

    def execute(self, name, command):
        self.check_sources()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        env.pop('PYTHONPATH', None); env.pop('PYTHONHOME', None)
        (self.outputs/f'{name}-command.json').write_text(json.dumps(command, indent=2)+'\n')
        with (self.outputs/f'{name}.log').open('x') as log:
            subprocess.run(command, cwd=self.root, env=env, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def finish(self, summary, *, gate=None):
        self.check_sources()
        artifacts = [{'path': str(p.relative_to(self.outputs.parent)), 'sha256': file_sha(p)}
                     for p in sorted(self.outputs.rglob('*')) if p.is_file()]
        result = {'schema': 'apbpf-stage-result-v1', 'stage': self.request['stage'],
                  'fingerprint': self.request['fingerprint'], 'dependencies': self.request['dependencies'],
                  'summary': summary, 'artifacts': artifacts}
        if gate is not None:
            result['gate'] = gate
        with Path(os.environ['APBPF_RESULT']).open('x') as stream:
            json.dump(result, stream, indent=2)


def bank_inventory(io):
    path = io.artifact('materialize', 'candidate-import.json')
    receipt = json.loads(path.read_text())
    if (receipt['schema'] != 'apbpf-public-generation-import-v1'
            or receipt['evaluation_results_imported'] is not False):
        raise ValueError('requires a declared public generation import without evaluation results')
    return receipt['banks']
