#!/usr/bin/env python3
"""Run locked official EvalPlus CLIs only inside a fail-closed bubblewrap sandbox.

Input and infrastructure failures are never scored as model failures. The output
is create-once, separate from the transfer artifact; only its work subdirectory
is exposed writable to the sandbox. --probe-only never mounts candidate data.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from pbpf.eesd.evalplus_public import EVALPLUS_VERSION, RELEASES, load_public_dataset, sha


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def output_file(work, name):
    path=Path(work)/name
    if path.is_symlink() or not path.is_file() or path.resolve().parent != Path(work).resolve():
        raise ValueError("sandbox output must be a regular file within its work directory")
    return path


def sample_coverage(path, expected):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if any(set(r) != {'task_id','solution'} or not isinstance(r['solution'], str) for r in rows):
        raise ValueError('invalid official samples schema')
    ids = [r['task_id'] for r in rows]
    if len(ids) != len(expected) or set(ids) != set(expected):
        raise ValueError('samples require exactly one row per locked task')


def validate_inputs(*, transfer_root, transfer_report_sha256, public_root,
                    public_manifest_sha256, private_root):
    report_path = Path(transfer_root)/'report.json'
    if sha(report_path) != transfer_report_sha256:
        raise ValueError('transfer report checksum mismatch')
    report = json.loads(report_path.read_text())
    if report.get('schema') != 'eesd-evalplus-transfer-v1':
        raise ValueError('unsupported transfer report')
    dataset = report['dataset']
    _, binding = load_public_dataset(public_root, dataset, public_manifest_sha256)
    if report.get('data_lock') != binding or report.get('tasks') != binding['tasks']:
        raise ValueError('transfer data binding mismatch')
    samples = Path(transfer_root)/'samples.jsonl'
    if sha(samples) != report.get('samples_sha256'):
        raise ValueError('transfer samples checksum mismatch')
    sample_coverage(samples,binding['task_ids'])
    private = Path(private_root)
    manifest = json.loads((private/'manifest.json').read_text())
    if (manifest.get('schema') != 'eesd-evalplus-private-v1'
            or manifest.get('evalplus_version') != EVALPLUS_VERSION
            or set(manifest.get('datasets',{})) != set(RELEASES)):
        raise ValueError('private manifest/version mismatch')
    if sha(private/'download-receipt.json') != binding['download_receipt_sha256']:
        raise ValueError('private receipt mismatch')
    raw_hashes = {}
    for name, (_, version) in RELEASES.items():
        _, other = load_public_dataset(public_root,name,public_manifest_sha256)
        entry = manifest['datasets'][name]
        digest = sha(private/f'{name}.jsonl')
        if (entry.get('file') != f'{name}.jsonl' or entry.get('version') != version
                or digest != entry.get('sha256') or digest != other['raw_source_sha256']):
            raise ValueError('private raw release/checksum mismatch')
        raw_hashes[name] = digest
    return dict(dataset=dataset, data_binding=binding, private_raw_sha256=raw_hashes,
                private_manifest_sha256=sha(private/'manifest.json'),
                transfer_report_sha256=transfer_report_sha256, samples_sha256=sha(samples))


def runtime_info():
    if importlib.metadata.version('evalplus') != EVALPLUS_VERSION:
        raise ValueError('official EvalPlus 0.3.1 required')
    venv = Path(sys.prefix).absolute()
    base = Path(sys.base_prefix).resolve()
    if venv == base or not (venv/'pyvenv.cfg').is_file():
        raise ValueError('dedicated virtual environment required')
    package = Path(importlib.metadata.distribution('evalplus').locate_file('evalplus')).resolve()
    if not package.is_relative_to(venv.resolve()):
        raise ValueError('EvalPlus must be installed inside the mounted virtual environment')
    hashes = {str(p.relative_to(venv)):sha(p) for p in sorted(package.rglob('*.py'))}
    for name in ('sanitize','evaluate'):
        path=venv/'bin'/f'evalplus.{name}'
        hashes[str(path.relative_to(venv))]=sha(path)
    return dict(venv=str(venv),base=str(base),evalplus_version=EVALPLUS_VERSION,
                official_source_sha256=hashes)


def sandbox_command(runtime, work, stage, *, dataset=None, private_root=None):
    if stage not in {'probe','sanitize','evaluate'}:
        raise ValueError('stage not allowlisted')
    if stage != 'probe' and (dataset not in RELEASES or private_root is None):
        raise ValueError('locked dataset required')
    venv=Path(runtime['venv']);base=Path(runtime['base'])
    if venv.name != '.venv' or base in (Path('/'),Path('/root'),Path('/usr')):
        raise ValueError('broad runtime mounts forbidden')
    command=['/usr/bin/bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv']
    for path in ('/usr/bin','/usr/lib','/usr/lib64','/lib','/lib64','/etc/ld.so.cache','/etc/alternatives'):
        if Path(path).exists(): command += ['--ro-bind',path,path]
    for path in (venv,base): command += ['--ro-bind',str(path),str(path)]
    command += ['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
                '--dir','/tmp/home','--dir','/tmp/cache',
                '--bind',str(Path(work).resolve()),'/evaluation','--chdir','/evaluation']
    env={'PATH':f'{venv}/bin:/usr/bin','LANG':'C.UTF-8','HOME':'/tmp/home',
         'XDG_CACHE_HOME':'/tmp/cache','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
         'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    if stage != 'probe':
        for name,variable in [('humaneval','HUMANEVAL_OVERRIDE_PATH'),('mbpp','MBPP_OVERRIDE_PATH')]:
            target=f'/data/{name}.jsonl'
            command += ['--ro-bind',str((Path(private_root)/f'{name}.jsonl').resolve()),target]
            env[variable]=target
    if stage != 'probe':
        # Candidates cannot rewrite the submitted samples while being evaluated.
        sample_name = 'samples.jsonl' if stage == 'sanitize' else 'samples-sanitized.jsonl'
        command += ['--ro-bind',str(Path(work).resolve()/sample_name),f'/evaluation/{sample_name}']
    for key,value in env.items(): command += ['--setenv',key,value]
    command += ['--',str(venv/'bin/python'),'-I','-B']
    if stage == 'probe':
        command += ['-c',"import importlib.metadata; assert importlib.metadata.version('evalplus') == '0.3.1'; print('EESD_SANDBOX_OK')"]
    elif stage == 'sanitize':
        command += [str(venv/'bin/evalplus.sanitize'),'--samples','/evaluation/samples.jsonl']
    else:
        command += [str(venv/'bin/evalplus.evaluate'),'--dataset',dataset,
                    '--samples','/evaluation/samples-sanitized.jsonl','--parallel','1']
    return command


def run_command(command, log, timeout):
    with Path(log).open('xb') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                 env={'PATH':'/usr/bin','LANG':'C.UTF-8'},start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return 124
        finally:
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.wait()


def summarize(result, expected):
    rows=result.get('eval')
    if not isinstance(rows,dict) or set(rows)!=set(expected) or len(rows)!=len(expected):
        raise ValueError('official result coverage mismatch')
    base=plus=0
    for task, samples in rows.items():
        if not isinstance(samples,list) or len(samples)!=1:
            raise ValueError('exactly one result per task required')
        row=samples[0]
        if row.get('task_id')!=task or any(row.get(k) not in {'pass','fail','timeout'}
                                         for k in ('base_status','plus_status')):
            raise ValueError('missing/unknown official result status or task id')
        b=row['base_status']=='pass';p=row['plus_status']=='pass'
        base+=int(b);plus+=int(b and p)
    return dict(tasks=len(rows),base_passes=base,plus_passes=plus,
                base_pass_at_1=base/len(rows),plus_pass_at_1=plus/len(rows))


def evaluate(*,output,probe_only=False,timeout=3600,**inputs):
    if timeout <= 0: raise ValueError('positive timeout required')
    binding = None if probe_only else validate_inputs(**inputs)
    runtime=runtime_info()
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    work=output/'work';work.mkdir()
    commands=[]
    write_json(output/'invocation.json',dict(schema='eesd-evalplus-isolated-invocation-v1',
        binding=binding,runtime=runtime,timeout_seconds=timeout,probe_only=probe_only,
        evaluator_source_sha256=sha(Path(__file__))))
    try:
        for stage in (['probe'] if probe_only else ['probe','sanitize','evaluate']):
            if stage=='sanitize':
                # No candidate bytes exist in the probe's writable mount.
                shutil.copyfile(Path(inputs['transfer_root'])/'samples.jsonl',work/'samples.jsonl')
                if sha(work/'samples.jsonl')!=binding['samples_sha256']:
                    raise ValueError('samples changed during preparation')
            if stage=='evaluate':
                sample_coverage(output_file(work,'samples-sanitized.jsonl'),binding['data_binding']['task_ids'])
            command=sandbox_command(runtime,work,stage,dataset=binding['dataset'] if binding else None,
                                    private_root=inputs.get('private_root'))
            commands.append(command)
            rc=run_command(command,output/f'{stage}.log',min(timeout,30) if stage=='probe' else timeout)
            if rc:
                raise RuntimeError(f'{stage} infrastructure/CLI failure (exit {rc})')
        if probe_only:
            write_json(output/'status.json',dict(status='probe_passed',commands=commands))
            return 0
        if validate_inputs(**inputs)!=binding or runtime_info()!=runtime:
            raise ValueError('locked inputs/runtime changed during evaluation')
        sample_coverage(output_file(work,'samples-sanitized.jsonl'),binding['data_binding']['task_ids'])
        results=output_file(work,'samples-sanitized_eval_results.json')
        metrics=summarize(json.loads(results.read_text()),binding['data_binding']['task_ids'])
        report=dict(schema='eesd-evalplus-isolated-result-v1',status='completed',metrics=metrics,
                    binding=binding,runtime=runtime,commands=commands,
                    evaluator_source_sha256=sha(Path(__file__)),
                    sanitized_samples_sha256=sha(work/'samples-sanitized.jsonl'),results_sha256=sha(results))
        write_json(output/'report.json',report)
        write_json(output/'complete.json',dict(report_sha256=sha(output/'report.json'),
            results_sha256=sha(results),sanitized_samples_sha256=sha(work/'samples-sanitized.jsonl'),
            invocation_sha256=sha(output/'invocation.json')))
        return 0
    except (OSError,ValueError,RuntimeError) as exc:
        write_json(output/'status.json',dict(status='infrastructure_failure',error=str(exc),commands=commands,
                                             model_failure_counted=False))
        return 3


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--probe-only',action='store_true')
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--transfer-root',type=Path)
    p.add_argument('--transfer-report-sha256')
    p.add_argument('--public-root',type=Path)
    p.add_argument('--public-manifest-sha256')
    p.add_argument('--private-root',type=Path)
    args=vars(p.parse_args())
    required=('transfer_root','transfer_report_sha256','public_root','public_manifest_sha256','private_root')
    if not args['probe_only'] and any(args[k] is None for k in required):
        p.error('locked transfer/public/private inputs and external report/manifest hashes required')
    try: return evaluate(**args)
    except (OSError,ValueError,KeyError) as exc: p.error(str(exc))


if __name__=='__main__':
    raise SystemExit(main())
