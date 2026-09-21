#!/usr/bin/env python3
"""Original 8-cell/three-seed inference under a separately bound direct profile."""
import argparse
import json
from pathlib import Path
import runpy
from pbpf.eesd import direct_artifacts as adapter,direct_runtime as runtime
from pbpf.eesd.mechanism_artifacts import read,require,file_sha
ROOT=Path(__file__).resolve().parents[1]
BASE=runpy.run_path(str(ROOT/'scripts/report_eesd_mechanism_inference.py'))


def pointer(path):return {'path':str(Path(path).resolve()),'sha256':file_sha(path)}
def source_inventory():return adapter.source_inventory()


def build_statistical_lock(original_statistical_lock,original_sha256,execution_lock,execution_sha256):
    require(file_sha(original_statistical_lock)==original_sha256,'original statistical lock checksum mismatch')
    execution=runtime.validate_execution_lock(execution_lock,execution_sha256)
    BASE['validate_inputs'](execution['manifest']['path'],execution['config']['path'],original_statistical_lock,original_sha256)
    return dict(schema='eesd-direct-statistical-lock-v1',execution_profile='direct-no-sandbox',
        original_statistical_lock=pointer(original_statistical_lock),execution_lock=pointer(execution_lock),
        amendment=execution['amendment'],sources=source_inventory(),
        protocol={'cells':8,'seeds':[1701,1702,1703],'primary_slots':8,'secondary_slots':1560,
                  'contrast_math':'unchanged','public_support_before_assessment':True,'mix_execution_profiles':False})


def validate_statistical_lock(path,expected_sha256):
    require(file_sha(path)==expected_sha256,'direct statistical lock checksum mismatch')
    lock=read(path);old=lock['original_statistical_lock'];execution=lock['execution_lock']
    require(lock==build_statistical_lock(old['path'],old['sha256'],execution['path'],execution['sha256']),'direct statistics source/profile changed')
    return lock


def run_report(*,statistical_lock,statistical_lock_sha256,results_root,output,preflight_only=False):
    lock=validate_statistical_lock(statistical_lock,statistical_lock_sha256);ref=lock['execution_lock']
    execution=runtime.validate_execution_lock(ref['path'],ref['sha256'])
    manifest=Path(execution['manifest']['path']);config=Path(execution['config']['path'])
    cells=BASE['validate_inputs'](manifest,config,lock['original_statistical_lock']['path'],lock['original_statistical_lock']['sha256'])
    cellmap={(c['domain'],c['model']):c for c in cells}
    def inspect(cell,root,cfg,verifier):
        inventory=BASE['public_inventory'](cell);coverage={}
        for seed in BASE['SEEDS']:
            directory=Path(root)/'mechanism'/cell['dataset']/cell['model']/f'seed{seed}'
            entry={'report_dir':str(directory/'report'),'cache':str(directory/'cache.json'),'verified_banks':[]}
            if (directory/'complete.json').exists():
                adapter.verify_direct_cell(directory,cell,seed,cfg,ref['path'],ref['sha256'],public_only=True)
                entry.update(status='verified',reasons=[])
            else:entry.update(status='missing',reasons=['direct cell incomplete' if directory.exists() else 'direct cell missing'])
            coverage[str(seed)]=entry
        return inventory,coverage
    def arguments(kwargs):
        return {**kwargs,'cell':cellmap[(kwargs['domain'],kwargs['model'])],'execution_lock':ref['path'],'lock_sha':ref['sha256']}
    def support(**kwargs):return adapter.derive_public_support(**arguments(kwargs))
    def reconstruct(**kwargs):return adapter.reconstruct_contrasts(**arguments(kwargs))
    def write(path,value):
        if path.name in ('input-lock.json','report.json','complete.json'):
            value.update(execution_profile='direct-no-sandbox',direct_statistical_lock_sha256=statistical_lock_sha256)
            if 'schema' in value:value['schema']=value['schema'].replace('eesd-mechanism-','eesd-direct-mechanism-')
            if path.name=='input-lock.json':value.update(direct_sources=source_inventory(),execution_amendment=lock['amendment'])
        BASE['write'](path,value)
    fn=adapter.isolated(BASE['run_report'],validate_inputs=lambda *args:cells,inspect_cell=inspect,
        derive_public_support=support,reconstruct_contrasts=reconstruct,write=write,__file__=__file__)
    result=fn(manifest=manifest,config=config,results_root=Path(results_root),output=Path(output),
        statistical_lock=Path(statistical_lock),statistical_lock_sha256=statistical_lock_sha256,preflight_only=preflight_only)
    validate_statistical_lock(statistical_lock,statistical_lock_sha256)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('statistical-lock','results-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--statistical-lock-sha256',required=True);p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args();result=run_report(**vars(args));print(json.dumps({'status':result['status'],'execution_profile':'direct-no-sandbox'}))

if __name__=='__main__':main()
