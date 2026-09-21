import json
from pathlib import Path

import pytest

from pbpf.apbpf.assessment_bundle import create_bundle, forward_bundle, load_bundle
from pbpf.apbpf.codearc_bank import file_sha
from pbpf.apbpf.stage_prediction import DOMAINS


class DeclaredIO:
    """Small checksummed direct-dependency fixture, not experimental evidence."""
    def __init__(self, output, inputs, protocol):
        self.outputs=output;output.mkdir()
        self.inputs=inputs
        self.inventory={name:{str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
                        for name,root in inputs.items()}
        self.request={'fingerprint':'a'*64,'dependencies':{name:'b'*64 for name in inputs}}
        self.config={'protocol':protocol}

    def artifact(self,name,relative):
        path=self.inputs[name]/relative
        assert file_sha(path)==self.inventory[name][relative]
        return path

    def directory(self,name,relative):
        root=self.inputs[name]/relative
        for path in root.rglob('*'):
            if path.is_file():self.artifact(name,str(path.relative_to(self.inputs[name])))
        return root


def prepared(tmp_path):
    execution=tmp_path/'execution';execution.mkdir();training=tmp_path/'training';training.mkdir()
    protocol={'seeds':[1701,1702,1703]};index={'training_caches':{}};cells={}
    for domain,dataset in DOMAINS.items():
        cache=execution/f'{domain}.json';cache.write_text(json.dumps({'fixture':dataset}))
        index['training_caches'][domain]={'full':{'path':cache.name,'sha256':file_sha(cache)}}
        for seed in protocol['seeds']:
            name=f'{domain}-{seed}';directory=training/name;directory.mkdir();checkpoint=directory/'belief.pt'
            checkpoint.write_bytes(f'fixture:{domain}:{seed}'.encode())
            report={'dataset':dataset,'config':{'seed':seed,'protocol':protocol},
                    'arms':{'belief':{'checkpoint_sha256':file_sha(checkpoint)}}}
            path=directory/'training.json';path.write_text(json.dumps(report))
            cells[domain,seed]={'entry':{'directory':name,'cache_sha256':file_sha(cache)},
                'checkpoints':{'belief':checkpoint},'report':report,'training_report_sha256':file_sha(path)}
    (execution/'execution-index.json').write_text(json.dumps(index))
    io=DeclaredIO(tmp_path/'association',{'execution_cache':execution,'train_belief':training},protocol)
    return io,cells,protocol


def test_bundle_forwards_exact_cache_checkpoint_and_origin_dependency_bytes(tmp_path):
    io,cells,protocol=prepared(tmp_path);manifest=create_bundle(io,cells)
    gate=DeclaredIO(tmp_path/'association-gate',{'association':io.outputs},protocol)
    assert forward_bundle(gate,'association')==manifest
    oracle=DeclaredIO(tmp_path/'oracle',{'association_gate':gate.outputs},protocol)
    loaded=load_bundle(oracle,'association_gate')
    assert set(loaded)==set(DOMAINS)
    for domain in DOMAINS:
        assert set(loaded[domain]['models'])==set(protocol['seeds'])
        for seed in protocol['seeds']:
            assert loaded[domain]['models'][seed]['checkpoint'].read_bytes()==cells[domain,seed]['checkpoints']['belief'].read_bytes()
    assert manifest['origin_dependencies']==io.request['dependencies']


def test_bundle_rejects_checkpoint_trained_on_a_different_cache(tmp_path):
    io,cells,_=prepared(tmp_path);cells['rbr',1701]['entry']['cache_sha256']='f'*64
    with pytest.raises(ValueError,match='training cache differs'):
        create_bundle(io,cells)


def test_bundle_manifest_cannot_swap_seed_identity(tmp_path):
    io,cells,protocol=prepared(tmp_path);create_bundle(io,cells)
    path=io.outputs/'assessment-bundle/manifest.json';value=json.loads(path.read_text())
    value['domains']['rbr']['models'][0]['seed']=1702;path.write_text(json.dumps(value))
    consumer=DeclaredIO(tmp_path/'consumer',{'association_gate':io.outputs},protocol)
    with pytest.raises(ValueError,match='identity mismatch'):
        load_bundle(consumer,'association_gate')
