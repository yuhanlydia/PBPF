import copy
import runpy
from pathlib import Path

import pytest


def api():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/run_apbpf_parameter_diagnostic.py'))


def records():
    return [dict(split='train', problem_id=f'p{i}', source_component_id=f's{i // 2}', value=i)
            for i in range(20)] + [dict(split='development', problem_id='dev', value=1),
                                   dict(split='test', problem_id='test', value='HIDDEN')]


def test_partition_excludes_test_and_keeps_source_components_disjoint():
    split = api()['partition_development']
    rows = records()
    before = copy.deepcopy(rows)
    parts = split(rows)
    assert rows == before
    assert set(parts) == {'train', 'validation', 'assessment'}
    assert len(parts['train']) == 16 and len(parts['validation']) == 4
    assert parts['assessment'] == [rows[-2]]
    assert 'HIDDEN' not in repr(parts)
    groups = [{r.get('source_component_id', r['problem_id']) for r in parts[k]}
              for k in parts]
    assert all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:])
    rows[-1]['value'] = 'OTHER SECRET'
    assert split(rows) == parts


def test_partition_rejects_overlapping_development_sources():
    rows = records()
    rows[-2]['source_component_id'] = 's0'
    with pytest.raises(ValueError, match='overlap'):
        api()['partition_development'](rows)


def test_variants_change_one_factor_and_keep_reference():
    variants = api()['VARIANTS']
    assert variants['reference'] == {}
    assert len(variants) == 6
    assert all(len(v) == 1 for k, v in variants.items() if k != 'reference')
