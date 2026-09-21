import copy
import importlib.util
from pathlib import Path

import pytest


def loader():
    spec = importlib.util.spec_from_file_location('bank_evaluation_contract',
        Path(__file__).resolve().parents[2]/'scripts/run_local_bank_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_banks():
    identity = dict(model='pinned', revision='sha', seed=1701, max_new_tokens=512,
                    temperature=.8, top_p=.95, public_tasks_sha256='public', candidates=8)
    return {name: ({**identity, 'split': 'development' if name.startswith('development-') else name},
                   [{'source_component_id': f'{name}/{i}'} for i in range(count)], 'hash')
            for name, count in [('train', 212), ('development-pilot', 16),
                                ('development-remainder', 384), ('primary', 500)]}


def test_full_inventory_and_cross_split_source_disjointness():
    banks = complete_banks(); module = loader()
    module.validate_populations(banks, domain='codearc')
    banks['primary'][1][0]['source_component_id'] = 'train/0'
    with pytest.raises(ValueError, match='disjoint'):
        module.validate_populations(banks, domain='codearc')


def test_partial_population_and_mixed_model_families_are_rejected():
    banks = complete_banks(); module = loader()
    partial = copy.deepcopy(banks); partial['train'][1].pop()
    with pytest.raises(ValueError, match='complete disjoint'):
        module.validate_populations(partial, domain='codearc')
    banks['primary'][0]['model'] = 'another-model'
    with pytest.raises(ValueError, match='model'):
        module.validate_populations(banks, domain='codearc')
