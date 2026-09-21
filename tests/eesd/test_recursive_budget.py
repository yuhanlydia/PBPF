"""Recursive training uses explicit arm-specific experience and sealed budgets."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from test_mechanism_runner import ROOT, load_script
from test_downstream_cli import fake_training_job
from pbpf.eesd.distillation import TRAIN_RULES


def scored_fixture(adapter=None):
    return '\n'.join(json.dumps({'trajectory_id':split,'source_component_id':split,'split':split,
        'prompt':'repair','correction':'pass','before_outcomes':[1],'after_outcomes':[0],
        'relevance':[1.0],'training_weights':{rule:1.0 for rule in TRAIN_RULES},
        'model_id':'test/model','model_revision':'a'*40,'adapter':str(adapter)})
        for split in ('train','development'))+'\n'



@pytest.mark.parametrize('flags,missing', [([], '--response-token-budget'),
    (['--response-token-budget', '53'], '--experience-policy'),
    (['--response-token-budget', '0', '--experience-policy', 'arm-specific'], 'positive')])
def test_recursive_cli_requires_explicit_budget_and_policy(tmp_path, flags, missing):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/run_eesd_recursive.py'),
        '--domain', 'rbr', '--public-root', str(tmp_path), '--evaluator-root', str(tmp_path),
        '--model-config', str(tmp_path / 'model.yaml'), '--family', 'qwen25_7b',
        '--output', str(tmp_path / 'out'), '--seed', '1701', *flags], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert missing in result.stderr
    assert not (tmp_path / 'out').exists()


@pytest.fixture
def train_request(tmp_path):
    model = tmp_path / 'model.yaml'
    model.write_text('model_id: test/model\nrevision: ' + 'a' * 40 + '\n')
    scored = tmp_path / 'scored.jsonl'
    scored.write_text(scored_fixture())
    return dict(root=ROOT, model_config=model, scored=scored, rule='eesd_full', previous_adapter=None,
        output=tmp_path / 'training', seed=1701, max_steps=200, anchor_beta=.03, response_token_budget=53)


def test_recursive_train_budget_binding_and_adapter_tamper(train_request, monkeypatch):
    recursive = load_script('run_eesd_recursive')
    commands = []
    def call(command, **kwargs):
        commands.append(command)
        fake_training_job(command)
    monkeypatch.setattr(recursive, 'call', call)
    outcome = recursive.train_next(**train_request)
    assert outcome['status']=='trained'
    adapter = Path(outcome['adapter'])
    assert commands[0][commands[0].index('--response-token-budget') + 1] == '53'
    assert (adapter.parent / 'training-binding.json').exists()
    commands.clear()
    assert recursive.train_next(**train_request) == outcome
    assert commands == []
    with pytest.raises(ValueError, match='binding'):
        recursive.train_next(**{**train_request, 'response_token_budget': 54})
    (adapter / 'adapter_model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='binding'):
        recursive.train_next(**train_request)
    assert commands == []


def test_recursive_train_rejects_unbound_legacy_report(train_request, monkeypatch):
    recursive = load_script('run_eesd_recursive')
    output = train_request['output']
    output.mkdir()
    (output / 'training-report.json').write_text('{"max_steps":200}')
    monkeypatch.setattr(recursive, 'call', lambda *a, **kw: pytest.fail('must not launch'))
    with pytest.raises(ValueError, match='unbound'):
        recursive.train_next(**train_request)


@pytest.mark.parametrize('policy',['arm-specific','shared_eesd_teacher'])
def test_three_round_pipeline_forwards_budget_and_labels_arm_specific_experience(tmp_path, monkeypatch,policy):
    recursive = load_script('run_eesd_recursive')
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump({'schema': 'eesd-iclr2027-v1', 'seeds': [1701],
        'distillation': {'transition_utility': [1, -1, 0, 0]}}))  # Explicit test fixture only.
    model = tmp_path / 'model.yaml'
    model.write_text('model_id: test/model\nrevision: ' + 'a' * 40 + '\n')
    for name in ('public', 'evaluator'):
        directory = tmp_path / name
        directory.mkdir()
        (directory / 'manifest.json').write_text('{}')
    output = tmp_path / 'recursive'
    argv = ['recursive', '--config', str(config), '--domain', 'rbr',
        '--public-root', str(tmp_path / 'public'), '--evaluator-root', str(tmp_path / 'evaluator'),
        '--model-config', str(model), '--family', 'qwen25_7b', '--output', str(output),
        '--seed', '1701', '--rounds', '3', '--response-token-budget', '53',
        '--experience-policy', policy]
    monkeypatch.setattr(sys, 'argv', argv)
    jobs, experiences, generation, comparisons = [], [], [], []
    monkeypatch.setattr(recursive, 'generate_bank', lambda **kw: generation.append(kw))
    def evaluate(**kw):
        kw['output'].mkdir(parents=True, exist_ok=True)
        (kw['output'] / 'report.json').write_text('{"fresh_all_tests_pass_at_1":0.5}')
    monkeypatch.setattr(recursive, 'evaluate_primary', evaluate)
    def collect(**kw):
        experiences.append(kw['adapter'])
        kw['output'].mkdir(parents=True, exist_ok=True)
        scored = kw['output'] / 'scored.jsonl'
        scored.write_text(scored_fixture(kw['adapter']))
        return scored
    monkeypatch.setattr(recursive, 'collect_experience', collect)
    def compare(root, baseline, method, output):
        comparisons.append((baseline,method,output))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{}')
    monkeypatch.setattr(recursive, 'compare_reports', compare)
    def call(command, **kw):
        jobs.append(command)
        fake_training_job(command)
    monkeypatch.setattr(recursive, 'call', call)
    recursive.main()
    assert len(jobs) == 6
    assert all(c[c.index('--response-token-budget') + 1] == '53' for c in jobs)
    if policy=='arm-specific':
        assert experiences == [None,
            output / 'round1/equal_weight/training/adapter', output / 'round1/eesd_full/training/adapter',
            output / 'round2/equal_weight/training/adapter', output / 'round2/eesd_full/training/adapter']
    else:
        assert experiences==[None,output/'round1/eesd_full/training/adapter',output/'round2/eesd_full/training/adapter']
        for r in range(1,4):
            commands=[c for c in jobs if str(output/f'round{r}/') in c[c.index('--output')+1]]
            assert len(commands)==2
            assert commands[0][commands[0].index('--input')+1]==commands[1][commands[1].index('--input')+1]
            teacher=output/f'round{r-1}/eesd_full/training/adapter' if r>1 else None
            for c in commands:
                if teacher is None:assert '--previous-adapter' not in c
                else:assert c[c.index('--previous-adapter')+1]==str(teacher)
            previous=output/f'round{r-1}/eesd_full/fresh-eval/report.json' if r>1 else output/'round0/base/fresh-eval/report.json'
            assert [baseline for baseline,_,target in comparisons if target.name=='vs-previous.json' and target.parent.parent.name==f'round{r}']==[previous,previous]
            binding=json.loads((output/f'round{r}/update-inputs.json').read_text())
            assert binding['shared_experience'] is True
            assert binding['arms']['equal_weight']['scored_sha256']==binding['arms']['eesd_full']['scored_sha256']
            assert binding['arms']['equal_weight']['previous_adapter_sha256']==binding['arms']['eesd_full']['previous_adapter_sha256']
            assert binding['response_token_budget']==53
    report = json.loads((output / 'recursive-report.json').read_text())
    assert report['experience_policy'] == policy
    assert report['response_token_budget_per_arm_per_round'] == 53
    assert len(report['summary']) == 3
    assert len(list(output.glob('round*/*/training/training-binding.json'))) == 6
    jobs.clear()
    recursive.main()
    assert jobs == []
    update_path=output/'round1/update-inputs.json'
    original_binding=update_path.read_text()
    corrupt=json.loads(original_binding);corrupt['teacher_adapter_sha256']='tampered'
    update_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError,match='teacher/shared bank/budget'):
        recursive.main()
    assert jobs==[]
    update_path.write_text(original_binding)
    generation.clear()
    other=list(argv);other[other.index('--experience-policy')+1]=('arm-specific' if policy=='shared_eesd_teacher' else 'shared_eesd_teacher')
    monkeypatch.setattr(sys,'argv',other)
    with pytest.raises(ValueError,match='identity/budget/policy'):
        recursive.main()
    assert generation==[]
    changed = list(argv)
    changed[changed.index('--response-token-budget') + 1] = '54'
    monkeypatch.setattr(sys, 'argv', changed)
    with pytest.raises(ValueError, match='identity/budget/policy'):
        recursive.main()
    assert generation == []
    (output / 'recursive-run.json').unlink()
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(ValueError, match='unbound legacy'):
        recursive.main()
    assert generation == []
