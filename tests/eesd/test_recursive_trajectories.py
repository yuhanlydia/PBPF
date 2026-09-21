import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pbpf.eesd.recursive_trajectories import summarize_recursive_trajectories


def reports():
    paths = {'a': [True, False, True, False], 'b': [True] * 4,
             'c': [False, True, False, True], 'd': [False] * 4}
    return {i: {
        'schema': 'eesd-fresh-policy-eval-v1', 'domain': 'rbr', 'model': 'model',
        'revision': 'revision', 'task_manifest_sha256': 'manifest',
        'selection': 'none; exactly one fresh candidate per primary source',
        'sources': 4,
        'records': [{'source_component_id': s, 'task_id': s,
                     'candidate_id': f'{s}/round{i}', 'all_tests_pass': values[i]}
                    for s, values in paths.items()],
    } for i in range(4)}


def test_hand_computable_cumulative_trajectories():
    result = summarize_recursive_trajectories(reports())
    final = result['by_round'][-1]
    assert final['ever_regressed_sources'] == 2
    assert final['cumulative_regression_events'] == 3
    assert final['regressions_after_recovery'] == 1
    assert final['always_retained_base_correct_fraction'] == .5
    assert [r['cumulative_regression_events'] for r in result['by_round']] == [0, 1, 2, 3]
    assert result['trajectories'][0]['correctness'] == [True, False, True, False]
    assert result['trajectories'][0]['regression_rounds'] == [1, 3]
    assert result['trajectories'][0]['recovery_rounds'] == [2]


@pytest.mark.parametrize('mutation,match', [
    (lambda r: r.pop(2), 'rounds'),
    (lambda r: r[2]['records'].pop(), 'population'),
    (lambda r: r[1]['records'].append(copy.deepcopy(r[1]['records'][0])), 'population'),
    (lambda r: r[1].update(selection='best of eight'), 'one fresh candidate'),
    (lambda r: r[1]['records'][0].update(all_tests_pass='false'), 'boolean'),
    (lambda r: r[1].update(task_manifest_sha256='different'), 'identity'),
    (lambda r: r[1]['records'][0].update(task_id='different'), 'task'),
])
def test_rejects_unpaired_or_invalid_reports(mutation, match):
    values = reports()
    mutation(values)
    with pytest.raises(ValueError, match=match):
        summarize_recursive_trajectories(values)


def test_no_base_correct_has_undefined_retention_not_zero():
    values = reports()
    for row in values[0]['records']:
        row['all_tests_pass'] = False
    result = summarize_recursive_trajectories(values)
    assert all(r['always_retained_base_correct_fraction'] is None for r in result['by_round'])


def test_cli_reads_existing_reports_and_refuses_overwrite(tmp_path):
    (tmp_path / 'recursive-run.json').write_text(json.dumps({
        'schema': 'eesd-recursive-run-v1', 'experience_policy': 'arm-specific', 'rounds': 3}))
    for i, report in reports().items():
        arm = 'base' if i == 0 else 'eesd_full'
        path = tmp_path / f'round{i}' / arm / 'fresh-eval' / 'report.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
    output = tmp_path / 'summary.json'
    script = Path(__file__).resolve().parents[2] / 'scripts/summarize_eesd_recursive_trajectories.py'
    cmd = [sys.executable, str(script), '--study-root', str(tmp_path),
           '--rule', 'eesd_full', '--output', str(output)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    original = output.read_bytes()
    value = json.loads(original)
    assert value['rule'] == 'eesd_full'
    for entry in value['input_reports']:
        assert entry['sha256'] == hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest()
    repeated = subprocess.run(cmd, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert output.read_bytes() == original


def test_matched_control_compares_each_update_to_teacher_not_previous_control():
    from pbpf.eesd.recursive_trajectories import summarize_matched_controls
    teacher = reports()
    control = copy.deepcopy(teacher)
    for round_index in (1, 2, 3):
        for row in control[round_index]['records']:
            row['all_tests_pass'] = round_index != 2
    result = summarize_matched_controls(control, teacher)
    assert [r['regressions'] for r in result['by_round'][1:]] == [0, 2, 0]
    assert [r['fixes'] for r in result['by_round'][1:]] == [2, 0, 2]
    assert [r['retained_teacher_correct_fraction'] for r in result['by_round'][1:]] == [1, 0, 1]
    assert result['by_round'][-1]['cumulative_teacher_relative_regression_events'] == 2
    assert 'trajectories' not in result
    assert result['independent_closed_loop'] is False


def test_matched_control_rejects_different_teacher_population():
    from pbpf.eesd.recursive_trajectories import summarize_matched_controls
    teacher, control = reports(), reports()
    for report in teacher.values():
        for row in report['records']:
            row['task_id'] += '/other'
    with pytest.raises(ValueError, match='teacher'):
        summarize_matched_controls(control, teacher)


def test_matched_retention_undefined_without_correct_teacher_sources():
    from pbpf.eesd.recursive_trajectories import summarize_matched_controls
    teacher, control = reports(), reports()
    for row in teacher[1]['records']:
        row['all_tests_pass'] = False
    result = summarize_matched_controls(control, teacher)
    assert result['by_round'][2]['retained_teacher_correct_fraction'] is None
    assert result['by_round'][2]['regression_rate_on_teacher_correct'] is None


def shared_study(tmp_path):
    run = {'schema': 'eesd-recursive-run-v1', 'experience_policy': 'shared_eesd_teacher',
           'rounds': 3, 'seed': 1701, 'response_token_budget': 1234}
    lock = tmp_path / 'recursive-run.json'; lock.write_text(json.dumps(run))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    teacher = reports(); control = copy.deepcopy(teacher)
    for r in (1, 2, 3):
        teacher[r]['adapter_sha256'] = hashlib.sha256(f'teacher{r}'.encode()).hexdigest()
        control[r]['adapter_sha256'] = hashlib.sha256(f'control{r}'.encode()).hexdigest()
        for row in control[r]['records']: row['all_tests_pass'] = r != 2
    for rule, values in [('eesd_full', teacher), ('equal_weight', control)]:
        for r, report in values.items():
            path = tmp_path / f'round{r}' / ('base' if r == 0 else rule) / 'fresh-eval/report.json'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report))
    for r in (1, 2, 3):
        parent = tmp_path / f'round{r-1}' / ('base' if r == 1 else 'eesd_full') / 'fresh-eval/report.json'
        scored = tmp_path / f'round{r}/shared-scored.jsonl'; scored.write_text('{}\n')
        adapter = None if r == 1 else str(tmp_path / f'round{r-1}/eesd_full/adapter')
        parent_sha = teacher[r-1].get('adapter_sha256')
        receipt = dict(schema='eesd-recursive-update-inputs-v1', round=r, seed=1701,
            experience_policy='shared_eesd_teacher', response_token_budget=1234,
            shared_experience=True, teacher_rule='base' if r == 1 else 'eesd_full', teacher_round=r-1,
            teacher_adapter=adapter, teacher_adapter_sha256=parent_sha,
            shared_scored_sha256=digest(scored), recursive_run_sha256=digest(lock),
            arms={rule: dict(parent_evaluation=str(parent), parent_evaluation_sha256=digest(parent),
                  previous_adapter=adapter, previous_adapter_sha256=parent_sha,
                  scored=str(scored), scored_sha256=digest(scored)) for rule in ('equal_weight','eesd_full')})
        (tmp_path / f'round{r}/update-inputs.json').write_text(json.dumps(receipt))
    return [sys.executable, str(Path(__file__).resolve().parents[2] / 'scripts/summarize_eesd_recursive_trajectories.py'),
            '--study-root', str(tmp_path), '--rule', 'equal_weight', '--output', str(tmp_path / 'matched.json')]


def test_cli_shared_control_uses_bound_teacher_parent(tmp_path):
    command = shared_study(tmp_path)
    subprocess.run(command, check=True, capture_output=True, text=True)
    result = json.loads((tmp_path / 'matched.json').read_text())
    assert result['schema'] == 'eesd-recursive-matched-controls-v1'
    assert result['by_round'][2]['regressions'] == 2
    assert len(result['update_inputs']) == 3


def test_cli_rejects_receipt_pointing_to_previous_control(tmp_path):
    command = shared_study(tmp_path)
    path = tmp_path / 'round2/update-inputs.json'
    receipt = json.loads(path.read_text())
    wrong = tmp_path / 'round1/equal_weight/fresh-eval/report.json'
    receipt['arms']['equal_weight']['parent_evaluation'] = str(wrong)
    receipt['arms']['equal_weight']['parent_evaluation_sha256'] = hashlib.sha256(wrong.read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt))
    process = subprocess.run(command, capture_output=True, text=True)
    assert process.returncode != 0
    assert not (tmp_path / 'matched.json').exists()


@pytest.mark.parametrize('tamper',['scored_bytes','run_lock','budget','adapter_path','shared_path'])
def test_shared_cli_rejects_binding_tampering(tmp_path,tamper):
    command=shared_study(tmp_path)
    path=tmp_path/'round2/update-inputs.json'
    receipt=json.loads(path.read_text())
    if tamper=='scored_bytes':
        Path(receipt['arms']['equal_weight']['scored']).write_text('changed\n')
    elif tamper=='run_lock':
        lock=tmp_path/'recursive-run.json'
        lock.write_text(lock.read_text()+'\n') # Same semantic settings, broken exact run binding.
    elif tamper=='budget':
        receipt['response_token_budget']+=1
    elif tamper=='adapter_path':
        receipt['arms']['equal_weight']['previous_adapter']=str(tmp_path/'different-adapter')
    else:
        other=tmp_path/'identical-but-different-bank.jsonl'
        other.write_bytes(Path(receipt['arms']['equal_weight']['scored']).read_bytes())
        receipt['arms']['equal_weight']['scored']=str(other)
    path.write_text(json.dumps(receipt))
    process=subprocess.run(command,capture_output=True,text=True)
    assert process.returncode!=0
    assert not (tmp_path/'matched.json').exists()


def test_shared_eesd_summary_retains_real_chain_and_teacher_update_table(tmp_path):
    command=shared_study(tmp_path)
    command[command.index('--rule')+1]='eesd_full'
    subprocess.run(command,check=True,capture_output=True,text=True)
    result=json.loads((tmp_path/'matched.json').read_text())
    assert result['schema']=='eesd-recursive-trajectories-v1'
    assert result['independent_closed_loop'] is True
    assert len(result['teacher_relative_updates']['by_round'])==4
    assert len(result['update_inputs'])==3
    assert 'adapter trees not reverified' in result['binding_scope']
