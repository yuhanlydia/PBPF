"""Prospective, separate Replay inference families and immutable input lock."""
from pathlib import Path
import hashlib
import itertools
import json
import yaml
from .mechanism_contrasts import contrast_ids, ALPHAS, STRENGTHS
from .mechanism_inference import METRICS

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ('apps_replay', 'codecontests_replay')
MODELS = ('qwen25_7b', 'deepseek_6p7b', 'seed_coder_8b', 'starcoder2_15b')
SEEDS = (1701, 1702, 1703)
SOURCES = (
    'src/pbpf/eesd/replay_statistics.py', 'src/pbpf/eesd/replay_artifacts.py',
    'scripts/run_eesd_replay_evidence.py', 'scripts/report_eesd_replay_inference.py',
    'scripts/report_eesd_mechanism_inference.py', 'scripts/run_eesd_evidence_matrix.py',
    'src/pbpf/eesd/evidence.py', 'src/pbpf/eesd/mechanism_artifacts.py',
    'src/pbpf/eesd/mechanism_contrasts.py', 'src/pbpf/eesd/mechanism_inference.py',
    'src/pbpf/eesd/replay_execution_cache.py', 'src/pbpf/eesd/replay_runtime.py',
    'src/pbpf/eesd/replay_generation.py', 'src/pbpf/eesd/replay_prompt.py',
    'scripts/build_eesd_replay_mechanism_cache.py', 'scripts/plan_eesd_replay_caches.py',
)
DOCUMENTS = ('docs/EESD_REPLAY_STATISTICS_LOCK_20260920.md',
             'docs/EESD_REPLAY_EXTENSION_SPEC_20260920.md')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pointer(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path)}


def families():
    primary, secondary = [], []
    for domain, model, contrast, metric in itertools.product(DOMAINS, MODELS, contrast_ids(), METRICS):
        key = f'{domain}/{model}/{contrast}/{metric}'
        (primary if contrast == 'core/effective_params' and metric == 'nll' else secondary).append(key)
    return primary, secondary


def protocol():
    return dict(domains=list(DOMAINS), models=list(MODELS), seeds=list(SEEDS),
                development_sources=200, primary_sources=500, contrasts=list(contrast_ids()),
                metrics=list(METRICS), primary_slots=8, secondary_slots=1560,
                bootstrap_draws=10000, bootstrap_seed=314159, ece_bins=10, alpha=.05,
                multiplicity='separate-Holm-families', joint_study_wide_fwer_claim=False,
                secondary_resolution_prevents_first_rejection=True,
                public_support_before_assessment_reconstruction=True,
                missing_slots='retain-and-block-complete-family-decisions',
                n4_alias='core/tuned')


def validate_config(config):
    require(config.get('schema') == 'eesd-iclr2027-v1' and config.get('seeds') == list(SEEDS)
            and config.get('bootstrap_seed') == 314159, 'config identity mismatch')
    expected = dict(alphas=list(ALPHAS), strengths=list(STRENGTHS), visible_counts=[1,2,4,8],
                    bootstrap_draws=10000, global_mass_grid=[.25,.5,1,2,4,8,16],
                    ece_bins=10, permutation_seeds=list(SEEDS), concentration_bins_for_n4=[1,2,3,4],
                    primary_metric='nll', secondary_metrics=['brier','ece','accuracy'])
    require(config.get('evidence') == expected, 'locked evidence grid or budget mismatch')
    expected_stats = dict(unit='source_problem', paired_bootstrap=True, holm_secondary=True,
                          report_all_predeclared_cells=True, forbid_posthoc_cell_dropping=True)
    require(config.get('statistics') == expected_stats, 'statistical policy mismatch')


def validate_matrix(matrix):
    require(matrix.get('schema') == 'eesd-replay-cache-matrix-v1', 'cache matrix schema mismatch')
    cells = matrix['cells']
    require(len(cells) == 24 and {(c['domain'], c['family'], c['seed']) for c in cells}
            == set(itertools.product(DOMAINS, MODELS, SEEDS)), 'incomplete cache matrix')
    require(all(c['candidate_jobs'] == 700 and c['test_executions'] == 7000 for c in cells),
            'cache population mismatch')
    for item in ('generation_manifest', 'execution_lock'):
        ref = matrix[item]
        require(sha(ref['path']) == ref['sha256'], f'{item} checksum mismatch')
    require(sha(Path(matrix['bundle'])/'admission.json') == matrix['admission_sha256'],
            'admission checksum mismatch')
    for name, digest in matrix['sources'].items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha(path) == digest, 'cache matrix source changed')


def build_statistical_lock(config_path, cache_manifest_path):
    config = pointer(config_path); matrix_ref = pointer(cache_manifest_path)
    validate_config(yaml.safe_load(Path(config['path']).read_text()))
    matrix = json.loads(Path(matrix_ref['path']).read_text()); validate_matrix(matrix)
    return dict(schema='eesd-replay-statistical-lock-v1', protocol=protocol(), config=config,
                cache_manifest=matrix_ref, documents={p: sha(ROOT/p) for p in DOCUMENTS},
                sources={p: sha(ROOT/p) for p in SOURCES},
                admission_sha256=matrix['admission_sha256'],
                execution_lock_sha256=matrix['execution_lock']['sha256'])


def validate_statistical_lock(path, expected_sha256, config_path=None, cache_manifest_path=None):
    require(sha(path) == expected_sha256, 'statistical lock checksum mismatch')
    lock = json.loads(Path(path).read_text())
    require(lock.get('schema') == 'eesd-replay-statistical-lock-v1', 'statistical lock schema mismatch')
    for name, override in [('config', config_path), ('cache_manifest', cache_manifest_path)]:
        if override is not None:
            require(Path(override).resolve() == Path(lock[name]['path']).resolve(), f'{name} path mismatch')
    expected = build_statistical_lock(lock['config']['path'], lock['cache_manifest']['path'])
    require(lock == expected, 'statistical lock inputs, protocol or sources changed')
    return lock
