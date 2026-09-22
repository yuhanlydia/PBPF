#!/usr/bin/env python3
"""Seal the twelve paired source-level baseline reports under one direct lock."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/eesd-direct-20260921/fresh-baselines'
INDEX = ROOT / 'runs/eesd-direct-20260921/operations/seed1701-initial-baselines.json'
LOCK_SHA = '8f3098efea7b9a2eb38a9434963cf200d330bfb53e869f4ed3a12280b01500e2'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    index = json.loads(INDEX.read_text())
    if len(index['cells']) != 12:
        raise ValueError('expected twelve benchmark/model cells')
    cells = []
    for item in index['cells']:
        domain, family = item['domain'], item['family']
        path = BASE / domain / family / 'seed1701/report.json'
        report = json.loads(path.read_text())
        records = report['records']
        sources = [row['source_component_id'] for row in records]
        if (report.get('schema') != 'eesd-fresh-policy-eval-v1'
                or report.get('domain') not in (domain, 'rbr' if domain == 'runbugrun' else domain)
                or report.get('execution_profile') != 'direct-no-sandbox'
                or report.get('execution_lock_sha256') != LOCK_SHA
                or report.get('adapter_sha256') is not None
                or report.get('sources') != 500 or len(records) != 500
                or len(set(sources)) != 500
                or report.get('all_tests_passes') != item['all_ten_pass']
                or sum(bool(row['all_tests_pass']) for row in records) != item['all_ten_pass']):
            raise ValueError('baseline cell report differs: ' + domain + '/' + family)
        cells.append({'domain': domain, 'family': family, 'seed': 1701,
                      'model': report['model'], 'revision': report['revision'],
                      'primary_sources': 500, 'all_ten_pass': item['all_ten_pass'],
                      'report': str(path), 'report_sha256': sha(path),
                      'source_ids_sha256': hashlib.sha256(json.dumps(
                          sorted(sources), separators=(',', ':')).encode()).hexdigest(),
                      'task_manifest_sha256': report['task_manifest_sha256']})
    result = {'schema': 'eesd-seed1701-baseline-matrix-v1',
              'execution_profile': 'direct-no-sandbox',
              'execution_lock_sha256': LOCK_SHA,
              'cells': cells, 'cell_count': len(cells),
              'scientific_scope': 'fixed no_update primary Pass@1 baselines; no trained method result'}
    path = BASE / 'manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('existing baseline matrix seal differs')
    else:
        partial = path.with_suffix('.partial')
        partial.write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
        partial.rename(path)
    print(json.dumps({'status': 'sealed', 'cells': len(cells),
                      'manifest': str(path), 'sha256': sha(path)}))


if __name__ == '__main__':
    main()
