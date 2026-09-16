"""Build an explicit development-only assessment cache for method iteration."""
from collections import Counter
import copy
import hashlib
import json


def development_cache(payload, *, seed=2701, inner_validation_fraction=.2):
    """Use only original train/development sources; exclude the original test.

    The existing fitter's test slot is an assessment slot in this derived cache.
    Its role is recorded explicitly and must not support confirmatory claims.
    """
    if not 0 < inner_validation_fraction < 1:
        raise ValueError("inner validation fraction must be between zero and one")
    training = [r for r in payload["records"] if r["split"] == "train"]
    assessment = [r for r in payload["records"] if r["split"] == "development"]

    def source(row):
        return row.get("source_component_id", row["problem_id"])

    train_sources = {source(r) for r in training}
    if train_sources & {source(r) for r in assessment}:
        raise ValueError("original training and development sources overlap")
    ordered = sorted(train_sources, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).digest())
    count = max(1, round(len(ordered) * inner_validation_fraction))
    if count >= len(ordered) or not assessment:
        raise ValueError("development tuning requires nonempty disjoint fitting and assessment populations")
    validation_sources = set(ordered[:count])
    rows = []
    for row in training:
        rows.append({**copy.deepcopy(row), "split": "development" if source(row) in validation_sources else "train"})
    rows.extend({**copy.deepcopy(row), "split": "test"} for row in assessment)
    # Do not inherit old test inventory/count metadata or any test record bytes.
    result = {key: copy.deepcopy(payload[key]) for key in
              ("schema", "dataset", "seed", "tests_per_candidate", "official_md5", "problem_descriptions",
               "execution_fields_visibility", "source_split_policy", "execution_protocol") if key in payload}
    result.update(records=rows, counts=dict(Counter(r["split"] for r in rows)),
                  problem_counts={s: len({source(r) for r in rows if r["split"] == s})
                                  for s in ("train", "development", "test")},
                  evaluation_role="development_assessment_only",
                  population_policy="original train sources split into fitting/validation; original development used for assessment; original test excluded",
                  development_split_seed=seed,
                  assessment_source_ids=sorted({source(r) for r in assessment}),
                  source_training_sha256=hashlib.sha256(json.dumps(training, sort_keys=True).encode()).hexdigest(),
                  source_development_sha256=hashlib.sha256(json.dumps(assessment, sort_keys=True).encode()).hexdigest())
    return result
