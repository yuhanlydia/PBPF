from __future__ import annotations

from .schema import DatasetSpec, ModelSpec

MODELS = {
    "plumbing": ModelSpec("Qwen/Qwen2.5-Coder-1.5B-Instruct", "plumbing"),
    "primary": ModelSpec("Qwen/Qwen2.5-Coder-7B-Instruct", "primary"),
    "architecture_replication": ModelSpec("Qwen/Qwen3-8B", "architecture_replication"),
    "cross_family": ModelSpec(
        "deepseek-ai/deepseek-coder-6.7b-instruct", "cross_family"
    ),
    "seed_coder_replication": ModelSpec(
        "ByteDance-Seed/Seed-Coder-8B-Instruct", "cross_family_replication"
    ),
    "starcoder2_replication": ModelSpec(
        "bigcode/starcoder2-15b-instruct-v0.1", "cross_family_replication"
    ),
    "flagship_coder_replication": ModelSpec(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct", "flagship_coder_replication"
    ),
}

DATASETS = {
    "finite": DatasetSpec("finite", "exact_identification"),
    "runbugrun": DatasetSpec("runbugrun", "real_repair"),
    "codearc": DatasetSpec("codearc", "interactive_transfer"),
    "evalplus": DatasetSpec("evalplus", "functional_smoke"),
    "livecodebench_v6": DatasetSpec("livecodebench_v6", "coding_transfer"),
    "swebench_lite": DatasetSpec("swebench_lite", "repository_development"),
    "swebench_verified": DatasetSpec("swebench_verified", "repository_final"),
}

OUTCOMES = (
    "PASS",
    "WRONG_OUTPUT",
    "RUNTIME_EXCEPTION",
    "TIMEOUT",
    "COMPILE_ERROR",
)
INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"

BASELINE_PROVENANCE_MODES = {
    "official_adapter",
    "paper_spec_reimplementation",
    "controlled_ablation",
}

HARDWARE_PROFILES = {"cpu", "16gb", "24gb", "4x24gb", "h200_formal"}
FORMAL_PROFILES = {"4x24gb", "h200_formal"}

EXPERIMENT_ARMS = {
    "raw_transcript",
    "last_observation",
    "window_1",
    "window_2",
    "window_4",
    "orderless_set",
    "pass_rate",
    "matched_gru",
    "matched_exchangeable",
    "map",
    "posterior_mean",
    "p_way_ensemble",
    "random_matched_norm",
    "shared_kv_delta",
    "pbpf_soft_prompt",
    "pbpf_low_rank_kv",
    "tokenwise_remixture_fault",
    "independent_sampling",
    "self_debug",
    "rex",
    "rlef_paper_spec",
    "ldb",
    "rollout_roulette",
    "rsp",
    "upskill_optional",
    "ladi_rl_audit_required",
}

UNAVAILABLE_EXECUTION_ARMS = {
    "matched_gru": "equal parameter count is implemented, but matched compute has not been established",
    "matched_exchangeable": "equal parameter count is implemented, but matched compute has not been established",
    "independent_sampling": "standalone independent-sampling runner is not integrated",
    "self_debug": "Self-Debug paper-spec runtime is not integrated",
    "rex": "official REx adapter is not integrated",
    "rlef_paper_spec": "RLEF paper-spec training runtime is not integrated",
    "ldb": "official LDB adapter is not integrated",
    "rollout_roulette": "Rollout Roulette paper-spec runtime is not integrated",
    "upskill_optional": "no verified UpSkill adapter is integrated",
    "ladi_rl_audit_required": "independent code audit has not passed",
}

# All config entry points resolve through this single availability registry.
EXPERIMENT_METHODS = {
    name: UNAVAILABLE_EXECUTION_ARMS.get(name) for name in sorted(EXPERIMENT_ARMS)
}
