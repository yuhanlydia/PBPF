# Environment readiness, 2026-09-16

This is an environment check, **not a successful scientific gate or a complete
APBPF experiment**. The diagnostic reports in this directory retain their
original failing gate results and original CPU execution environment.

## Runtime and models

An isolated Conda environment at `/root/miniforge3/envs/pbpf-py311` now uses
Python 3.11, Torch 2.11.0+cu128, CUDA 12.8, Transformers 4.57.6, and
bitsandbytes 0.48.2. `pip check` passes. A GPU matrix multiplication agrees with
the CPU reference, and a bitsandbytes NF4 linear forward produces finite values.
NVIDIA runtime dependencies were installed as copies from locally reconstructed
wheels whose source payloads matched installed RECORD hashes; these are not
claimed to be byte-identical upstream wheel archives. Torch and Triton were
downloaded as official wheels. The system Python environment was not modified.

Both pinned model snapshots were downloaded, all safetensors shards checked
against their upstream LFS SHA-256 values, and shard keys checked against their
indexes. Loading used only local files, with `trust_remote_code=False`.

| Model | Immutable revision | Tensors | Peak allocated GPU bytes |
| --- | --- | ---: | ---: |
| Qwen/Qwen2.5-Coder-7B-Instruct | c03e6d358207e414f1eca0bb1891e29f1db0e242 | 339 | 5890939392 |
| deepseek-ai/deepseek-coder-6.7b-instruct | e5d64addd26a6a1db0f9b863abf6ee3141936807 | 291 | 6608392192 |

Each model completed offline greedy generation of 48 tokens for a trivial
addition-function prompt using NF4 double quantization and bfloat16 compute on
an RTX 5090. A 28% process GPU memory limit was enforced; unrelated GPU jobs
were not stopped. These checks establish executability, not benchmark accuracy.
Operator logs are `runs/qwen-gpu-readiness.log` and
`runs/deepseek-gpu-readiness.log` in the preparation workspace.

To fit the persistent disk, Qwen shard 1 and DeepSeek shard 2 are verified
RAM-backed cache files under `/dev/shm`, reached through the normal HF cache
symlinks. **They are ephemeral and may need downloading again after restart.**
The committed diagnostic results and checkpoint archive do not depend on those
cache files for preservation.

## Regression verification

After the CUDA runtime change, the full offline/non-GPU suite initially had
726 passes, one failure, and one optional skip. The failing resampling test
assumed a particular random draw from seed 373. The fixture now supplies
explicit proposal noise and a zero systematic offset, checking zero-mass
boundaries and exact selected indices without changing production code.
The 35 focused neural tests pass, independent review found no issues, and a
mutation replacing right-sided with left-sided boundary search is detected.

The complete rerun finished successfully: **727 passed, 1 skipped in 358.50s**.
The skip is the optional TeX/Poppler paper-build check. Command:

```bash
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /root/miniforge3/envs/pbpf-py311/bin/python -m pytest -q -m 'not gpu and not network'
```

## Remaining scientific work

The real-profile doctor reports valid configuration and no missing data/cache
directories, but all 21 stage commands remain unprovisioned. The repository's
`docs/APBPF_WORKERS.md` explicitly describes prospective infrastructure, not a
complete set of scientifically validated adapters. Model readiness does not
fill this gap. No smoke-stage output is presented as real evidence.

The newer RunBugRun v2 Python projection is prepared separately from the legacy
data used in the committed diagnostic. Its official validation and test labels
share 226 problem IDs, so a future formal materializer must establish the
required source-component isolation instead of assuming those labels suffice.
Existing failed diagnostic gates must not be repaired by tuning on held-out
results, lowering thresholds, or relabeling negative evidence as success.
