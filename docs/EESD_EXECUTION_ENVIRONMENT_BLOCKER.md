# Required execution environment

Candidate generation is running. Candidate execution and scientific scoring are
blocked in the current container: bubblewrap cannot create its namespaces.
The process has `Seccomp: 2`; this observation alone does not identify which
host policy denies namespace creation. Installing bubblewrap and enabling the
user-namespace sysctls inside the container did not resolve the actual probe.

## Host-side action

Provide a Linux execution environment in which the project's rootless
bubblewrap sandbox can create user, mount, PID, network, IPC and UTS namespaces.
Check the container runtime's seccomp and host LSM policies. Preserve the
sandbox's network isolation and restricted mounts. The current process cannot
relax an inherited seccomp filter; changing Python or test expectations does not
resolve this prerequisite.

Preserve `/root/PBPF`, its virtual environment, model cache and run artifacts
when changing the environment. Coordinate any container replacement with the
live generation supervisor to avoid interrupting active writers or duplicating
jobs. An alternative is a separate compatible execution machine with the same
locked inputs and evaluator environment.

## Acceptance probe

From `/root/PBPF`, choose a new output directory and run:

```bash
.venv/bin/python scripts/evaluate_eesd_evalplus_isolated.py \
  --probe-only --output runs/eesd-setup/host-readiness-after-change
```

Expected: exit 0 and `status.json` containing `probe_passed`. This probe does
not mount or execute generated candidates. A failed probe records an
infrastructure error rather than a model error. Existing failure evidence is
in `runs/eesd-setup/evalplus-isolation-probe-20260920-attempt2/`.

This is an infrastructure acceptance check, not an experiment result. After it
passes, also run the existing RunBugRun and CodeARC execution preflights before
starting their official scoring queues.
