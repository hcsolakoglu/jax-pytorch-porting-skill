# Contributing

Keep the core small. Add a reference only when it changes an agent's decision or prevents a concrete failure. Preserve native target behavior and do not turn this repository into a conversion framework.

Every substantive change should identify its failure mode, evidence, generalization beyond one architecture, relevant test and expected effect on existing contracts. Use `research/iterations.md` for skill-development changes. Do not weaken a golden, tolerance, workload or no-regression threshold to make a patch pass.

Run the inexpensive checks before optional model experiments:

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check tools tests validation skills/jax-pytorch-porting/scripts
.venv/bin/python tools/render_comparison.py --check
.venv/bin/python tools/check_repo.py
.venv/bin/python tools/package_skill.py
```

Model or evaluator changes invalidate saved validation fingerprints. Re-run the relevant bounded commands in `validation/README.md` before making numerical/performance claims. Do not start accelerator jobs for documentation-only changes.

Record source revision, dependency versions and measured scope. Label EXTERNAL, OBSERVED, DESIGNED and UNTESTED evidence separately. New benchmark numbers require raw samples, equal-work configuration, synchronization, warmup and honest uncertainty. Preserve negative results and known-good commits.

For competitor updates, apply the same versioned rubric to every row. Do not score by keyword frequency or parent-repository popularity. Distinguish retrieval, metadata screening, full entrypoint review and controlled execution.

Do not commit secrets, private datasets, source checkpoints, raw conversation dumps, caches or machine-specific credentials. Keep this repository private unless its owner explicitly authorizes publication. Report suspected secret exposure privately through an authorized owner channel, not by pasting credentials into an issue. Do not run untrusted setup scripts or unrestricted pickle loaders.
