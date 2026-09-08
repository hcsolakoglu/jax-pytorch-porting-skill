# Minimal project records

These are forms to populate, not evidence that a test was run. Use a few existing project files rather than copying every template into a new framework.

## Port contract

```yaml
source: {repository: required, revision: required, license: required}
target: {framework: required, version: required, module_system: required}
environment: {python: required, dependencies: lockfile, hardware: required, backend: required}
scope:
  inference: []
  training: []
  shapes_modes_precision: []
  preprocessing_postprocessing: []
  checkpoint_resume: []
  exclusions_with_reasons: []
acceptance:
  numerical_budgets: versioned_registry
  task_metrics: []
  performance_baselines: [B0, B1, T0]
  maximum_regression: user_contract
budgets: {wall_time: required, memory: required, accelerator_time: required, candidates: required}
clean_room: {required: false, allowed_sources: [], blocked_target_ports: []}
```

## Semantic mapping

| Source key/location | Role | Source shape/dtype | Target key/shape | Transform | Alias group | Coverage/test |
|---|---|---|---|---|---|---|
| Populate before conversion | parameter/buffer/slot/counter | exact | exact | identity/transpose/split/merge | explicit | fixture + result |

Every source value has a disposition and every required target value is populated. Explain exceptions; no silent missing keys.

## Numerical budget registry

| Component/mode | Dtype/accumulator/backend | Metric and budget | Source-only calibration | Task rationale | Freeze revision |
|---|---|---|---|---|---|
| Populate before target evaluation | exact | explicit | repeatability/precision probe | sensitivity | immutable identifier |

## Test matrix

| Case | Risk/invariant | Shape/mode/precision/backend | Oracle | Cost/deadline | Status | Artifact/command |
|---|---|---|---|---|---|---|
| Unique ID | why this case matters | exact | pinned | bounded | PASS/FAIL/UNTESTED/N/A | reproducible |

## Experiment note

```text
Question and current failing invariant:
Evidence and first divergence:
One hypothesis and predicted result:
Change and baseline revision:
Command, resource bound and environment:
Observed result and raw artifact:
Decision: keep / revert / gather evidence / blocked
Generalization limits and regression checks:
Next cheapest discriminating action:
```

## Benchmark record

Record case/config/input digest, source and target revisions, hardware/topology, threads, precision settings, mode/compiler, warmup policy, synchronization, timing boundary, cache regime, repeats, raw sample units, median/IQR/p95, uncertainty, throughput denominator, compile/startup cost and separately measured memory. Include failed cases and all declared baseline results, not only successful speedups.

## Final port report

State source/target and exact supported scope. List gate status and representative commands. Separate structural, forward, intermediate, gradient, optimizer, training-state, checkpoint, task-metric and performance evidence. Report deviations and failures plainly. Distinguish EXTERNAL practice, OBSERVED result, DESIGNED procedure and UNTESTED area. Include remaining hardware/scale/harness gaps and cleanup status.
