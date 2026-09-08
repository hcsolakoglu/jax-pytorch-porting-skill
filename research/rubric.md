# Competitor rubric v1

Frozen before competitor scoring on 2026-09-08. Equal weights; twenty criteria, each 0 to 5, total 100. Assess a named skill and its directly linked supporting files, not unrelated capabilities elsewhere in its parent repository. Record revision and evidence locations. Group duplicates and forks; do not count identical mirrors as independent competitors.

## Shared anchors

0 = absent or no supporting evidence in inspected scope.
1 = mentioned without actionable procedure.
2 = partial actionable guidance with important missing safeguards.
3 = coherent usable procedure, explicit checks, and bounded limitations.
4 = strong coverage with concrete examples, failure handling, or reusable checks.
5 = unusually complete, technically specific guidance plus executable or independently documented validation for this criterion. A prose claim alone cannot earn 5.

Scores measure documented fitness for this bidirectional porting task. They do not establish general skill quality or controlled agent success rates. A specialist can be excellent for its own purpose yet score poorly here. Missing evidence is not proof that a feature is impossible.

## Criteria

| ID | Criterion | Evidence required for a high score |
|---|---|---|
| C01 | JAX coverage | PyTrees, state, transformations, precision, backend semantics |
| C02 | PyTorch coverage | Modules, state, autograd, modes, compiler/backend behavior |
| C03 | Bidirectional porting | Explicit mapping and workflow in both directions |
| C04 | Inference preservation | Observable API, preprocessing, outputs, shapes, modes |
| C05 | Training preservation | Loss, state, stochastic behavior, short trajectories, restore |
| C06 | Numerical validation | Calibrated budgets, multiple metrics, intermediate localization |
| C07 | Gradient validation | Parameter/input gradients, differentiation edge cases, checks |
| C08 | Optimizer validation | Update equations, decay, clipping, state, schedules, accumulation |
| C09 | Performance methodology | Equal work, warmup, synchronization, uncertainty, cold/warm split |
| C10 | Accelerator awareness | CPU/GPU/TPU, precision, sharding, resource safety |
| C11 | Debugging localization | First divergence, minimized cases, discriminating hypotheses |
| C12 | Failure recovery | Revert, retry budgets, blocked-state report, evidence-based restart |
| C13 | Fast fail design | Dependency-ordered cheap gates before expensive execution |
| C14 | Adversarial testing | Architecture-selected edge cases and negative controls |
| C15 | Regression protection | Rechecks after changes, checkpoints, test integrity |
| C16 | Harness portability | Standard packaging, verified installation, host-neutral workflow |
| C17 | Context efficiency | Compact entrypoint, explicit progressive loading, limited duplication |
| C18 | Documentation quality | Scope, sources, limitations, provenance, clear usable examples |
| C19 | Practical usability | Tested helpers, clear commands, small setup burden, actionable output |
| C20 | Evidence grounding | Primary citations, artifact-level observations, calibrated claims |

## Ranking policy

Sort by total descending, then C03, C06, C05, then stable name. Include score uncertainty and selection limitations. Report our score using identical anchors and evidence burden. Any rank is within this inspected set and rubric only. Do not claim rank #1 in real agent performance without matched, independent harness evaluations. Do not adjust rubric or scores to manufacture a winner. Future rubric revisions must rescore every entry and preserve this original record.
