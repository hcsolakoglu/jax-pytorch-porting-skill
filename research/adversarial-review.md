# Final adversarial review

Reviewed 2026-09-08 by the implementing agent using code, original sources, result fingerprints, negative tests and explicit hypothetical walkthroughs. No independent blinded reviewer or prospective multi-harness agent experiment was run. Static reasoning is labeled separately from executed evidence.

## Material findings and disposition

| Finding | Evidence | Resolution |
|---|---|---|
| Tool discovery failure erased previously verified work in a final response | Repository commits and passing reports contradicted the message | Reconcile durable state, preserve proof scope, inspect ambiguous writes before retries, push checkpoints |
| Complex NaNs hid an incorrect finite component | Negative test failed before repair | Componentwise exceptional comparison; regression passes |
| Overflowed error and tolerance could produce false success | Negative test failed before repair | Reject nonfinite comparison arithmetic; regression passes |
| Wide floating oracle silently narrowed to FP64 | Negative test failed before repair | Unsupported extended precision fails closed |
| Large equal values produced nonfinite diagnostics | Negative test failed before repair | Scaled cosine/RMS; finite JSON diagnostics |
| An old PASS could open a benchmark after code changes | Initial reports lacked a freshness gate | Code/budget/dependency fingerprint checked before benchmark and before saving proof |
| Correctness from one mode could appear to authorize another mode's timing | Static walk through gates 13 and 16 | Core explicitly requires relevant correctness on the exact backend/dtype/compiler before timing; later matrix only expands coverage |
| Package could accept missing/outside references and a root symlink | Three packaging negatives failed before repair | Fail-closed local-link and symlink checks; tests pass |
| Installation success could be mistaken for Antigravity discovery | Isolated Vercel installation versus two differing official global paths | Document actual paths, product ambiguity and project-scoped fallback |
| A documentation score could hide failed model performance | ResNet eager and Inductor CPU results are both slower than JAX | Preserve failed performance gates in README, reports and release checks; no general speed guarantee |

Executed cheap suite: **18 tests passed**, including comparator, timing eligibility, score-shape/uniqueness and packaging negatives. Native model proof is recorded separately: 46 ResNet forward, 645 ResNet training and 435 recurrent comparisons. These counts are records, not independent production configurations.

## Risk-selected walkthroughs

These are **DESIGNED/static review scenarios**, not claims of newly executed model experiments unless an observed report is named.

| Attack or edge case | Required decision / earliest useful gate | Evidence status |
|---|---|---|
| Missing checkpoint key hidden by `strict=False` | Exhaustive mapping fails before forward timing | Native converters enforce strict state coverage; code reviewed |
| Identical random seeds but different masks or initialization | Inject exact arrays; test native stochastic behavior separately | Both ports and RNG-resume tests executed |
| Zero residual scale hides a wrong branch | Active-branch negative fixture in addition to unmodified source initialization | ResNet source calibration and active fixtures executed |
| High cosine with wrong amplitude or one wrong token | Exact structure plus calibrated element/discrete gates, not cosine alone | Comparator negatives executed; large-LLM token scenario untested |
| All-masked attention differs by backend | Source-specific policy and gradient probe; no blind NaN cleanup | Operator mismatch observed; full transformer port untested |
| GRU reset before rather than after hidden affine | Nonzero hidden state and gradient localization | Recurrent port executed |
| Tied embedding serialized twice and updated twice | One trainable identity and explicit alias disposition | Tied recurrent gradients/updates executed |
| Adam epsilon/weight decay discrepancy invisible in large gradients | Supplied tiny/zero-gradient steps before model training | Three independent AdamW steps executed |
| Unequal microbatch sizes, ignored labels or DDP averaging | Compare effective loss denominator/global sample weighting | Procedure reviewed; distributed execution untested |
| Wrong TP/DP shard scale looks correct on rank zero | Exact global/per-rank shapes, collective semantics and per-rank maximum timing | Procedure reviewed; multiple devices untested |
| `vmap(cond)` and `scan(cond)` assumed equally cheap | Inspect actual transform composition and IR | JAXPR probe executed; hardware timing not measured |
| Custom VJP/JVP works forward but has wrong gradients | Directional checks and adjoint duality, excluding nonsmooth singular points | Basic duality probe executed; arbitrary custom kernels untested |
| TF32/mixed precision silently changes a benchmark | Record and align actual op-specific math policy; separate approved precision experiment | CPU FP32 measured; TF32/BF16 accelerator regimes untested |
| Compile time omitted only from target | Report startup, first call and steady-state separately | Three bounded CPU reports preserve first-call costs |
| A negative timing sentinel wins optimization | Require finite positive correct/no-error result | Trace integrity test executed |
| Agent overwrites a working candidate with its last failure | Correctness and performance incumbents plus hashes | Twelve released trajectories reviewed; prospective agent benefit untested |
| Three unrelated fixes leave earliest failure unknown | Reset to invariant, minimum case and one discriminating hypothesis | Recovery policy reviewed; no claim of universal success rate |
| Existing target port appears in search results | Existence-only exclusions; never inspect target implementation or derived explanations | Contamination log preserved for both selected projects |
| Skill works only in a checkout, not its ZIP | Validate all bundled links, metadata, file allowlist and archive integrity | Packaging unit tests executed |
| Local installation assumed available in remote/cloud ChatGPT | Verify actual host discovery/permissions and upload route | Isolated file installation executed; ChatGPT UI upload and agent invocation untested |

## Coverage and generalization audit

JAX and PyTorch operator/state conventions, both directions, inference and training, numerical budgets, gradients, optimizer state, regression tests, adversarial cases, checkpoint resume, compilation, performance boundaries, provenance, failure recovery and resource cleanup all have explicit guidance. No single tolerance or architecture-specific shape is prescribed as universal.

Convolutional and recurrent source paths are genuinely executed; attention, activation, optimizer and transformation probes add independent coverage. Sparse, quantized, diffusion, large transformer, very large checkpoint, BF16/FP16 accelerator and multi-host regimes remain procedure-level coverage. The skill directs future agents to risk-select and validate those paths rather than imply that small CPU tests certify them.

Context review: the main skill remains compact with eight conditionally loaded references. Research, competitor tables, long trace analysis and validation source trees are outside the ZIP. No agent model/provider choice, forced delegation, global configuration mutation or accelerator provisioning is baked into the skill.

Repository hygiene review: original source licenses and hashes are preserved; source snapshots, environments, compiler caches and raw traces are ignored; package allowlist excludes them. Cheap CI is CPU-only and action revisions are pinned. A successful repository release check is explicitly not a successful ResNet performance gate.

## Remaining material limits

No GPU/TPU/CUDA/distributed validation, full-dataset accuracy, long-run convergence, independent competitor scoring or matched multi-harness agent trials were performed. ResNet no-regression performance failed for both measured CPU target modes. These gaps are disclosed, not converted to passes. The skill-development deliverable can be used with this evidence boundary; a production model port must still pass its own required workload/backend gates.
