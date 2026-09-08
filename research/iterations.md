# Substantive improvement log

Initial version `0.1.0`, commit `8dc8ece`, contains a complete staged workflow and eight focused references. Later entries record actual changes, not cosmetic renaming. External guidance, direct observations, designed procedures and untested areas remain distinct.

## Iteration 1: source-oracle sensitivity and state mismatch

- **Weakness:** generic forward parity can miss a wrong residual branch or a wrong normalization state update. Initial guidance described the risk but had no direct source-derived demonstration.
- **Evidence:** `validation/results/source-calibration.json`, executed via `python -m validation.source_probe` on CPU. Eight original ResNet18 terminal BatchNorm scales were zero. Source FP32/FP64 eval max drift was 1.89249e-7. Native Flax/Torch BatchNorm forward max drift was 3.57628e-7, while moving variances were 0.94725 versus 0.954 despite complementary momentum settings.
- **Proposed/implemented change:** add a pre-target source probe, freeze explicit development budgets, and require source-supported activation of otherwise dead paths plus direct state-transition checks. Negative controls must distinguish weak fixtures from correct implementations.
- **Generalization:** applies to zero-initialized residual/adapter branches, saturated activations, empty masks, zero recurrent states, EMA statistics and any stateful layer; it is not a rule to mutate production checkpoints.
- **Regression/contradiction check:** original source hashes remain unchanged. Diagnostic active-branch fixtures are separate from original-initialization tests. The calibration is CPU-only and does not set production tolerances.
- **Result:** source repeatability and the BN state discrepancy are observed; target correctness is not yet tested. Next step is implementing explicit source-equivalent behavior.

## Iteration 2: genuine ResNet stateful port and conversion coverage

- **Weakness:** initial procedures did not distinguish an axis-bound single-device training probe from actual distributed validation, or a target-native checkpoint round trip from source-state conversion.
- **Evidence:** `validation/results/resnet-forward.json` (46 comparison records) and `resnet-training.json` (645 records). Executed original Flax ResNet18 and unchanged AST-selected original `train_step` on one CPU replica. Tested basic blocks, projection paths, asymmetric SAME padding, active residuals, intermediate tensors, Nesterov momentum, L2 gradients, moving statistics, two updates and one resumed update.
- **Proposed/implemented change:** explicit source-axis binding guidance; source-to-target parameter/buffer/optimizer mapping; separate converted-state, native-load and resumed-transition checks. Added a native PyTorch model, auditable mapping and a bounded reproducible validator. Added strict NumPy comparison helper and seven negative/edge unit tests.
- **Generalization:** collective-axis binding applies to BatchNorm and reductions in transformed source code. Mapping-plus-resume applies across optimizers and serialization formats, not only SGD/Flax/Torch.
- **Regression/contradiction check:** frozen budgets and original source hashes unchanged; no target-port references; no GPU, TPU, multi-device, ImageNet accuracy or Orbax-file certification claimed. Cheap comparator gates ran before model validation.
- **Result:** required bounded JAX-to-PyTorch model/training port passes its declared numerical scope. Performance remains unmeasured at this iteration.

## Iteration 3: recurrent semantics, alias identity and stochastic evidence levels

- **Weakness:** a generic dropout or checkpoint gate could be overread as native stochastic equivalence; zero hidden state and equal-but-untied weights could provide false confidence.
- **Evidence:** `validation/results/rnn-training.json`: four original PyTorch recurrent families and a tied LSTM pass inference, parameter/hidden-state gradients, three truncated training chunks and one resumed step. The first run produced 430 comparison records. Source code and official GRU equations establish reset-after-affine behavior and separate input/recurrent biases. The original clipping implementation includes epsilon, which the port preserves explicitly.
- **Proposed/implemented change:** native JAX `scan` port and training step, nonzero-state probes, short final chunk, unique tied-parameter checks, injected dropout masks, native key-repeatability checks, and explicit separation of four stochastic evidence levels. RNG checkpoint records now include algorithm/dtype and a next-draw test.
- **Generalization:** parameter identity matters to all tied/shared networks; explicit stochastic evidence levels apply to dropout, augmentation, diffusion noise and distributed RNG. Raw-key metadata applies beyond either validation model.
- **Regression/contradiction check:** original files and frozen numerical budgets unchanged. Injected dropout is labeled a diagnostic seam, not native multi-layer RNG parity. Native fused inter-layer dropout distributions, corpus perplexity, long convergence and accelerators remain untested. No numerical failure occurred in the initial port run; that is not proof of exhaustive correctness.
- **Result:** bounded PyTorch-to-JAX inference and training port passes. Added RNG-resume regression is required before this iteration is committed.

## Iteration 4: interruption recovery and evidence-preserving checkpoints

- **Weakness:** a prior final response falsely reported no verified PC work after a tool-discovery failure, despite durable commits and passing validation reports. Local history was ahead of the remote.
- **Evidence:** recovery inspected private `hcsolakoglu/jax-pytorch-porting-skill`, HEAD `238ae95`, five existing commits, PASS reports with 46 ResNet forward, 645 ResNet training and 435 recurrent comparison records. The Local MCP read probe and GitHub identity/visibility query succeeded. No active tracked jobs remained. This establishes an erroneous summary, not the internal cause of the earlier discovery failure.
- **Proposed/implemented change:** add idempotent recovery rules for missing tools and ambiguous write responses; retain prior proof scope; reconcile HEAD, files, jobs and remote before restart; maintain a compact status note and push substantive stages.
- **Generalization:** applies to any harness with connection loss, context reset, interrupted writes or stale worker memory. Exact namespaces and backoff are host-dependent; no Local-MCP-specific dependency is imposed on the skill.
- **Regression/contradiction check:** no oracle, budget or model implementation changed. Existing commits were pushed and HEAD/origin divergence verified as zero. Bounded retries preserve stopping rules rather than create endless polling.
- **Result:** real work recovered and backed up; current access verified. Internal root cause of the prior tool failure remains unknown.

## Iteration 5: adversarial attack on the numerical oracle

- **Weakness:** the comparator could accept different finite complex components behind NaNs, accept overflowed error/budget comparisons, erase long-double differences through FP64 casting, and emit NaN cosine for large equal values.
- **Evidence:** four newly added negative tests failed against the previous helper (4 failed, 7 passed). These are observed helper defects, not defects in either model port.
- **Proposed/implemented change:** componentwise exceptional-value comparison, reject unsupported extended precision and arithmetic overflow, scale cosine/RMS computations, add p99 and worst-coordinate diagnostics. No parity tolerance changed.
- **Generalization:** relevant to gradients, complex operators, large reductions, exact checkpoint fields and any numeric comparison, independently of architecture or port direction. Unsupported-range rejection is explicit rather than claiming universal numeric coverage.
- **Regression/contradiction check:** 11 helper tests pass; serialized CPU revalidation passes 645 ResNet training and 435 recurrent comparison records with unchanged budgets. Exceptional slots require exact finite-component agreement, documented separately from ordinary tolerance comparisons.
- **Result:** four false-confidence cases are mechanically guarded; both genuine ports remain valid within their recorded development scope.

## Iteration 6: extract failure and recovery mechanisms from released trajectories

- **Weakness:** a simple correctness flag could admit invalid timing sentinels; final-attempt selection discards earlier working candidates; repeated identical code can look like productive iteration; failures can be mislabeled as compilation errors.
- **Evidence:** twelve matched KernelBench logs, 120 attempts, pinned dataset revision and hashes in `trace-cohort.json`; detailed code-transition review in `agent-traces.md`. V3 LayerNorm includes correctness passes with runtime -1; R1 matmul and composite end in compile failure after valid incumbents; repeated LayerNorm code and redundant quadratic reductions show wasted compute.
- **Proposed/implemented change:** timing eligibility checks and a unit test, remove misleading standalone-feedback interpretation, separate correctness/performance incumbents, configuration-aware hash deduplication, static work/traffic audit and minimal API probes in recovery guidance.
- **Generalization:** mechanisms apply to graph compilers, checkpoint conversion, stateful training and kernel work in both directions. Benchmark permissions to retain native PyTorch operations do not transfer into a requested source-free JAX runtime.
- **Regression/contradiction check:** 12 cheap tests pass; all 120 records remain represented. Published timings are marked external and are not treated as our measurements or current global agent rankings. No generated CUDA code was executed, and no validation budget changed.
- **Result:** trace analysis is grounded in complete released attempts, including unsuccessful trajectories and explicit evidence gaps.
