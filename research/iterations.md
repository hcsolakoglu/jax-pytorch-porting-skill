# Substantive improvement log

Initial version `0.1.0`, commit `8dc8ece`, contains a complete staged workflow and eight focused references. Later entries record actual changes, not cosmetic renaming. External guidance, direct observations, designed procedures and untested areas remain distinct.

## Iteration 1: source-oracle sensitivity and state mismatch

- **Weakness:** generic forward parity can miss a wrong residual branch or a wrong normalization state update. Initial guidance described the risk but had no direct source-derived demonstration.
- **Evidence:** `validation/results/source-calibration.json`, executed via `python -m validation.source_probe` on CPU. Eight original ResNet18 terminal BatchNorm scales were zero. Source FP32/FP64 eval max drift was 1.89249e-7. Native Flax/Torch BatchNorm forward max drift was 3.57628e-7, while moving variances were 0.94725 versus 0.954 despite complementary momentum settings.
- **Proposed/implemented change:** add a pre-target source probe, freeze explicit development budgets, and require source-supported activation of otherwise dead paths plus direct state-transition checks. Negative controls must distinguish weak fixtures from correct implementations.
- **Generalization:** applies to zero-initialized residual/adapter branches, saturated activations, empty masks, zero recurrent states, EMA statistics and any stateful layer; it is not a rule to mutate production checkpoints.
- **Regression/contradiction check:** original source hashes remain unchanged. Diagnostic active-branch fixtures are separate from original-initialization tests. The calibration is CPU-only and does not set production tolerances.
- **Result:** source repeatability and the BN state discrepancy are observed; target correctness is not yet tested. Next step is implementing explicit source-equivalent behavior.
