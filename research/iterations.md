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
