# Training and checkpoint parity

Training is a transition `(parameters, model_state, optimizer_state, RNG, counters, batch) -> next_state, outputs`. Compare that transition, not just inference weights. Read official optimizer equations and installed implementations linked in [sources](sources.md).

## Ordered probes

1. With transferred parameters and fixed inputs/state, compare raw predictions and each unreduced loss term.
2. Compare reduction axes, valid-example/token counts, class weights, ignore index, label smoothing and regularization. A mean of microbatch means differs from a global token mean when lengths differ.
3. Compare parameter and input gradients in a canonical layout. Distinguish absent gradients from exact zeros, frozen leaves, sparse gradients, and shared parameter identities.
4. Feed identical saved gradients to both optimizers. Compare updates, slots, step counters and learning rate independently of model numerics.
5. Compare a complete training step, including every mutable collection and skipped-update policy.
6. Compare a short trajectory on the same batch sequence, then save/restore and compare the next step. Record per-step losses, update norms and drift, not merely a final scalar.

## Loss and differentiation

Cross-entropy may expect logits, log-probabilities, integer labels or probability labels. Preserve normalization, ignored-position denominator, smoothing distribution and stable log-sum-exp formulation. Check padding-only batches and zero valid tokens according to source behavior. Match auxiliary losses, regularization masks and scaling before comparing gradients.

Torch accumulates into `.grad` unless cleared; JAX transformations return gradient trees. Preserve clearing/accumulation boundaries, `None` versus zero, frozen parameters, and order of hooks. A zero gradient can still cause momentum or weight decay updates; an absent gradient can skip a Torch parameter entirely.

For a scalar loss, compare canonical gradients and selected directional derivatives. When practical, cross-check a tiny smooth FP64 case with finite differences across several step sizes. Exclude or separately classify nondifferentiable boundaries, stochastic evaluation and ill-conditioned points; a failed finite-difference check is not automatically a port bug. Use JVP/VJP inner-product duality as an additional derivative invariant. Test higher-order derivatives only if part of the contract.

Custom `autograd.Function` rules must preserve backward dependencies, saved tensors, custom JVP/VJP, batching and higher-order behavior when required. JAX `custom_vjp` and `custom_jvp` have different transformation constraints. Replacing a forward op does not port its backward rule. `detach` and `stop_gradient` define differentiation boundaries; rematerialization must not replay unintended state mutation or consume a different RNG stream.

## Optimizer equation audit

Build a small table of each configured transformation in execution order, with equations, parameter masks and state dtypes. Never infer equivalence from an optimizer name.

| Feature | Audit |
|---|---|
| SGD/momentum | First-buffer initialization, dampening, Nesterov order, maximize sign, LR application and skipped leaves. Compare steps 1 and 2; zero initial momentum can conceal a difference at step 1. |
| Adam family | First/second moment updates, bias-correction count, epsilon inside/outside square root, AMSGrad max state, moment dtype and parameter groups. Optax exposes `eps` and `eps_root`; map deliberately. |
| Weight decay | Coupled L2 enters gradients/moments; decoupled decay does not. Match LR scaling, exclusions, group masks, tied identities and update ordering. Do not rename either as equivalent regularization. |
| Clipping | Global versus per-leaf norm, norm order, epsilon, clipping before/after unscale, decay or accumulation, finite-gradient policy and distributed aggregation. |
| Accumulation | Sum versus mean, actual valid-token counts, final short accumulation window, loss scaling and optimizer/scheduler step frequency. |
| Schedules | Step 0 versus step 1, warmup boundary, epoch versus optimizer step, resume counters, skipped-step semantics and dynamic LR propagation into optimizer groups. |
| Mixed precision | Master weights, gradient scaling/unscaling, dynamic scaler growth/backoff, overflow detection across ranks, state-update behavior on a skipped parameter update. |
| Parameter groups | Every source group and mask maps by semantic name. Avoid flatten-order assumptions or accidentally including buffers/frozen/duplicate tied parameters. |

For FP32 Adam-like comparisons, also probe tiny gradients where epsilon dominates. For clipping, probe norm below, exactly at, and above the threshold. Use known supplied gradients, not a full network, to distinguish optimizer formula bugs cheaply.

## Stochastic training

First establish deterministic training parity with dropout disabled or source-derived masks injected through a documented test seam. This does not validate native stochastic behavior. Separately verify mask rate, scaling, broadcast axes, temporal correlation, replica independence and reproducible restart. A recurrent backend may fuse dropout and offer no direct mask seam; document that limitation rather than claiming exact native RNG parity.

When backend arithmetic prevents samplewise long-trajectory matching, use a predeclared statistical protocol: same initialization/data distribution, several independent seeds, paired seeds only when random draws genuinely correspond, confidence intervals for loss/metric differences, convergence/stability criteria and held-out task metrics. Set sample size and equivalence margins from task sensitivity, not after seeing results. Failure to reject a difference is not proof of equivalence. Tiny development runs cannot certify large-scale convergence.

## Checkpoints and resume

Record source format, revision, shard index, tensor key/shape/dtype, compression/quantization metadata, alias groups and transformation digest. Model weights, mutable state, optimizer slots, step/schedule/scaler counters and required RNG state belong in a resumable training checkpoint. Python/Torch/NumPy/JAX RNG and data-sampler position may be separate.

Load untrusted data only with safe formats or documented restricted deserialization. Do not enable `trust_remote_code` or unrestricted pickle because an old tutorial does. Review and pin any required custom code. For large checkpoints, inspect metadata first, stream shards, convert each value once, avoid simultaneous full source/target copies and use atomic writes in an owned directory.

Round-trip test: initialize a fresh model with deliberately different values, load converted state, verify exhaustive mapping and aliases, compare forward, then compare the next training step including slots and counters. A save/load test into identical initial values can pass even when nothing was loaded. Check missing/corrupt shards, extra/missing keys and incompatible metadata with small negative tests. A model-only reload must not be described as full training-resume parity.

Distributed optimizer checkpoints may encode global tensors, local shards, flattened slots or rank-specific RNG. Restore into a declared topology and verify global correspondence before claiming reshardability. Do not gather a multi-terabyte state onto one host as a default conversion strategy.
