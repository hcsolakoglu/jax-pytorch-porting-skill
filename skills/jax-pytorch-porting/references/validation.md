# Numerical, regression and adversarial validation

## Establish trustworthy oracles

Pin a working source revision and validate its expected task behavior. Use original parameters/configuration and source-only repeatability checks. When inputs or weights are synthetic, say so; tiny random networks do not replace representative trained-weight validation.

Test that the oracle and comparator are sensitive: intentionally perturb one mapped parameter, reorder one axis, change one mask bit or state value, and confirm an expected failure. Mark these as injected negative controls, not bugs discovered in a real port. A constant output, zero residual scale, saturation, dead activation or zero hidden state can conceal a broken implementation. Inspect intermediate activations and nonzero gradients; use source-supported alternative values to activate relevant paths.

Keep original-checkpoint and diagnostic active-path cases separate. Never alter a production checkpoint to make its test more convenient. For each claimed parameter path, establish that at least one fixture is sensitive to that path, or mark it untested. A forward-only normalization probe is insufficient: compare the next running-state value and a subsequent evaluation call as well.

Freeze goldens and test logic separately from target changes. Never derive expected values by calling the target. A source implementation and a mathematically independent high-precision diagnostic are different oracles with different purposes. Do not silently replace source behavior with a supposedly better formula.

## Numerical budgets, not one universal tolerance

For each component/mode, record reference dtype, accumulator policy, value scale, reduction length/order, depth, conditioning, hardware/backend, deterministic settings and task sensitivity. Use source repeatability and controlled source-only precision/backend comparisons to inform a budget before target outcomes. FP64 tiny calculations can diagnose drift but are not always a valid production baseline.

For finite real/complex values, an elementwise gate may use `abs(candidate-reference) <= atol + rtol*abs(reference)`. Near zero, absolute error dominates. A global max-relative error with a nearly zero denominator can be misleading; declare any floor. Do not widen a budget merely until tests pass. A change requires independent evidence, versioned rationale and re-evaluation of earlier results.

| Metric | Role and caveat |
|---|---|
| Shape/dtype/key/alias identity | Exact structural gates before arithmetic. Prevent broadcasting or silent conversion from hiding a mismatch. |
| Max and mean absolute error | Max catches local defects; mean describes typical drift but can hide a rare catastrophic value. |
| Relative/scaled error | Use a stated denominator policy and near-zero handling; report threshold violation count. |
| Percentiles | p50/p95/p99 help distinguish widespread drift from outliers; retain max and worst coordinates. |
| Cosine similarity | Directional diagnostic only. Scale/offset, one bad token, or wrong argmax can remain hidden. Handle zero vectors explicitly. |
| ULP distance | Useful for matching-format finite scalar/operator probes. Do not compare different formats or treat one ULP budget as a model-wide accuracy standard. Handle sign/zero/order correctly. |
| Activations/gradients/updates | Compare semantically aligned boundaries and per-leaf norms, not just concatenated whole-model summaries. |
| Distributions/trajectories | For native stochastic paths or unavoidable long-run drift; predeclare equivalence margins and uncertainty. |
| Task metrics | Necessary at representative scale, but aggregate accuracy cannot replace structural/gradient checks. Include margins or disagreement cases where relevant. |

Integers, booleans, indices, counters, masks and exact metadata normally require exact equality. Do not cast uint64/int64 to float and erase differences. NaN/Inf rules are explicit: default fail on unexpected nonfinite values; for specified propagation, compare locations and infinity signs separately. `equal_nan=True` is not a universal parity policy. Empty outputs need an explicit contract rather than vacuous success.

Validate comparator arithmetic too. Overflow in both error and tolerance can make `inf > inf` false and admit a mismatch. Complex NaNs require componentwise handling so a NaN real part cannot hide an incorrect finite imaginary part. Never silently downcast extended-precision or exact integer oracles. The bundled `scripts/parity.py` rejects unsupported precision and overflow, checks exceptional complex components strictly, stabilizes cosine/RMS diagnostics, and reports worst coordinates. It supports bounded NumPy float16/32/64 and complex64/128 tensors, not every framework dtype; for BF16, validate original dtypes first, then explicitly promote both sides for diagnostics without claiming FP32 execution.

## First-divergence localization

1. Align inputs, parameters, mode, state, dtype and actual random samples. Verify dump adapters with a known identity/permutation probe.
2. Capture model inputs/outputs and coarse block boundaries. Compare semantic values in canonical layout, not similarly named tensors that occur before versus after normalization.
3. Locate earliest failing boundary. Bisect within that dependency region, including residual branches, attention mask/cache, recurrent carry and mutable-state outputs.
4. Capture only the relevant tensors; attach key, shape, dtype, mode, step, source revision and digest. Avoid dumping an entire giant model by default.
5. Check an operator on a minimized nonsymmetric tensor; classify semantics, numerics, state, layout, initialization, compiler, hardware or evaluator.
6. Form one hypothesis and change one material variable. Preserve the failing fixture as a regression test when fixed.

Fusion can remove a boundary; compare an unfused diagnostic implementation or constrain compilation temporarily, then revalidate the actual fused path. Instrumentation can change execution and must not be included in performance measurements. A late output mismatch with matching teacher-forced tensors calls for a state/preprocessing/serving investigation, not another arbitrary graph rewrite.

## Risk-selected adversarial cases

Choose cases that discriminate actual risks; do not build an exhaustive Cartesian product.

| Risk | Small useful cases |
|---|---|
| Layout/broadcast | Unequal axes, non-square weights, transposed/sliced inputs, NCHW/NHWC position-coded values. |
| Padding/convolution/pooling | Odd/even/minimum spatial sizes, impulse corners, negative max-pool input, groups, dilation and last partial window. |
| Attention/masks | Batch 1, odd/short/maximum supported sequences, one valid token, all valid/invalid masks, ragged padding, nonzero cache offset, prefill versus decode. |
| Norm/reduction | Constant or near-constant inputs, large common mean with small variance, mixed signs, large/small magnitudes and reduction length. |
| Recurrence/state | Nonzero initial state, multiple chunks, last short chunk, reset versus carry, train/eval/train, restored state. |
| Embeddings/ties | Repeated tokens, padding index, first/last valid vocabulary entry, tied-gradient accumulation, alias-preserving reload. |
| Optimizers | Tiny/zero/absent gradients, clipping boundaries, step 1/2, group masks, decay, skipped updates and scheduler boundary. |
| Compilation | Same shape repeated without retracing, changed supported shapes, static configuration change, eager/compiled forward and backward. |
| Distribution | Uneven valid counts, local/global shapes, collective order, padding, per-rank RNG and resharded checkpoint. |

For invalid inputs, test documented errors rather than requiring a model to support an undefined domain. NaN/Inf, empty dimensions, maximum sizes and non-contiguous inputs are conditional on contract and resource budget.

## Advanced checks

Use differential, metamorphic and property-based testing where an independent expected output is expensive. Examples: batch permutation equivariance only for batch-independent eval computation; masked-token invariance only when masks fully exclude it; full-sequence versus chunked recurrent execution only with matching carry and no reset; linearity only for genuinely linear operators. Training BatchNorm, dropout draw order, cross-sample attention and batch-dependent losses invalidate naive metamorphic assumptions.

Sample a small covering array of high-risk interactions (e.g. odd sequence + mask + BF16, or tied weights + decay + resume) instead of only isolated flags. Hold out a few shape/value/config combinations from optimization selection. Repeatedly tuning against a holdout converts it into a development set; create a new untouched final set.

Differentiation checks: finite-difference step-size sweeps on tiny smooth cases, JVP/VJP duality and parameter-direction probes can expose shared forward/backward errors. Do not make expensive full-Jacobian construction a default. Sparse, quantized and complex differentiation require source-specific rules.

## Regression tiers

Every commit: imports, structure, deterministic unit parity, comparator negative controls. Every semantic/optimization change: affected forward, gradient, optimizer, state and checkpoint cases. Before release: full declared shape/mode/precision matrix, representative task metrics and required hardware. Cache immutable source fixtures by source/config/input/precision digest; invalidate deliberately when any part changes. Report both passing coverage and explicit gaps.
