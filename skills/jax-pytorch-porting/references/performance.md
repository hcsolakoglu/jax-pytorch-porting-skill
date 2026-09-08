# Fair measurement and safe optimization

Correctness gates precede candidate speed claims. A source-only feasibility profile is allowed earlier. Read JAX benchmarking, PyTorch benchmark/profiler and numerical-accuracy sources in [sources](sources.md).

## Baselines and measurement contract

Preserve B0 (original configuration), B1 (reasonably tuned source with unchanged semantics), T0 (correct minimal target), and candidate T1. Use equal tuning budgets where comparing framework capability. Report against B0 and B1; beating an intentionally weak eager baseline is not evidence of beating a tuned source.

Match model/config/checkpoint, precision including accumulator policy, mode, batch and sequence distributions, input placement, output materialization, CPU thread limits, device/topology, compiler options, RNG/cache state, allocation and data loading. If a boundary differs, provide separately labeled model-only and end-to-end measurements. Changing algorithm, quantization, attention approximation, sparsity, precision or task quality is a different experiment, not a silent port speedup.

## Procedure

1. Verify source and target correctness on the exact benchmark configuration and representative values.
2. Measure cold process/import/load/startup separately. Record first-call/compile/autotune cost and compile-cache state. Do not mix a cached target with an uncached source.
3. Warm each shape/mode/backend until compilation and one-time setup finish and timing stabilizes within a declared budget. Do not assume a fixed one-call warmup is sufficient.
4. Synchronize pending work before starting a latency timer; execute; wait for all measured outputs and required state/collectives before stopping. JAX uses `block_until_ready` over relevant outputs; Torch CUDA uses appropriate event or device/stream synchronization. CPU wall-clock around asynchronous dispatch is not execution latency.
5. Use repeated samples and alternate or randomize source/target order. Keep raw samples, median, IQR/MAD, p95 when meaningful, sample count, units, variability and failures. A minimum run is diagnostic, not an honest production headline.
6. Estimate uncertainty for the comparison, preferably using paired/blocked samples or repeated independent sessions. Autocorrelated repetitions are not independent evidence. If noise exceeds claimed gain, stop optimizing or improve measurement isolation.
7. Measure throughput separately with realistic batching/concurrency; per-call synchronization can destroy intended overlap. State whether timing is isolated latency or sustained pipeline throughput.
8. Measure memory in a separate controlled run if instrumentation perturbs timing. Distinguish live allocated tensors, allocator reservations/caches, device process memory, host RSS and compile-time peak. Torch allocator counters do not include every external CUDA/JAX allocation.
9. Recheck correctness after material changes and across held-out risk-selected shapes. Commit an optimization only with reproducible improvement and maintained contract.

For tiny functions, amortize timer overhead over a declared repeat block, still preserving state semantics. Do not repeatedly run a training step against stale inputs or reset parameters on only one side. Decide whether state reset, data transfer and checkpoint conversion are included, and do the same work in both implementations. Prevent dead-code elimination by consuming actual outputs/state; constant captured inputs/weights can accidentally benchmark a different program.

## Caches, shapes and deployment

Separate weight conversion, compilation, input preprocessing cache, KV/recurrent cache and hardware cache. Report cold and warm regimes relevant to deployment, not an arbitrarily favorable cache. A repeated tiny tensor fitting in L2 does not represent streaming large data. CUDA graphs change launch overhead and buffer-lifetime requirements; use them on both sides only when production allows equivalent capture.

Benchmark batch-size and sequence-length sensitivity, including decoding versus prefill, odd boundary shapes and actual distribution quantiles. Static shape buckets can reduce retracing but add padding and memory; count that work. Bound cache growth and compile cardinality. Dynamic shapes may avoid recompilation but choose slower kernels; measure the tradeoff rather than prescribing one mode.

For distributed latency, synchronize according to the declared contract, measure every rank, and use the maximum rank duration for each sample before summary statistics. Report topology, world size, global/local batch, collective backend, communication overlap and load imbalance. Do not compare different tensor-parallel or data-parallel sizes as equal work without a scaling question.

## Optimization selection

Use profiles and a simple cost model to choose the next experiment. If a fraction `f` of total time is improved by factor `s`, ideal overall speedup is bounded by `1 / ((1-f) + f/s)` before added overhead. This helps reject low-impact kernel work. Roofline or bandwidth estimates are diagnostics; state operation/FLOP/byte accounting and hardware assumptions.

| Candidate | Useful when | Preserve/check |
|---|---|---|
| Remove copies/conversions | Repeated contiguous/cast/transposes or duplicate checkpoint materialization dominate. | Alias/lifetime contract, layout, dtype, strides; distinguish view from copy. |
| Reduce transfers/sync | Python logging, `.item()`, NumPy conversion or host callbacks serialize execution. | Required output visibility, logging semantics and device memory budget. |
| Vectorize/batch | Independent work is Python-bound. | RNG stream identity, batch-dependent normalization, memory and cross-sample behavior. |
| Compile/fuse | Stable repeated tensor regions dominate dispatch or memory traffic. | Forward/backward numerics, graph breaks, state mutation, recompilation, startup. |
| Library attention/convolution | Equivalent supported fast kernels exist. | Masks, scale, dropout, dtype, accumulation, determinism and supported shapes. |
| Static buckets/cache | A bounded recurring shape distribution exists. | Padding masks/work, cache limit, config key correctness and cold misses. |
| Scan/loop transforms | Long recurrent Python unrolling inflates traces or overhead. | Carry dtype/shape, detach boundary, step order, state/RNG and actual runtime. |
| Rematerialization | Activation memory limits useful batch/sequence size. | Additional compute, stochastic replay, mutable state, backward parity. |
| Sharding/overlap | A measured memory/compute/communication constraint justifies complexity. | Global math, collectives, per-rank RNG, checkpoint and topology. |
| Mixed precision | Contract explicitly allows a quality tradeoff or source uses it. | Accuracy/error budget and training stability; do not silently relax parity. |

Do not assume vectorization, custom kernels, pre-transposing weights, more workers or more compilation is faster. Revert unsuccessful performance changes. Stop at a maintainable solution that meets requirements; retain comprehensive test procedures for later production scale without running them all during small development validation.

## Evidence freshness and bounded negative results

Bind every correctness prerequisite and timing report to source/target/evaluator code hashes, frozen budgets, dependency versions, input hashes and actual hardware/configuration. A previous PASS with a different fingerprint is not a benchmark gate. If code changes during a run, invalidate that run. Match available CPU capacity as well as framework thread settings: one runtime constrained to one thread versus another allowed multiple cores is not a matched single-core comparison.

Report a slower correct port as a performance failure when a no-regression gate applies. Do not delete the slow case, change precision, change batch size, time only an easier subgraph or prolong optimization indefinitely to create a success claim. Preserve valid implementations and document the measured bottleneck, tested candidates, cold/warm costs and a bounded next investigation. A development exercise may end with a useful failed performance gate; that does not authorize a production release requiring the gate. Combined-process peak RSS cannot be presented as per-model peak memory. A one-session paired bootstrap is descriptive, not independent deployment confidence.
