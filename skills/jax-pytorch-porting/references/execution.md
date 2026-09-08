# Transforms, compilers and accelerator execution

Version-check current APIs before applying a remembered recipe. CPU evidence does not certify CUDA, TPU, XLA, Inductor or distributed execution. Sources are indexed in [sources](sources.md).

## JAX transforms

Design a pure compute boundary with explicit arrays/state/RNG. Keep I/O, logging, mutable Python containers and data-dependent Python side effects outside transformed regions. Static metadata must be immutable or explicitly reflected in cache keys; a changed object field captured by a cached function may not change execution.

Use `eval_shape`/abstract shapes for cheap structural feasibility where supported. `jit` compiles a function specialized by relevant shape/dtype/static/sharding information. Reusing a stable function object avoids accidental retracing from repeatedly constructed lambdas/partials. Keep compilation count and memory bounded. Compiling the outer useful region is often better than individually jitting many tiny ops, but measure.

`vmap` is batching semantics, not free parallel speed. Record in/out axes and how parameters, state and random keys are shared or mapped. `lax.scan` carries fixed-shaped/dtyped state and can avoid enormous Python-unrolled graphs. Dynamic `while_loop` and static loops have different reverse-mode constraints; validate needed derivatives. `cond` branches require compatible outputs and tracing semantics; do not use Python `if` on a tracer.

Donation permits buffer reuse and invalidates donated values; exclude aliased or subsequently read oracle buffers. Rematerialization/checkpointing trades compute for memory and must preserve stochastic/state behavior. Custom JVP/VJP, callbacks, FFI and custom primitives require explicit transformation support, not just a passing forward call.

For older `pmap`/`pjit` code, preserve source axis/collective semantics first. Current JAX places `pmap` in maintenance mode and encourages `shard_map` for new explicit per-device code. Sharded `jit`/NamedSharding and `shard_map` do different levels of work specification. Verify installed-version migration guidance rather than mechanically renaming APIs. Identify global shape, local shard shape, mesh axis types, replicated/partitioned values, reduction axes and collective order.

## PyTorch transforms and compilation

Establish eager behavior first, including `train()/eval()`, `no_grad()/inference_mode()`, buffers and autograd. `eval()` does not disable gradients. `inference_mode()` has stronger tensor restrictions than `no_grad()`; use only when compatible with downstream behavior.

For functional comparison, `torch.func.functional_call` can expose parameters/buffers without mutating model weights. Account for tied parameters, parametrizations and buffer mutation. `torch.func.vmap`, `grad`, `jvp` and higher-order transforms have operator/state constraints; do not assume an eager `autograd.Function` automatically supports them.

Validate `torch.compile` separately for inference and training. Diagnose in increasing depth: eager -> compile with an eager backend -> AOTAutograd-oriented backend -> Inductor, where supported by installed versions. This isolates graph capture, differentiation and code generation rather than blaming the entire compiler. Use graph-break/recompile logs and a small minifier only after a stable minimal reproduction. Do not suppress compiler errors and then report eager fallback as compiled success.

Check dynamic shapes, strides, scalar guards, train/eval changes, mutable buffers and parameter updates. Debug flags, anomaly detection and profiler instrumentation are diagnostic only; remove them for final timing. Export/AOT artifacts have explicit device, shape, dtype and version constraints; a compiled CUDA artifact is not a CPU checkpoint.

## CPU-first resource policy

Inspect available memory/storage and current workload; do not assume all installed RAM or VRAM is free. Use an isolated environment, a small CPU-thread budget and bounded subprocesses. Avoid parallel source/target accelerator allocation when memory is tight; export immutable oracle fixtures and run frameworks sequentially if needed. Use tiny synthetic inputs and model-supported small dimensions for development, then clearly label missing production-scale evidence.

Do not shrink away the semantics being tested: retain active normalization, residual projections, gate/bias distinctions, masks, tied weights, multiple training steps and checkpoint state as relevant. Use cheap shape/import checks before runtime compilation. Stop a process at its deadline and collect an actionable failure rather than launching repeated unchanged runs.

## CUDA/GPU validation

Verify driver, runtime, framework build, device capability and actual selected backend. Follow current official JAX CUDA installation support; do not mix incompatible system libraries, pip CUDA packages and copied historical flags. Confirm device placement with a tiny operation and synchronization. Set allocation/preallocation limits before initializing a runtime when necessary; record them because they affect memory measurements.

Use separate processes when JAX and Torch memory pools compete. Never free another user's allocations or kill unrelated jobs. Distinguish a compute kernel failure from OOM, async execution failure, driver/runtime mismatch or an accidental CPU fallback.

Run forward, backward, optimizer/state and checkpoint smoke on each required precision and compiler mode. Pin or record matmul/convolution precision, reduced-precision reduction, deterministic-algorithm settings and cuDNN algorithm policy. CPU FP64 diagnostics are not an accelerator parity substitute.

For custom CUDA/Triton/Pallas/FFI code, test boundary masks, non-contiguous strides, initialized outputs, synchronization and allocator/stream lifetime. Use compute-sanitizer tools appropriate to memory, initialization, race and synchronization issues when relevant; keep expensive instrumentation scoped to minimized cases. A passing Python wrapper test cannot certify memory safety. Consult current Pallas backend support instead of assuming historical Triton lowering remains supported.

## TPU and distributed validation

Provision only when required and authorized; a CPU-backed multi-device simulation is not a real TPU test. Verify TPU generation, topology, JAX/jaxlib/libtpu versions, precision/accumulation policy, layouts and mesh. Initialize distributed coordination before device access when required. All processes must enter collectives in compatible order; rank-specific Python branches can hang.

Reconstruct global mathematical behavior: sum versus mean gradients, unequal valid-token counts, cross-replica BatchNorm statistics, optimizer state partitioning, gradient accumulation and clip norm over global parameters. Do not blindly apply an extra pmean to values already globally reduced. State whether batch size is local or global.

Validate one device -> a small actual multi-device case -> required topology. Compare gathered small reference outputs and canonical state; avoid gathering huge models merely to test metadata. Check partition divisibility/padding, replicated parameters, sharded RNG streams, process-index fold-in and checkpoint restore/resharding. Bound collective waits and preserve per-rank logs. Record slowest-rank timing and memory, not just rank 0.

End every owned accelerator/remote session when evidence is collected. Remove only project-created temporary resources. Report unsupported/missing hardware as untested with exact commands and expected checks for later execution; do not present unexecuted templates as measured results.
