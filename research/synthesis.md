# Research synthesis and evidence boundaries

Reviewed 2026-09-08. This is a design synthesis, not a claim that every retrieved file was deeply analyzed or that our skill has a measured agent-success advantage.

## Breadth, depth and provenance

The ecosystem snapshot retrieved **371 unique files: 208 SKILL.md entrypoints and 163 supporting files**, from 14 repositories. Entry metadata was screened, 31 related entries were shortlisted, and an additional directly relevant PyTorch-to-Equinox gist was inspected. The final [comparison](competitors.md) evaluates 25 selected skills using 20 criteria frozen before scoring. Every included entrypoint was read in full; supporting files receive credit only where inspected. Retrieval alone receives no evidence credit. Exact revisions, URLs, hashes and retrieval results live in [source ledger](source-ledger.json), [repository revisions](revisions.json) and [gist provenance](equinox-gist.json).

Primary framework documentation, original model implementations, Hugging Face's released KernelBench logs, and academic searches were separate evidence streams. Sider Scholar was used for literature discovery. Search results and abstracts were not treated as full-paper review. Papers were inspected through primary HTML/venue pages; no large benchmark was reproduced. Twelve complete matched logs contain 120 attempts, reviewed in [agent-traces.md](agent-traces.md).

The two user-provided planning/execution skills informed falsification-first gates, dependency-aware evidence, immutable acceptance targets, bounded recovery and durable checkpoints. Their worked example supplied a planning pattern, not model code or imported numerical thresholds. No NISQA implementation was reused for either validation model.

## Academic and benchmark evidence

### KernelBench: joint correctness and performance

[KernelBench, ICML 2025](https://proceedings.mlr.press/v267/ouyang25a.html) formalizes correctness-plus-speed evaluation on 250 PyTorch workloads. Its execution/profiling feedback and [Caesar](https://github.com/ScalingIntelligence/caesar) trajectory artifacts support iterative diagnosis rather than one-shot code generation. Our detailed review adds examples of discarded valid incumbents, repeated code, missing model state, invalid timing sentinels and asymptotically wasteful reductions. These are observations of released artifacts, not current global rankings or proof that all profiling improves outcomes.

### Kevin: iteration helps, but reward design and permitted work matter

[Kevin, arXiv:2507.11948v1](https://arxiv.org/html/2507.11948v1), sections 3-6 and appendix F, reports benefits from multi-turn training and serial refinement under its evaluation setup. It also documents shortcuts such as copying reference work and exploiting evaluator behavior. Rewards for merely compiling can favor incomplete solutions. Its pure-inline-CUDA restrictions differ from an original KernelBench prompt permitting selected native operators. Our transfer is to freeze allowed computation, verify full semantic work and retain compact change/evidence summaries. We do not transplant RL reward coefficients, turn counts, model rankings or benchmark permissions into native framework-port acceptance.

### KernelSkill / KernelMem: useful memory, incomplete public trajectory evidence

[KernelSkill, arXiv:2603.10085v1](https://arxiv.org/html/2603.10085v1), includes ablations of long/short-term memory and case studies; [KernelMem implementation](https://github.com/0satan0/KernelMem) describes retained correctness, repair and profiling feedback. These support a bounded hypothesis/evidence ledger as a design mechanism. We inspected publication cases and repository descriptions, but did not establish a complete public matched best/worst trajectory corpus comparable to the twelve KernelBench logs. Aggregate ablations and selected code snippets are therefore labeled **EXTERNAL published findings**, not detailed observed agent trajectories or validation of our memory policy.

### KernelBench-Verified: stronger baselines can reverse conclusions

[KernelBench-Verified, arXiv:2607.16241v1](https://arxiv.org/html/2607.16241v1), and its [primary implementation](https://github.com/facebookresearch/kernel_bench_verified) examine realistic baselines, multiple input distributions and memory. Appendix case studies include input-dependent shortcuts, incorrect normalization and legitimate fusion. Precision-specific results differ: some BF16 fusion gains remain even when broader FP32 claims weaken. Thus neither 'generated kernels always win' nor 'custom kernels never win' follows. Keep precision, work boundaries, hidden distributions and memory in the contract. A matmul precision setting does not by itself configure every convolution backend; authoritative framework behavior overrides simplified benchmark README wording. These cases are not the same revision or input distribution as our 2025 ReLU trajectory cohort.

### Numerical reproducibility and transformations

[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) and [JAX FAQ](https://docs.jax.dev/en/latest/faq.html) independently explain floating-point and compiler effects that prevent universal bitwise parity. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) and [JAX random numbers](https://docs.jax.dev/en/latest/random-numbers.html) motivate explicit random draws rather than equal integer seeds. These are authoritative practice references, not controlled evidence for a universal tolerance. Our budgets are source-calibrated and component-specific; higher-precision diagnostics remain separate from the source-behavior oracle.

[Equinox's paper](https://arxiv.org/abs/2111.00254) and [stateful API documentation](https://docs.kidger.site/equinox/api/nn/stateful/) support filtered transformations and explicit state threading. Functional programming does not imply that BatchNorm, cache or stochastic state disappears. [JAX cond documentation](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) distinguishes tracing both branches from actual execution; our JAXPR probes corroborate scan versus batched-vmap behavior without claiming a hardware timing result.

## Framework mapping conclusions

The main reference index is [sources.md](../skills/jax-pytorch-porting/references/sources.md). Consequential mappings were checked against both framework behavior and executable probes where practical.

| Risk | Evidence and consequence |
|---|---|
| Initialization and stochastic equivalence | Source parameters, noise and masks must be copied/injected. Equal seeds are not equal draws; typed RNG checkpoint metadata matters. |
| BatchNorm | Source-only probe showed matching current output but different running variance after complementing momentum. Preserve population/unbiased update convention, not just momentum value. |
| GELU | Official defaults differ. CPU probe measured about 4.73e-4 default discrepancy and 4.44e-16 after explicit exact-mode alignment. |
| Attention | Native APIs use different axis orders; allowed-position boolean semantics match but fully masked XLA/CPU behavior differed in our exact versions. Select an explicit source-equivalent adapter rather than global NaN cleanup. |
| GRU and tied parameters | Reset placement, separate affine biases and one optimizer identity are observable training semantics. Original source and recurrent fixture gradients expose errors hidden by zero initial state. |
| Optimizers | Supplied-gradient tests isolate update equations from model differentiation. Three AdamW steps compare parameters, moments and counters; convolutional validation compares Nesterov momentum. |
| Compilers and distribution | Eager equivalence is not compiled or distributed equivalence. Record static/dynamic shape policy, collectives, per-rank dimensions, sharding and actual completion boundaries. CPU evidence does not certify accelerators. |
| Checkpoints | Exact key/shape/dtype coverage and next-step resume matter; a same-seed reinitialization can make an unloaded checkpoint appear correct. |

Detailed observed values are in `validation/results/semantic-probes.json` and source/model reports, not inferred from prose similarity.

## Conversion and interoperability landscape

These projects were inspected as option discovery and failure-mode context, not used as implementation references for our selected validation targets.

| Approach | Useful role | Boundary that must remain explicit |
|---|---|---|
| [torchax](https://github.com/google/torchax) | PyTorch programming surface backed by JAX; feasibility and transformation experiments | A supported interoperability layer is not automatically a standalone source-framework-free rewrite. Check operator coverage, mutation and deployment dependencies. |
| [torch_jax_interop](https://github.com/mila-iqia/torch_jax_interop) | Tensor/gradient bridges and fixture transfer | DLPack sharing has device, layout, lifetime and mutation constraints; do not assume every conversion is zero-copy or preserves optimizer identities. |
| [torch2jax](https://github.com/samuela/torch2jax) | Automated conversion experiments | Coverage is implementation/version dependent; output examples do not establish custom gradients, mutation or full training-state equivalence. |
| [jax2torch bridge](https://github.com/lucidrains/jax2torch) | Calling JAX computations from Torch and interoperability experiments | A runtime bridge can retain JAX/XLA dependencies and distinct memory semantics; native PyTorch completion requires a separate contract. |
| [Transformers conversion utilities](https://github.com/huggingface/transformers) | Historical parameter-name/layout patterns and supported migration tooling | Pin a framework-compatible revision; model-family-specific conversion is not a universal guarantee. Do not inspect forbidden target implementations during a clean-room exercise. |

Choose the smallest supported approach that satisfies the deployment contract. A bridge can quickly falsify feasibility assumptions, but hiding it in a purported native port is not success. Native target operators often offer lower maintenance cost than a custom kernel, even where code generation is possible.

## Practices accepted, rejected and qualified

Accepted: exhaustive parameter/state inventory; exact fixture injection; calibrated multi-surface numerical gates; minimal first-divergence localization; independent optimizer tests; cheap negative controls; fair paired measurements; incumbent retention; bounded reset after three speculative fixes; partial reference loading; privacy-preserving evidence and meaningful commits.

Rejected as universal rules: one tolerance for all models; cosine-only acceptance; preserving a buggy source without disclosure; silently 'improving' losses/normalization; CPU comparisons forbidden; all loops must be vectorized; all dictionaries forbidden; automatic precision reduction; guaranteed speedups; unlimited retries; copying every competitor reference into the core context.

Qualified: finite-difference/JVP checks avoid nonsmooth points; metamorphic invariance requires architecture-specific assumptions; injected dropout masks verify conditional behavior but not native RNG distribution; short trajectories do not prove convergence; warm microbenchmark gains can lose to compilation overhead; tuned source baselines and production distributions remain necessary for release claims.

## Development evidence versus remaining uncertainty

**OBSERVED:** two native ports, 46 ResNet forward, 645 ResNet training and 435 recurrent comparison records; strict comparator negatives; cross-architecture operator probes; code-bound reports; actual isolated installer behavior; three bounded CPU benchmark reports.

**EXTERNAL:** published traces, source implementations, papers, framework/harness documentation and related-skill content. Published agent timings were not rerun here.

**DESIGNED:** production-scale accelerator/distributed gates, adaptive tolerance policy, recovery process, risk-selected adversarial coverage, future task-quality/statistical checks and the scoring rubric.

**UNTESTED:** GPU/TPU/CUDA/distributed execution, full dataset accuracy, long training, all architecture families, all precision modes, all checkpoint formats, matched multi-agent harness trials and independent scoring. The small ResNet Torch port remains slower than JAX in both measured eager and Inductor CPU cases; the LSTM JAX target is faster in its measured case. This is useful counterevidence against blanket speed promises, not a reason to alter the workload or conceal a failed gate.
