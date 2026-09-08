# Authoritative reading map

Reviewed against current documentation during development on 2026-09-08. Recheck installed versions: links with `latest`/`main` evolve. Project-level research keeps revision/digest evidence and critical review; this short map is for task-time lookup, not a requirement to load every page.

## JAX and module systems

- [JAX sharp bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html): purity, indexing, mutation, precision, tracing.
- [PRNG behavior](https://docs.jax.dev/en/latest/random-numbers.html) and [design](https://docs.jax.dev/en/latest/jep/263-prng.html): explicit keys, split streams, no sequential-equivalence assumption.
- [Type promotion](https://docs.jax.dev/en/latest/101/type_promotion.html) and [promotion design](https://docs.jax.dev/en/latest/jep/9407-type-promotion.html).
- [JAX benchmarking](https://docs.jax.dev/en/latest/benchmarking.html), [GPU memory allocation](https://docs.jax.dev/en/latest/gpu_memory_allocation.html), [installation](https://docs.jax.dev/en/latest/installation.html).
- [JIT](https://docs.jax.dev/en/latest/jit-compilation.html), [automatic differentiation](https://docs.jax.dev/en/latest/advanced-autodiff.html), [control flow](https://docs.jax.dev/en/latest/control-flow.html), [checkify](https://docs.jax.dev/en/latest/debugging/checkify_guide.html).
- [Sharding](https://docs.jax.dev/en/latest/sharding.html), [shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html), [multi-process execution](https://docs.jax.dev/en/latest/multi_process.html), [changelog](https://docs.jax.dev/en/latest/changelog.html).
- [Flax Linen normalization implementation, v0.12.9](https://flax-linen.readthedocs.io/en/v0.12.9/_modules/flax/linen/normalization.html): variance, momentum, state and initialization.
- [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html), [Equinox transformations](https://docs.kidger.site/equinox/api/transformations/), [Haiku transformations/state](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html).
- [Optax Adam](https://optax.readthedocs.io/en/latest/api/generated/optax.adam.html) and [optimizer index](https://optax.readthedocs.io/en/latest/api/optimizers.html).

## PyTorch

- [Numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html), [reproducibility](https://docs.pytorch.org/docs/main/notes/randomness.html), [autograd mechanics](https://docs.pytorch.org/docs/main/notes/autograd.html).
- [BatchNorm2d, v2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html), [GRU equations](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html), [scaled dot product attention](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html).
- [SGD](https://docs.pytorch.org/docs/main/generated/torch.optim.SGD.html), [AdamW](https://docs.pytorch.org/docs/main/generated/torch.optim.AdamW.html), [gradient clipping](https://docs.pytorch.org/docs/main/generated/torch.nn.utils.clip_grad_norm_.html).
- [torch.func](https://docs.pytorch.org/docs/main/func.html), [compiler troubleshooting](https://docs.pytorch.org/docs/main/torch.compiler_troubleshooting.html), [benchmark utilities](https://docs.pytorch.org/docs/main/benchmark_utils.html).
- [Serialization](https://docs.pytorch.org/docs/main/notes/serialization.html), [distributed overview](https://docs.pytorch.org/tutorials/beginner/dist_overview.html).

## Interoperability and benchmark evidence

- [torchax](https://github.com/google/torchax): a PyTorch frontend/JAX interoperability option, not automatically a native rewrite.
- [KernelBench](https://github.com/ScalingIntelligence/KernelBench), [Caesar](https://github.com/ScalingIntelligence/caesar), [released refinement samples](https://huggingface.co/datasets/ScalingIntelligence/kernelbench-samples).
- [Kevin, multi-turn kernel generation](https://arxiv.org/html/2507.11948v1): feedback, trajectory scaling and evaluator exploit examples; protocol-specific findings.
- [KernelBench-Verified](https://arxiv.org/html/2607.16241v1): baseline sensitivity, adversarial values and failure cases. Cross-check precision/API claims with vendor documentation; papers can contain technical errors.
- [KernelSkill](https://arxiv.org/html/2603.10085v1): explicit optimization knowledge and trajectory memory. Reported results are authors' measurements, not our reproduction.
- [NablaFuzz](https://github.com/ise-uiuc/NablaFuzz): differential gradient testing; apply numerical-instability guards rather than assuming every discrepancy is a bug.

Do not use target-model implementations from these or other repositories as references during a clean-room validation port. General framework behavior is allowed; target port code and explanations are not.

## Focused semantic counterchecks

- [PyTorch GELU](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.gelu.html) and [JAX GELU](https://docs.jax.dev/en/latest/_autosummary/jax.nn.gelu.html): explicitly align exact/tanh defaults.
- [JAX attention](https://docs.jax.dev/en/latest/_autosummary/jax.nn.dot_product_attention.html) and [Torch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html): axes, scale, masks and backend restrictions.
- [JAX cond](https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html) and [control flow](https://docs.jax.dev/en/latest/201/control-flow.html): tracing versus execution and batched-predicate lowering.
- [Equinox stateful example](https://docs.kidger.site/equinox/examples/stateful/) and [state API](https://docs.kidger.site/equinox/api/nn/stateful/): functional state threading is not absence of state.
