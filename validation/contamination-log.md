# Validation selection and contamination log

Date: 2026-09-08. Policy: implementation references are restricted to original source projects, official framework documentation/implementation, mathematics, and independently derived code. Existing target ports are not implementation references.

## Selected original projects

| Direction | Original project and revision | Bounded scope | Selection rationale |
|---|---|---|---|
| JAX to PyTorch | google/flax, 01854da11286b4109c59d7fd9205f3822fe807d6, examples/imagenet | ResNet model family, tiny source-supported widths/shapes, inference and short training with mutable BatchNorm state | Convolution layouts, asymmetric padding, residual projections, initialization, state, gradients and SGD semantics |
| PyTorch to JAX | pytorch/examples, acc295dc7b90714f1bf47f06004fc19a7fe235c4, word_language_model | RNNModel family (LSTM, GRU, tanh/ReLU RNN), inference and bounded truncated-backpropagation training | Recurrent gates and biases, scan, hidden state, embeddings, tied weights, dropout, loss reduction, clipping and checkpoint semantics |

These are model/workflow ports, not claims to reproduce original large-dataset training campaigns. Original dataset preparation, distributed training launcher and unrelated TransformerModel are not acceptance surfaces for these bounded projects. Wider architecture guidance remains required in our general skill.

## Existence checks before implementation

- GitHub repository metadata search `"flax" "pytorch" "resnet"` returned `eltsai/resnet18_pytorch_jax_tpu`. Repository metadata only was viewed. Its implementation, README, issues and diffs were not opened. Treat it and other target ResNet implementations as blocked references.
- GitHub repository metadata searches `"word_language_model" "jax"` and `pytorch jax LSTM language model` returned no repositories. This does not prove absence. All third-party JAX ports of this source remain blocked whether discovered or not.

## Rejected candidate and accidental discovery exposure

A broad web existence search for nanoGPT in JAX returned expanded README snippets, including high-level design descriptions, before selection. No target implementation files were opened. To avoid relying on potentially revealing target explanations, nanoGPT was rejected as a validation project. Its target repositories (including cgarciae/nanoGPT-jax, surgeglobal/nanoJAXGPT and mahakal001/nanogpt-jax) are blocked from implementation research. General skill research may mention unrelated architecture techniques but may not supply implementation details for our selected ports.

## Allowed sources and handling

`tools/fetch_originals.py` has an explicit two-repository, exact-revision file allowlist. Downloaded original files and licenses receive SHA-256 hashes in `validation/originals/manifest.json`. Original files are immutable test oracles. Any test seam or adaptation must be outside original files and explicitly recorded. Source framework internals may be read to resolve behavior; no target-model implementation may be substituted as an oracle.

No implementation has been written at this selection checkpoint. Later observations must distinguish direct experimental results from hypothetical failure cases and external guidance.
