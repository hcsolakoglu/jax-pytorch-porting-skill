# Semantic mapping reference

Read source implementation and installed framework behavior, not names alone. Sources: JAX sharp bits, promotion and PRNG design; PyTorch operator documentation; Flax/Equinox/Haiku state guides in [sources](sources.md). These are documented hazards and designed checks, not a claim that every combination was experimentally validated.

## Modules, trees, state and identity

| Surface | Mapping requirement and cheap probe |
|---|---|
| PyTorch modules | `named_parameters`, buffers, submodules, training flags, hooks, parametrizations and actual caller form the contract. `state_dict` is not necessarily the entire runtime state. Inspect persistent/nonpersistent buffers. |
| Flax Linen | Separate `params`, `batch_stats`, cache and other collections. `init` and `apply` differ; mutable collection outputs must be threaded into the next call. Initialization may suppress normal state updates. |
| Flax NNX | Distinguish graph structure/aliasing from variable state and RNG objects. Use documented split/merge/state/filter APIs; do not flatten an object graph as independent leaves and lose sharing. |
| Equinox | Separate differentiable arrays, nondifferentiable arrays and static metadata; use filtered transforms or explicit partition/combine. A float array is not automatically a trainable parameter. |
| Haiku | `transform` versus `transform_with_state`, initialization keys, RNG sequences and state-return conventions matter. Transform stateful functions before applying JAX transforms; avoid tracing side effects through raw module construction. |
| Tied weights | Keep one parameter identity, not two equal initial arrays. Check aliases, sum all gradient contributions once, create one optimizer slot, and reconstruct ties after load. |
| Mutations/views | `.detach()` removes gradient tracking but can share storage; `stop_gradient` is not a promise of copied storage. In-place operations, hooks and view aliasing require an observable-behavior decision. Do not replace a view with an independent parameter. |

Build mapping records with full semantic names. Require exhaustive source dispositions and target coverage. One-to-many, many-to-one, packed gates and aliases are explicit exceptions to a bijection, not reasons to weaken coverage. Compare mapped values before executing a model.

## Layout and operator semantics

| Area | Frequent mismatch | Discriminating test |
|---|---|---|
| Dense | Torch weight `[out,in]`; many JAX libraries use `[in,out]`. Equinox Linear uses its own convention. | Rectangular matrix with identifiable entries, not square identity alone. |
| Conv | Torch commonly NCHW/OIHW; Linen commonly NHWC/HWIO. `lax.conv_general_dilated` dimension numbers are explicit. Groups/depthwise and transposed convolutions need separate derivations. | Unequal channel counts, rectangular kernel/input, groups and impulse at each boundary. |
| Padding | `SAME` can be asymmetric for stride > 1; fixed symmetric padding is not interchangeable. Dilation changes effective kernel size. | Compute `out=ceil(in/stride)`, `total=max((out-1)*stride+effective_kernel-in,0)`; check placement for odd/even lengths. |
| Pooling | Ceil/floor output size, boundary fill, count/include-pad and indices/tie gradients differ. Max-padding requires a neutral value, not zero for negative inputs. | Negative-only tensor and last partial window. |
| Broadcast | Trailing-axis alignment can silently normalize an unintended axis when dimensions happen to match. | Pairwise unequal dimensions; explicit axis annotation before broadcasting. |
| Indexing | JAX out-of-bounds gather/scatter behavior need not raise as Torch does. Duplicate index updates, negative indices, sorting stability and scatter reduction order matter. | Bounds, duplicate indices, negative indices and expected exceptions; validate bounds outside JIT or use checkify when required. |
| Reshape | Logical order, non-contiguous Torch views, flattening sequence/batch order and channel-last memory format are distinct. | Transposed/sliced inputs and position-coded values. |
| Einsum | Ellipsis behavior, contraction path, preferred accumulator dtype and precision differ. Mathematical notation alone does not fix reduction order. | Explicit output indices, tiny direct contraction oracle, then production-shaped reduction. |
| Embedding | Padding indices, sparse gradients, `max_norm` mutation, frequency scaling and repeated tokens affect gradients. | Repeated/padding tokens, nonzero hidden state, gradients and post-forward weights. |
| Activation | GELU exact/approximate, ReLU boundary derivatives, complex behavior and saturation can differ. | Non-boundary values first, then boundary behavior separately. |

For transposed convolution, derive axis permutation, spatial reversal if required, stride, dilation, kernel origin and output padding from source semantics. Do not assume a normal-convolution transpose rule is sufficient.

## Normalization

Specify reduction axes, affine axes, epsilon location/value, accumulation dtype, fast-versus-stable variance, training mode and state update law.

Torch BatchNorm uses batch population variance for current normalization but an unbiased estimate for its moving variance. Its momentum weights the new statistic. Flax Linen's momentum weights the previous statistic and its implementation updates with its computed batch variance. A complement of momentum alone therefore does not ensure training-state parity. Use a source-equivalent state update or a documented compatibility adapter. Check initialization and one-element reduction behavior separately. Do not silently replace BatchNorm with GroupNorm to avoid state/vmap problems.

LayerNorm/RMSNorm/GroupNorm/InstanceNorm differ in axes, centering, scale offsets and epsilon defaults. Match exact formulas; `rsqrt(var+eps)` and `1/(sqrt(var)+eps)` are different. Near-constant inputs with a large mean distinguish cancellation-sensitive variance from centered variance. Keep source-equivalence and mathematical-stability findings separate.

## RNG, initialization and stochastic layers

JAX keys are explicit immutable values; split/fold-in according to stream, step, replica and sample semantics. Reusing a key repeats or correlates samples. Torch generators are stateful and backend-dependent. The same seed is not a cross-framework random tensor contract; neither vectorizing draws nor changing draw order necessarily preserves samples.

For deterministic comparison, inject canonical parameters, inputs and stochastic masks/noise generated once. Record dtype and shape before and after conversion. Match fan-in/fan-out, variance scaling, truncation correction, bias initialization, orthogonal conventions and source initialization order only when native initialization itself is required. Otherwise transferred source parameters are the oracle; no need to reverse-engineer both RNG algorithms.

Dropout: distinguish drop probability from keep probability, inverse-keep scaling, broadcast axes, per-step versus locked masks, recurrent inter-layer dropout, and train/eval mode. Functional dropout and attention APIs may require an explicit zero probability in evaluation rather than honoring a module flag. Preserve stochastic state over checkpoint restore. Test deterministic injected masks and native stochastic distributions as separate gates; a distribution test does not prove samplewise equality.

## Attention, masks and recurrent models

Record Q/K/V axis orders, packed/interleaved head mapping, GQA grouping, scaling, RoPE pairing/base/scaling, positional offsets, logit caps, bias, mask convention, dropout, softmax accumulator dtype and output projection. Boolean masks can mean allowed positions in one API and forbidden positions in another. Additive `-inf`, finite sentinels and fully masked rows are not generally equivalent. Check source-defined all-masked behavior rather than replacing NaNs with zeros.

Validate teacher-forced forward, incremental decode, cache write/read offsets, ragged padding and cache restore separately. Passing teacher-forced logits does not prove decoding-state correctness. Test a nonzero cache/hidden state; zero-state-only tests can hide omitted recurrence. High cosine can coexist with a wrong top token; compare discrete decisions and logit margins.

For LSTM/GRU/RNN, record sequence-major/batch-major convention, layer/direction stacking, gate order, input and recurrent bias identities, initial/final hidden state, cell state, truncation/detach boundary, and dropout placement. Torch GRU applies its reset gate after the hidden affine for the new gate; a generic GRU formula may apply it before. Do not merge two biases merely because their forward sum looks redundant: their optimizer updates and decay can differ.

## Dtype, precision and exceptional values

Declare parameter, input, output, accumulator, gradient, optimizer-state and master-weight dtypes separately. JAX defaults commonly restrict 64-bit values unless enabled; NumPy host construction may accidentally introduce FP64. Weak Python scalars and strongly typed 0-D arrays can promote differently. Lower-level `lax` primitives are not interchangeable with NumPy-style promotion. Check mixed signed/unsigned integers and complex conjugate differentiation explicitly when relevant.

BF16 has wider exponent range but fewer significand bits than FP16; neither implies better accuracy in every operation. Autocast, matmul precision, convolution precision, reduced-precision accumulation, fast math, flush-to-zero and fused kernels are independent settings. Match intentional FP32 islands and cast-back points. Enabling TF32 for matmul does not automatically configure convolution precision.

Do not interpret a dtype label as an arithmetic guarantee. Inspect configured and observed kernel behavior when a precision-sensitive gate fails. Test overflow/underflow and cancellation on valid source inputs. Finite inputs can legitimately produce infinite outputs; expected behavior is contractual, not a blanket requirement that all results remain finite.

## Interoperability decision

Evaluate torchax, DLPack bridges, torch2jax/jax2torch or export formats only against required forward, gradient, higher-order transform, state, device and compiler behavior. DLPack shares array storage under a lifetime/stream contract; it does not automatically transfer an autograd graph. A callback can introduce host transfers, serialization, unavailable batching/AD or compilation barriers. A converted ONNX/StableHLO artifact is not automatically a trainable native module. Verify current versions and operator support with one high-risk microcase before adopting a bridge.
