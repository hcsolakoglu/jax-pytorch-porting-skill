# Bounded native-port validation

These experiments improve the skill; they are not production certification. No existing target implementation, target-port diff or target-code issue was used. See [contamination log](contamination-log.md), [original manifest](originals/manifest.json) and [frozen budgets](budgets.json).

## Environment and replay

Actual versions: Python 3.12.14, Torch 2.14.0+cpu, JAX/JAXlib 0.11.1, Flax 0.12.9, Optax 0.2.8 and NumPy 2.5.2. The full observed package set is in [validation-environment.lock](../research/validation-environment.lock). This is a version inventory, not a cryptographic wheel lock or cross-platform reproducibility guarantee.

For a fresh isolated environment, install the CPU Torch wheel first, then CPU JAX and development dependencies. Do not install CUDA extras for these tests:

```sh
uv venv --python 3.12 .venv
uv pip install --no-cache --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
uv pip install --no-cache --python .venv/bin/python \
  jax==0.11.1 flax==0.12.9 optax==0.2.8 numpy==2.5.2 \
  pytest==9.1.1 pyyaml==6.0.3 ruff==0.16.6
```

Review the resolved environment against the saved inventory. Exact transitive package availability can differ later. `environment.py` configures CPU execution and bounded thread/affinity use before importing frameworks. The benchmark further restricts both implementations to one allowed CPU core. These settings apply to owned experiment processes, not global machine configuration.

Run cheap checks first, then each model command separately with a deadline:

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check tools tests validation skills/jax-pytorch-porting/scripts
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.source_probe
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.validate_resnet
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.validate_resnet --training
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.validate_rnn
timeout --signal=TERM --kill-after=5s 75s .venv/bin/python -m validation.semantic_probes
```

`timeout` examples target GNU/Linux; use a host-native equivalent elsewhere. No dataset downloads are needed. Do not launch all model processes in parallel on a memory-constrained machine.

Benchmarks refuse stale correctness reports:

```sh
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.benchmark_cpu --model resnet
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.benchmark_cpu --model rnn
timeout --signal=TERM --kill-after=5s 90s .venv/bin/python -m validation.benchmark_cpu --model resnet --compile-torch
```

The last command is an optional bounded Inductor experiment, not a requirement for every replay. Results preserve failures rather than alter budgets. First-call time is not total process startup, and peak RSS covers both frameworks together.

## JAX to PyTorch: Flax ImageNet ResNet V1.5

**Source:** `google/flax` revision `01854da11286b4109c59d7fd9205f3822fe807d6`, original model and training code with Apache-2.0 license. **Native target:** [resnet_torch.py](ports/resnet_torch.py); runtime imports Torch/NumPy, not JAX.

The port implements dense basic and bottleneck residual blocks, explicit NHWC input handling, source-equivalent SAME padding/pooling, parameter layout conversion, population-variance BatchNorm state and classifier projection. Width is reduced for CPU tests while preserving genuine source code paths. ResNet18 stage depth remains intact; selected bottleneck paths receive forward checks. Convolution-local variants, multiple replicas and full ImageNet preprocessing are excluded.

Training validation uses selected original `train_step` function definitions without modifying their bodies, bound to a one-device CPU collective axis. A minimal harness avoids importing the original TensorFlow data launcher. It compares CE/L2 loss, gradients, Nesterov updates/momentum and mutable state over short steps, plus checkpoint conversion and resumed next-step behavior. This is not a test of the entire ImageNet data pipeline, multi-host training, dynamic loss scaling or every Orbax format.

Observed lessons: complementing BatchNorm momentum does not fix running-variance convention; zero-initialized residual scales hide inactive branches; source state must remain immutable across independent comparisons; same-seed reload is a weak checkpoint test. Diagnostic active-branch fixtures are explicit and never disguised as trained weights.

**Recorded result:** 46 forward comparison records and 645 training/state/restore records pass unchanged budgets. These are comparison records, not 691 independent model configurations.

## PyTorch to JAX: recurrent word language model

**Source:** `pytorch/examples` revision `acc295dc7b90714f1bf47f06004fc19a7fe235c4`, original `RNNModel` and training logic, BSD-style license. **Native target:** [rnn_jax.py](ports/rnn_jax.py); runtime imports JAX/NumPy, not Torch.

Supported exercised paths include LSTM, GRU, tanh RNN and ReLU RNN, stacked layers, sequence-major flattening, input/output dropout, hidden state and tied embeddings. The source Transformer branch and full corpus data/generation CLI are not ported. Source parameters are imported explicitly rather than relying on cross-framework random initialization.

Validation compares nonzero hidden-state inference, gradients, clipped manual-SGD updates, tied parameter identity, short truncated-BPTT chunks, deterministic injected-mask behavior and native JAX RNG checkpoint resume. GRU reset placement and separate affine biases are preserved. Injected masks establish conditional parity, not equivalence of fused native dropout random streams. The example's optional AdamW training path is not claimed covered by this model port; independent three-step AdamW probes cover update equations only.

Typed RNG serialization retains key bits and implementation/dtype metadata. Target NPZ reload and subsequent draws/updates are checked; no claim is made that every framework checkpoint format interchanges transparently.

**Recorded result:** 435 comparison records across selected configurations pass. No long-run convergence, language-model perplexity or trained-checkpoint accuracy was measured.

## Cross-architecture controls

`semantic_probes.py` checks exact-versus-approximate GELU, native attention axes and causal masks, all-masked behavior, AdamW supplied-gradient transitions, transformed control flow and JVP/VJP duality. These extend beyond both validation architectures. All-masked attention mismatch is an observed version/backend difference, not a failed port disguised as a pass: the probe records the difference and tests the claimed nonempty-mask equivalence separately.

Unit tests intentionally reject shape broadcasting, exact-integer loss, vacuous empty comparisons, nonfinite mismatches, complex NaN masking, arithmetic overflow and silent extended-precision casts. Negative trace tests reject invalid timing sentinels. Documentation and package checks target stale evidence and missing resources rather than executing heavy accelerator suites.

## Interpretation and evidence classes

`results/*.json` reports distinguish source calibration, port comparisons and timings. `reporting.py` uses a common strict comparator; a source-code/budget/dependency fingerprint invalidates stale evidence. Correctness and performance are separate statuses. Small CPU LSTM speedup does not imply universal JAX superiority; slower ResNet Torch results remain failed no-regression gates.

The skill contains production workflows for GPU/TPU/distributed execution, mixed precision, large checkpoints, task metrics and statistical trajectories. Those are **DESIGNED** procedures supported by external documentation, not **OBSERVED** local validation. Future production users must execute relevant backend and workload gates.
