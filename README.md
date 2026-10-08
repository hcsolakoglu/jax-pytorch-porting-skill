# JAX ↔ PyTorch Model Porting

[![Checks](https://github.com/hcsolakoglu/jax-pytorch-porting-skill/actions/workflows/checks.yml/badge.svg)](https://github.com/hcsolakoglu/jax-pytorch-porting-skill/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Version 1.0.0](https://img.shields.io/badge/version-1.0.0-informational.svg)

An agent skill for **native model migration between PyTorch and JAX**, in both directions. It guides a coding agent through inference, training, checkpoint conversion, numerical parity, gradient and optimizer validation, compiler behavior and fair performance measurement.

Principles: correctness first, cheap failures first, no invented speedups. A slower correct port does not pass a required no-regression performance gate.

Works with Codex, Claude Code, Cursor and other agents that load `SKILL.md` skills.

[Core skill](skills/jax-pytorch-porting/SKILL.md) · [Installation](INSTALL.md) · [Research and comparison](research/synthesis.md) · [Validation evidence](validation/README.md) · [Improvement log](research/iterations.md)

## Install

```sh
npx --yes skills@1.5.25 add hcsolakoglu/jax-pytorch-porting-skill \
  --skill jax-pytorch-porting --global --agent codex claude-code cursor --yes
```

A local-checkout fallback, ChatGPT ZIP upload, verified discovery paths and Antigravity's product-specific path differences are documented in [INSTALL.md](INSTALL.md). Installation does not require installing JAX, PyTorch or a GPU stack. The optional parity helper requires NumPy only.

For eligible ChatGPT workspaces, upload `dist/skill.zip` through Plugins → Skills → Create → Upload from your computer. Other products do not automatically inherit local skill installations.

## Use

In Codex:

```text
$jax-pytorch-porting Port this JAX model to native PyTorch.
Preserve inference and training, checkpoint state and supported input behavior.
Inspect the source revision and environment, define numerical and performance
contracts, run cheap CPU gates first, and report every untested backend explicitly.
```

Reverse direction:

```text
$jax-pytorch-porting Port this PyTorch model to native JAX.
Preserve tied weights, optimizer state, stochastic layers and checkpoint resume.
Use Flax, Equinox, Haiku or plain JAX according to the source and deployment needs.
Do not call PyTorch from the target runtime or change precision to claim speedup.
```

For existing ports, ask for first-divergence localization, training-step parity, checkpoint audit, benchmark review or evidence-backed optimization. Other harnesses use their own skill selection syntax; see [installation](INSTALL.md).

## What it guides

| Area | Coverage |
|---|---|
| Semantic mapping | Functional/stateful boundaries; parameters, buffers and aliases; layouts, padding, indexing, dtype promotion, normalization, dropout, embeddings, attention and masks |
| Training | Forward/loss, input and parameter gradients, optimizer equations/moments, clipping, accumulation, weight decay, schedules, mutable state, RNG and resumed next steps |
| Numerical validation | Source-calibrated component budgets, exact structural gates, error distributions, first-divergence localization, negative controls and statistical limits |
| Performance | Equivalent workloads, warmup/synchronization, paired repetitions, uncertainty, startup/compile/steady-state separation, memory and measured optimization |
| Compilers and hardware | JAX jit/scan/vmap/pmap/shard_map/sharding/XLA and PyTorch compile/AOTAutograd/Inductor; risk-based CUDA/GPU/TPU/distributed procedures |
| Agent reliability | Fast-fail hierarchy, bounded hypotheses, anti-contamination, incumbent retention, tool-state recovery, ownership-aware cleanup and durable evidence |

The core `SKILL.md` is about 1,500 words, with eight references loaded only when needed. It does not force a model, provider, subagent setup or JAX module library.

## Validation

Two genuine ports were derived from pinned original implementations without inspecting existing target-framework ports:

| Direction | Original project and native target | Executed evidence |
|---|---|---|
| JAX → PyTorch | Flax ImageNet ResNet V1.5, basic/bottleneck paths and mutable BatchNorm | 46 forward and 645 training/state/restore comparison records |
| PyTorch → JAX | PyTorch word_language_model, LSTM/GRU/tanh/ReLU RNN, stacked and tied weights | 435 inference/gradient/training/RNG-resume comparison records |

These are **bounded CPU synthetic-input tests**, not full-dataset accuracy, long training or production accelerator certification. See [scope and commands](validation/README.md), [original-source hashes](validation/originals/manifest.json), [contamination log](validation/contamination-log.md) and [adversarial review](research/adversarial-review.md).

Additional probes expose differing GELU defaults and all-masked attention behavior, verify three AdamW steps, inspect transformed control flow, and test JVP/VJP duality. Comparator negative tests guard overflow, complex NaNs, exact integers and unsupported precision. Saved correctness evidence is fingerprinted to code, budgets and dependencies; stale PASS reports cannot authorize a new benchmark.

### Measured CPU performance, including failures

Matched single-core FP32 developer workloads; five warmups, 30 alternating pairs and pre/post parity. Speedup is source latency divided by target latency.

| Comparison | Paired median speedup | Descriptive 95% bootstrap interval | ≥1× gate |
|---|---:|---:|---|
| Torch native LSTM → JAX jit/scan | 2.660× | 2.346–3.044× | Passed in this run |
| JAX jit ResNet → Torch eager | 0.252× | 0.239–0.262× | **Failed** |
| JAX jit ResNet → Torch Inductor | 0.538× | 0.515–0.562× | **Failed** |

Raw samples, first-call costs, input hashes and limits are retained in [benchmark reports](validation/results/). Within-session bootstrap intervals are not independent deployment confidence; host load and autocorrelation remain possible. Combined-process peak RSS is not per-model memory. These are not exhaustive tuned baselines. No GPU/TPU performance was measured. The slower ResNet result remains visible rather than changing batch size, precision or the acceptance threshold.

## Research and comparison

The ecosystem snapshot retrieved 371 files, including 208 skill entrypoints screened by metadata. Twenty-five related skills were assessed across **20 criteria, 500 individual scores**, with pinned source links, pros and cons. [Full comparison](research/competitors.md) distinguishes documented task fitness from controlled agent success. The scores are the author's own assessment under that rubric, not an independent ranking.

[Trace analysis](research/agent-traces.md) reviews twelve complete published KernelBench trajectories, totaling 120 attempts, including successful recovery and repeated failure. It separates released-code observations, published aggregate findings, our design inferences and missing trace evidence. No large benchmark was rerun.

## Repository map

```text
skills/jax-pytorch-porting/   Installable skill, UI metadata, references, NumPy helper
validation/                   Two native ports, pinned originals, bounded tests and results
research/                     Rubric, comparison, sources, trace analysis, iterations and review
tests/                        Cheap utility and release-integrity regressions
tools/                        Bounded research, comparison, checks and packaging utilities
INSTALL.md                    Verified harness mechanisms and installation caveats
```

Research snapshots, framework environments and compiler caches are ignored. The skill ZIP contains only runtime guidance, metadata, a small helper and its license, not originals, logs, model weights or dependencies.

## Checks and packaging

```sh
.venv/bin/python -m pytest -q
.venv/bin/ruff check tools tests validation skills/jax-pytorch-porting/scripts
.venv/bin/python tools/check_repo.py
.venv/bin/python tools/package_skill.py
```

Cheap checks do not import ML frameworks or allocate accelerators. Development environment setup and optional bounded model validation are documented in [validation/README.md](validation/README.md). See [contribution guidance](CONTRIBUTING.md) for evidence and scope rules.

## Limitations and release meaning

A skill is guidance plus small utilities, not an automatic universal converter. CPU tests cover selected configurations; custom kernels, quantization, sparse operations, large distributed jobs and other precision modes require their own gates. No prospective multi-harness agent trial, independent competitor scoring or full training convergence study was performed. Neither an overall documentation score nor a passing small-model test authorizes an untested production claim.

Original project material follows [LICENSE](LICENSE); vendored originals and derived validation ports retain their separate licenses in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Do not commit private source checkpoints or credentials.

## Related

[ArchCopilot](https://github.com/hcsolakoglu/archcopilot), a companion agent-driven CLI for IFC-based architecture workflows, follows the same evidence-first approach.
