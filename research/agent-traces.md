# Published agent trajectories: detailed comparative review

Evidence class: **EXTERNAL**, reviewed 2026-09-08. No generated kernel was executed locally. Code changes and evaluation outcomes are observations of released artifacts; causal interpretations below are explicitly inferences, not access to private reasoning.

## Corpus and limits

[KernelBench](https://github.com/ScalingIntelligence/KernelBench), [Caesar](https://github.com/ScalingIntelligence/caesar) and [released samples](https://huggingface.co/datasets/ScalingIntelligence/kernelbench-samples) expose multi-turn refinement artifacts. We inspected twelve complete ten-turn logs: three models on four matched tasks, using the same `eval_result_last_only` feedback condition and sample index zero. The pinned dataset revision is `b32b70bf2d84c7395a94a214e83183152233f298`. Exact URLs, SHA256 hashes, every turn's code hash, outcomes and published timing statistics are in [trace-cohort.json](trace-cohort.json). `tools/trace_snapshot.py` recreates this bounded cohort; raw third-party logs remain outside the package and Git history.

Selection covers compute-bound matrix multiplication, a cheap elementwise operation, a numerically sensitive large reduction, and a parameterized composite. It is a purposive matched case study, not a random sample, model leaderboard or estimate of universal success rates. DeepSeek R1 produces the fastest valid recorded candidate on each selected task; Llama is weaker on these tasks, while V3 can be more consistent than R1. Those labels apply only to this released cohort, not current frontier agents. Device identifiers vary across L40S runs; fine timing differences are not controlled causal comparisons. Baseline PyTorch timings are not included in these selected logs, so no PyTorch speedup is inferred from candidate latency alone.

A recorded correctness pass with runtime `-1` is **not** valid timing. `compiled=false` sometimes accompanies runtime or device errors rather than a compiler diagnostic. Missing standalone `feedback` fields do not imply absent feedback: prompt context may contain evaluation results. Repeated code can reflect generation or orchestration behavior; we cannot distinguish internal causes from artifacts alone.

## All matched outcomes

Each row lists the best positive finite correctness-passing recorded latency and final attempt outcome. Values are published milliseconds, not measurements on our PC. Best recorded does not mean production-safe or statistically selected on a holdout.

| Task | Agent | Correct turns / 10 | Best valid recorded latency | Final outcome |
|---|---|---:|---:|---|
| L1 P1 square matmul | R1 | 3 | 1.42, turn 7 | Compile failure |
| L1 P1 square matmul | V3 | 10 | 2.44, turn 6 | Correct, 2.45 |
| L1 P1 square matmul | Llama 3.1 70B | 10 | 6.36, several turns | Correct, 6.37 |
| L1 P19 ReLU | R1 | 8 | 0.0036, turn 7 | Correct, 0.0139 |
| L1 P19 ReLU | V3 | 10 | 0.0121, turn 6 | Correct, 0.0129 |
| L1 P19 ReLU | Llama 3.1 70B | 9 | 0.019, turn 2 | Correct, 0.039 |
| L1 P40 LayerNorm | R1 | 4 | 1.5, turn 10 | Correct |
| L1 P40 LayerNorm | V3 | 8 | 1320, multiple turns | Correct; two earlier passes lack valid timing |
| L1 P40 LayerNorm | Llama 3.1 70B | 0 | None | Compile failure |
| L2 P37 Linear, Swish, bias, GroupNorm | R1 | 3 | 0.0802, turn 1 | Compile failure |
| L2 P37 composite | V3 | 0 | None | Numerical mismatch |
| L2 P37 composite | Llama 3.1 70B | 0 | None | Numerical mismatch |

## Matrix multiplication: progress, regression and fake optimization

**R1, L1 P1, turns 1–10.** A shared-memory tiled implementation first passes. A vectorized rewrite changes B's shared-memory orientation but reads it with inconsistent indices, producing large errors. Turn 3 corrects that indexing, but its recorded 6.54 ms is slower than turn 1's 2.39 ms. Double buffering then introduces further address/phase errors. Turn 7 fixes column offsets and phase handling and records 1.42 ms. Turns 8–10 replace this working implementation with tensor-core code that fails on fragment/type/API errors, including nonexistent half-vector helpers. The final artifact is invalid despite an earlier valid candidate.

**Inference and transfer:** retain separate known-correct and performance incumbents. A compilation capability probe is cheaper than a broad model rewrite around an unfamiliar API. A corrected numerical implementation is not necessarily faster. Changing precision or kernel family is a material semantic/measurement change, not a free optimization. Passing N=2048 does not establish tail safety: unguarded vector/tile reads must be tested at supported odd and boundary sizes.

**V3, same task.** Turn 2 adds shared-memory tiling and explicit zero-filled boundary loads, improving the recorded 3.44 ms to about 2.45 ms. Restrict qualifiers and host compilation flags then provide no visible improvement; turns 4–10 repeat identical code. This is a useful initial structural optimization followed by a plateau, not nine independent improvements.

**Llama, same task.** Shared memory reduces recorded latency from 13.0 to about 6.37 ms. Later variants duplicate essentially the same arithmetic under increasingly optimistic names, briefly remove duplicates, then add more. The source contains no load guards for partial tiles, although final stores are guarded. Names such as `optimized_tiled_warp` do not establish warp-level improvements or correctness outside the benchmark shape.

**Skill action:** require an actual changed hypothesis and measured bottleneck, account for asymptotic work/traffic, hash candidates and test configuration, and reject cosmetic renaming as an optimization iteration.

## Layer normalization: successful localization versus expensive repetition

**R1, L1 P40.** Turn 1 fails because manual module registration duplicates registration generated by `load_inline`. Removing that duplication exposes a numerical failure. Renaming statistics and forcing contiguity in turn 3 leaves the same errors. Turn 4 replaces lane-local shuffle results with a shared reduction/broadcast and also uses double precision; correctness now passes at 17.9 ms. Because both factors changed, the trace cannot establish that precision alone fixed the bug. A correctly broadcast FP32/vectorized version passes at 3.99 ms. Two attempts guess current-stream namespaces; adding the appropriate c10 header and namespace restores compilation. Turn 9 changes vector offsets incorrectly and fails, then turn 10 repairs contiguous offsets and passes at 1.5 ms.

**Inference and transfer:** earliest divergent statistics and their broadcast are stronger evidence than generic 'precision problem' labels. Broad changes confound causal attribution even when they succeed. Isolate reduction ownership, axis, normalization denominator and broadcast before dtype escalation. Successful recovery can simplify and then re-optimize, but the final fast candidate still needs cancellation, nonfinite and vector-tail tests outside this case study.

**V3, same task.** Initial code recomputes the entire normalized feature row independently for each output element. With about four million features per row, this changes a linear reduction into quadratic repeated work. Two attempts hit 600-second correctness timeouts. Two subsequent correctness passes have failed performance measurements (`runtime=-1`). A shared-memory rewrite still recomputes full-row statistics for many output tiles. Turns 5–10 repeat identical code, with roughly 1320 ms latencies and one extremely noisy 2440 ms result.

**Inference and transfer:** a static operation-count audit would reject this design before expensive execution. Do not benchmark just because output became correct; first inspect pathological duplicate computation. Never treat a negative timing sentinel as the fastest result. The noisy identical-code run is evidence against attributing every timing change to optimization.

**Llama, same task.** A tensor-construction overload error is followed by an incorrect use of `device` rather than `device()`. Turns 3–10 repeat the same compilation failure. Static inspection also shows a different normalization extent and missing affine behavior, which would remain even after fixing syntax. Compilation success would not have established semantic success.

**Skill action:** separate compile, runtime, semantic, numerical and evaluator failures; probe versioned APIs cheaply; stop repeated identical failures; verify full operator contract after build repair.

## Composite model: state inventory beats random-walk algebra

The source performs `Linear` including its internal bias, then Swish, an additional learned bias, and GroupNorm. Source checkpoint identity and initialization matter independently of output shape.

**R1, L2 P37.** The initial candidate retains native Linear and customizes selected remaining operators, which is allowed by this benchmark's prompt. A multi-factor optimization introduces invalid host compiler flags. After flag/runtime repairs, several dtype/contiguity changes leave identical numerical errors because reduction results still differ by lane. A shared reduction with common statistics restores correctness in turn 7. Turn 8 reinterprets FP32 storage as half storage and also changes reduction logic, producing NaNs. Turn 9 restores correct storage interpretation and an XOR reduction that distributes the result across the tested 32-lane group. Turn 10 again fails on an unfamiliar type API. The first valid candidate remains the fastest recorded within this trajectory.

**Inference and transfer:** do not reinterpret a tensor's storage to 'enable FP16'. A 32-lane reduction is not a general GroupNorm implementation for every group size. Partial replacement is legitimate here, but a requested native cross-framework port may not call its source runtime in production. Transfer the decision discipline, not benchmark-specific permissions.

**V3, same task.** A newly initialized weight layout replaces the source Linear, omits its pre-Swish bias, and uses a separate bias only after Swish. Group affine indexing also has an independent defect. Turn 2 rewrites division into reciprocal multiplication inside Swish; subsequent attempts repeat the same wrong code and errors. No change addresses missing state or the earliest differing affine computation.

**Llama, same task.** The first forward creates a new CPU random matrix and passes it into a CUDA path, with a device error. A later native Linear repair leaves Swish implemented with an extra factor of x. GroupNorm affine dimensions/group count remain inconsistent. Subsequent attempts toggle learned versus unit affine values and repeat two incorrect regimes rather than isolate the actual invariants.

**Skill action:** inventory every parameter and buffer before kernel work; prohibit fresh initialization in normal forward paths; compare affine output before later activations; keep one hypothesis/change per numerical debug step. After three speculative fixes, return to source behavior and minimal intermediate fixtures.

## ReLU: cheap work can still be overengineered

**R1, L1 P19.** Increasing vector width and per-thread work does not monotonically improve latency. A simple four-element vector implementation with explicit tail handling records the best latency at turn 7, after a device-unavailability error in turn 6. Further unrolling and grid changes regress or fail; the final passing result is slower. Removing fast-math, changing loads and adding contiguity occur together around turn 7, so their individual effect is not identifiable from this trace.

**V3, same task.** The trajectory alternates `fmaxf` and a conditional expression and repeatedly observes different small latencies. Allocation changes happen early as well. Without controlled paired repetitions, these observations do not isolate instruction choice. Exceptional-value behavior can differ even when random finite tests all pass.

**Llama, same task.** A scalar kernel grows unnecessary shared-memory paths and mislabeled warp variants. Vector access at an unscaled element index triggers a misaligned-address error. Fixing address scaling restores correctness but not the original latency. Small-input branches are not exercised by the large benchmark input; adding them is not evidence of coverage.

**Skill action:** use selected nonfinite/boundary tests, remove unmeasured speculative complexity, compare equivalent inputs/cache/allocation boundaries, and keep both performance and correctness incumbents.

## What these artifacts do and do not establish

They establish concrete examples of API guessing, missing state, wrong reduction broadcast, unsafe dtype reinterpretation, unchanged-code loops, invalid timing sentinels, asymptotic waste, noise and discarded incumbents. They also show effective structural tiling, restoration of simple correct reductions, targeted address fixes and reuse of a valid library operator.

They do **not** prove that our skill improves an agent's success rate, reveal private reasoning, provide independent holdout performance, validate every supported tensor shape, or identify today's strongest/weakest model. The skill's three-fix reset, incumbent rule and escalation hierarchy are **DESIGNED** safeguards informed by these observations. Effectiveness across harnesses requires prospective evaluation rather than a prose claim.
