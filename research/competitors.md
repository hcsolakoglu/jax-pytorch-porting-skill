# Related skills: 25-by-20 evidence-based comparison

Review date: 2026-09-08. [Frozen rubric](rubric.md); [manual scores](comparison-input.json).

Manual documented-fitness assessment, not controlled agent performance. Twenty criteria in frozen C01-C20 order. Source entrypoints were read in full; supporting material earns credit only where explicitly inspected. Uninspected linked material and parent-repository capabilities earn no inferred credit.

208 retrieved skill entrypoints were screened by metadata; 31 related entries and a directly relevant Equinox gist were shortlisted. These 25 cover direct porting plus complementary numerical, compiler, training, accelerator and recovery skills. Near-duplicate generic JAX/testing entries and peripheral export-only/paper-writing workflows were excluded. This is a purposive inspected set, not an exhaustive global top-25 claim.

Ordinal judgments can reasonably differ by one point per criterion. No independent blinded assessor or matched agent trials were run; do not interpret small total differences as empirical superiority.

**Scores measure documented fitness for this task, not general quality or measured agent success.**

## Ranking

| Rank | Related skill | Score / 100 |
|---:|---|---:|
| 1 | pytorch/pytorch: `pt2-bug-basher` | 63 |
| 2 | vllm-project/vllm: `kernel-microbenchmark` | 59 |
| 3 | Orchestra-Research/AI-Research-SKILLs: `pytorch-fsdp2` | 55 |
| 4 | nboyd/pytorch-to-equinox: `pytorch-to-equinox` | 53 |
| 5 | apple/coreai-models: `model-authoring` | 53 |
| 6 | Orchestra-Research/AI-Research-SKILLs: `ml-training-recipes` | 53 |
| 7 | Orchestra-Research/AI-Research-SKILLs: `openpi` | 52 |
| 8 | modular/skills: `import-model` | 52 |
| 9 | mflux-community/mflux: `mflux-model-porting` | 50 |
| 10 | K-Dense-AI/claude-scientific-skills: `pytorch-lightning` | 50 |
| 11 | modular/skills: `debug-model` | 49 |
| 12 | pytorch/pytorch: `aoti-debug` | 49 |
| 13 | dgrauet/claude-skill-mlx-porting: `mlx-porting` | 48 |
| 14 | modular/skills: `profile-model` | 48 |
| 15 | tensormux/kernel-skills: `debug-cuda-kernel-correctness` | 47 |
| 16 | tensormux/kernel-skills: `write-kernel-test-plan` | 46 |
| 17 | Orchestra-Research/AI-Research-SKILLs: `flash-attention` | 46 |
| 18 | mflux-community/mflux: `mflux-model-tiny-test` | 44 |
| 19 | mflux-community/mflux: `mflux-debugging` | 43 |
| 20 | tensormux/kernel-skills: `port-cuda-kernel-to-triton` | 42 |
| 21 | modular/skills: `benchmark-model` | 41 |
| 22 | tensormux/kernel-skills: `write-backend-agnostic-kernel-plan` | 40 |
| 23 | obra/superpowers: `systematic-debugging` | 39 |
| 24 | tensormux/kernel-skills: `write-numerically-stable-kernel` | 35 |
| 25 | mindrally/skills: `jax-best-practices` | 17 |

**Our self-assessed score: 79 / 100, position 1 when added to this inspected set.**
This is not a blinded assessment, global skill ranking or controlled proof that an agent will perform best.
Self-assessment under the same rubric, not an independent benchmark. CPU synthetic ports do not establish production accuracy, full architecture coverage or accelerator portability. ResNet no-regression performance gate remains failed in measured CPU cases. Score five for bidirectionality is supported by two executed native ports, not a claim of universal conversion completeness.

## Complete criterion matrix

### C01 through C10

| Entry | C01 | C02 | C03 | C04 | C05 | C06 | C07 | C08 | C09 | C10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 4 | 0 | 3 | 3 | 4 | 3 | 1 | 3 | 4 |
| 2 | 0 | 4 | 0 | 4 | 1 | 3 | 1 | 0 | 5 | 4 |
| 3 | 0 | 4 | 0 | 3 | 4 | 2 | 2 | 3 | 2 | 4 |
| 4 | 3 | 3 | 2 | 4 | 2 | 3 | 2 | 1 | 2 | 2 |
| 5 | 0 | 4 | 0 | 4 | 0 | 3 | 1 | 0 | 4 | 4 |
| 6 | 0 | 4 | 0 | 2 | 4 | 2 | 2 | 3 | 3 | 4 |
| 7 | 4 | 4 | 2 | 4 | 4 | 1 | 1 | 2 | 1 | 4 |
| 8 | 0 | 4 | 0 | 4 | 0 | 3 | 0 | 0 | 1 | 3 |
| 9 | 0 | 4 | 0 | 4 | 2 | 2 | 0 | 1 | 2 | 2 |
| 10 | 0 | 4 | 0 | 3 | 4 | 1 | 1 | 2 | 2 | 4 |
| 11 | 0 | 3 | 0 | 4 | 0 | 3 | 0 | 0 | 1 | 3 |
| 12 | 0 | 4 | 0 | 3 | 0 | 2 | 0 | 0 | 2 | 4 |
| 13 | 0 | 4 | 0 | 4 | 0 | 3 | 0 | 0 | 2 | 3 |
| 14 | 0 | 3 | 0 | 3 | 0 | 1 | 0 | 0 | 4 | 4 |
| 15 | 0 | 3 | 0 | 2 | 0 | 3 | 0 | 0 | 1 | 4 |
| 16 | 0 | 3 | 0 | 3 | 1 | 3 | 1 | 0 | 3 | 3 |
| 17 | 0 | 4 | 0 | 4 | 3 | 2 | 1 | 1 | 2 | 4 |
| 18 | 0 | 2 | 0 | 3 | 0 | 3 | 0 | 0 | 0 | 1 |
| 19 | 0 | 3 | 0 | 4 | 0 | 3 | 0 | 0 | 1 | 2 |
| 20 | 0 | 3 | 0 | 3 | 0 | 3 | 0 | 0 | 3 | 3 |
| 21 | 0 | 2 | 0 | 3 | 0 | 1 | 0 | 0 | 4 | 3 |
| 22 | 1 | 2 | 0 | 2 | 1 | 2 | 0 | 0 | 3 | 4 |
| 23 | 0 | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |
| 24 | 0 | 2 | 0 | 2 | 1 | 2 | 1 | 0 | 2 | 3 |
| 25 | 2 | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 |
| Ours | 4 | 4 | 5 | 4 | 4 | 4 | 4 | 4 | 4 | 3 |

C01 JAX; C02 PyTorch; C03 Bidirectional; C04 Inference; C05 Training; C06 Numerics; C07 Gradients; C08 Optimizer; C09 Performance; C10 Accelerators

### C11 through C20

| Entry | C11 | C12 | C13 | C14 | C15 | C16 | C17 | C18 | C19 | C20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 4 | 4 | 4 | 4 | 2 | 4 | 4 | 4 | 4 |
| 2 | 3 | 3 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | 4 |
| 3 | 3 | 3 | 3 | 2 | 3 | 2 | 4 | 4 | 3 | 4 |
| 4 | 4 | 3 | 3 | 2 | 3 | 2 | 2 | 3 | 4 | 3 |
| 5 | 4 | 2 | 3 | 3 | 3 | 2 | 4 | 4 | 4 | 4 |
| 6 | 3 | 3 | 3 | 2 | 3 | 2 | 4 | 4 | 3 | 2 |
| 7 | 2 | 2 | 2 | 1 | 2 | 2 | 4 | 4 | 3 | 3 |
| 8 | 4 | 4 | 4 | 3 | 4 | 2 | 4 | 4 | 4 | 4 |
| 9 | 4 | 3 | 3 | 3 | 4 | 2 | 3 | 4 | 4 | 3 |
| 10 | 2 | 2 | 3 | 2 | 3 | 2 | 4 | 4 | 4 | 3 |
| 11 | 4 | 4 | 3 | 3 | 3 | 2 | 4 | 4 | 4 | 4 |
| 12 | 4 | 3 | 4 | 2 | 3 | 2 | 4 | 4 | 4 | 4 |
| 13 | 4 | 3 | 3 | 3 | 3 | 2 | 3 | 4 | 4 | 3 |
| 14 | 4 | 3 | 4 | 2 | 2 | 2 | 4 | 4 | 4 | 4 |
| 15 | 4 | 3 | 4 | 4 | 4 | 1 | 3 | 4 | 4 | 3 |
| 16 | 3 | 2 | 3 | 4 | 4 | 1 | 3 | 4 | 3 | 2 |
| 17 | 2 | 2 | 3 | 2 | 2 | 2 | 4 | 3 | 3 | 2 |
| 18 | 3 | 3 | 4 | 3 | 4 | 2 | 4 | 4 | 4 | 4 |
| 19 | 4 | 3 | 3 | 2 | 3 | 2 | 3 | 3 | 4 | 3 |
| 20 | 3 | 2 | 3 | 3 | 3 | 1 | 3 | 4 | 3 | 2 |
| 21 | 2 | 3 | 3 | 1 | 2 | 2 | 4 | 4 | 4 | 3 |
| 22 | 2 | 2 | 3 | 3 | 3 | 1 | 3 | 3 | 3 | 2 |
| 23 | 4 | 4 | 4 | 3 | 4 | 3 | 3 | 4 | 4 | 2 |
| 24 | 3 | 2 | 2 | 4 | 2 | 1 | 3 | 2 | 2 | 1 |
| 25 | 0 | 0 | 0 | 0 | 0 | 2 | 4 | 2 | 2 | 0 |
| Ours | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | 4 |

C11 Localization; C12 Recovery; C13 Fast fail; C14 Adversarial; C15 Regression; C16 Harness; C17 Context; C18 Documentation; C19 Usability; C20 Evidence

## Per-skill evidence, strengths and weaknesses

### 1. pytorch/pytorch / .claude/skills/pt2-bug-basher/SKILL.md

Score: **63 / 100**. [Pinned source](https://github.com/pytorch/pytorch/blob/2b1d72fc614c57d5ebe70ebaf22da2abaf11a3d4/.claude/skills/pt2-bug-basher/SKILL.md).
Revision `2b1d72fc614c57d5ebe70ebaf22da2abaf11a3d4`; entry SHA256 `0280f5dc183f96f057513c9dbda8f9b7aead12fc42190825db47a66912ce879e`.

**Strengths:** Excellent compiler-layer separation, failing tests, minimal repro, eager/compiled accuracy diagnosis, generated code and neighboring regression checks.

**Limitations:** PyTorch-internal environment and harness assumptions; no JAX/checkpoint conversion. Private compiler APIs and flags are version-sensitive; mandatory conda/worktree mechanics are not universally available.

**Inspected evidence:** Full workflow, compile-mode distinction, error-triage table and minifier examples.

### 2. vllm-project/vllm / .agents/skills/kernel-microbenchmark/SKILL.md

Score: **59 / 100**. [Pinned source](https://github.com/vllm-project/vllm/blob/8ebc5b0a182b3351a1c62fe73e97be2c489ed2a6/.agents/skills/kernel-microbenchmark/SKILL.md).
Revision `8ebc5b0a182b3351a1c62fe73e97be2c489ed2a6`; entry SHA256 `e8e985525c368124d2cff368202b3263416dd9429dc4aabf4f6eaa021c64a5b1`.

**Strengths:** Strong synchronized timing, cache/work boundaries, roofline sanity, per-rank maxima, distributed lifetime/barrier discipline and reusable benchmark examples.

**Limitations:** Kernel/CUDA focus rather than native model conversion and training parity. CUPTI/cache defaults require workload justification; rough hardware references are not acceptance targets.

**Inspected evidence:** Full workflow and multi-GPU sections; directly linked single-device and multi-GPU benchmark examples support performance score five. Examples were inspected, not executed on GPUs here.

### 3. Orchestra-Research/AI-Research-SKILLs / 08-distributed-training/pytorch-fsdp2/SKILL.md

Score: **55 / 100**. [Pinned source](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/08-distributed-training/pytorch-fsdp2/SKILL.md).
Revision `773a52944ba4747a18bd4ae9ade53fff041adcbc`; entry SHA256 `b540043bd17d741840843df2f76d556d41be29c82a2acd85565ce2b1ce088a50`.

**Strengths:** Concrete DTensor/FSDP2 state, optimizer-after-sharding, module-hook, mixed-precision and checkpoint guidance with versioned primary references.

**Limitations:** PyTorch distributed specialist, not JAX conversion or cross-framework gradient/optimizer equivalence. Requires actual topology validation beyond local instructions.

**Inspected evidence:** Full sharding order, optimizer, checkpoint, pitfalls and official-reference sections.

### 4. nboyd/pytorch-to-equinox / SKILL.md

Score: **53 / 100**. [Pinned source](https://gist.github.com/nboyd/a1cb9592dab91bde66771bab2d5cc120/f7bd40a760076fbb6e129a2e650245678bca5255).
Revision `f7bd40a760076fbb6e129a2e650245678bca5255`; entry SHA256 `3bbee57ff5f34bf5583d1e16b4640a53098985174d41b869c5897098fe8df53c`.

**Strengths:** Direct PyTorch-to-JAX/Equinox workflow with captured real module inputs, bottom-up conversion, checkpoint serialization and scan guidance. Stronger direct task fit than generic framework tips.

**Limitations:** One direction only; optimizer transition validation is thin. Several blanket rules are, in our review, overbroad or contested: GELU defaults differ, scalar cond inside scan does not inherently execute both branches, Equinox supports explicit state, and dictionaries are valid pytrees. Fixed tolerances and oversized entrypoint reduce reliability.

**Inspected evidence:** Full gist entrypoint; bottom-up parity, serialization, scan, operation mappings and common-issue sections. Technical counterchecks are in semantic-probes.json and synthesis.md.

### 5. apple/coreai-models / skills/skills/model-authoring/SKILL.md

Score: **53 / 100**. [Pinned source](https://github.com/apple/coreai-models/blob/df8119879f125ad1e2e4c6249c2cddded75c190a/skills/skills/model-authoring/SKILL.md).
Revision `df8119879f125ad1e2e4c6249c2cddded75c190a`; entry SHA256 `0786578f0fd33562c88cbba19d84f8650a11b8a14b5c953186d926252cc57879`.

**Strengths:** Concrete on-device operator/layout constraints, bottom-up captured-input testing, cache handling and compiled-model verification.

**Limitations:** Apple deployment specialist without JAX training. Fixed PSNR thresholds and strong source-reading/execution prescriptions do not generalize across architectures and numerical budgets.

**Inspected evidence:** Full empirical rules, CPU correctness, cache, configuration and full-model verification guidance.

### 6. Orchestra-Research/AI-Research-SKILLs / 10-optimization/ml-training-recipes/SKILL.md

Score: **53 / 100**. [Pinned source](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/10-optimization/ml-training-recipes/SKILL.md).
Revision `773a52944ba4747a18bd4ae9ade53fff041adcbc`; entry SHA256 `1cb661b0d3bac57ffed65075707a9079ac23123a5a597700e85c05e03e166a57`.

**Strengths:** Broad training loops, optimizer/clipping/accumulation, mixed precision, profiles and experiment keep/discard discipline.

**Limitations:** Recipe improvements can silently alter a port: default decay masks, loss cutoffs, normalization or batch changes are not semantic equivalence. No JAX mapping and limited calibrated cross-framework checks.

**Inspected evidence:** Full training, optimization, debugging and experimentation sections.

### 7. Orchestra-Research/AI-Research-SKILLs / 18-multimodal/openpi/SKILL.md

Score: **52 / 100**. [Pinned source](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/18-multimodal/openpi/SKILL.md).
Revision `773a52944ba4747a18bd4ae9ade53fff041adcbc`; entry SHA256 `60af709a253bd745330e43067b22667b932a393e65a3cde6660a25d0ffbe4742`.

**Strengths:** Actual JAX and PyTorch model/training/serving workflows, checkpoint conversion and domain-specific normalization/resource details.

**Limitations:** OpenPI-specific rather than general bidirectional parity. Conversion commands do not prove gradients or optimizer equivalence. Large resource defaults, overwrite flags and environment-specific paths require care.

**Inspected evidence:** Full JAX/PyTorch conversion, training, inference and troubleshooting sections; general capability credit is limited to described workflows.

### 8. modular/skills / import-model/SKILL.md

Score: **52 / 100**. [Pinned source](https://github.com/modular/skills/blob/d55e84c7e2a6d954b5a38eee751a75237dee1b7d/import-model/SKILL.md).
Revision `d55e84c7e2a6d954b5a38eee751a75237dee1b7d`; entry SHA256 `96a8f6ee3fff61f2bfbffe56bdaf1d9be3f4274bbc747f67f364bae6760aafab`.

**Strengths:** Strong configuration and state inventory, structural delta tracking, scaffold-versus-implementation guards and cheap local preflight before costly serving. Clear artifact and docstring honesty checks.

**Limitations:** MAX inference rather than bidirectional framework training. Trust-remote-code example and fixed environment assumptions need review; whole-graph serving flow is not a substitute for gradient or optimizer parity.

**Inspected evidence:** Full decide/implement/verify phases; selected state audit, preflight and validation-tier references.

### 9. mflux-community/mflux / .cursor/skills/mflux-model-porting/SKILL.md

Score: **50 / 100**. [Pinned source](https://github.com/mflux-community/mflux/blob/051ba9ff25c9a8a8703356053a012a0dfad3fe39/.cursor/skills/mflux-model-porting/SKILL.md).
Revision `051ba9ff25c9a8a8703356053a012a0dfad3fe39`; entry SHA256 `9c4076e56d3043cb53d44b9981f4d650523a8fef71d9da391953e1c0b65199d8`.

**Strengths:** Correctness-first MLX integration, injected RNG inputs, checkpoint/LoRA mappings and specific regression-prone CLI/config seams. Meaningful milestone commits and real integration lessons.

**Limitations:** Repository-specific conventions and visual judgment can replace stronger numerical evidence. No JAX or detailed training transition parity; mandatory architecture conventions do not generalize.

**Inspected evidence:** Full workflow, integration-surface tables, testing and training-adapter sections.

### 10. K-Dense-AI/claude-scientific-skills / skills/pytorch-lightning/SKILL.md

Score: **50 / 100**. [Pinned source](https://github.com/K-Dense-AI/claude-scientific-skills/blob/9cf7d9aea7d84754db4c167ab04b299d33c444bc/skills/pytorch-lightning/SKILL.md).
Revision `9cf7d9aea7d84754db4c167ab04b299d33c444bc`; entry SHA256 `eacd09ffc22ab266c385b4897136636dbaee15fe62dc8cec2264e7fa4f557810`.

**Strengths:** Practical module/data/trainer templates, fast development runs, checkpointing and multi-device training setup.

**Limitations:** No cross-framework mapping or numerical transition parity. High-level training automation can conceal defaults; hardware selection must use actual memory and topology rather than model-size heuristics alone.

**Inspected evidence:** Full module/trainer, fast-dev, checkpoint and distributed guidance.

### 11. modular/skills / debug-model/SKILL.md

Score: **49 / 100**. [Pinned source](https://github.com/modular/skills/blob/d55e84c7e2a6d954b5a38eee751a75237dee1b7d/debug-model/SKILL.md).
Revision `d55e84c7e2a6d954b5a38eee751a75237dee1b7d`; entry SHA256 `593181f5c1d0042f5f908a501c6d98af149062b845ffce13373b928878255d70`.

**Strengths:** Validates dumpers, localizes first divergence, compares per-token dimensions and separates teacher-forced, incremental and serving failures. Tests hypotheses against captured tensors before recompiling.

**Limitations:** MAX/GPU-specific inference workflow; no JAX training or optimizer checks. Broad cosine thresholds can hide amplitude and discrete-decision errors without stronger budget rules.

**Inspected evidence:** Full protocol steps 0-6 and linked comparator/stacked-failure guidance.

### 12. pytorch/pytorch / .claude/skills/aoti-debug/SKILL.md

Score: **49 / 100**. [Pinned source](https://github.com/pytorch/pytorch/blob/2b1d72fc614c57d5ebe70ebaf22da2abaf11a3d4/.claude/skills/aoti-debug/SKILL.md).
Revision `2b1d72fc614c57d5ebe70ebaf22da2abaf11a3d4`; entry SHA256 `dfa5d8f1dbf8201c82435f76f5a2cd1c1e073dc38e09e57d43758e3b689a7ebb`.

**Strengths:** Cheap shape/device checks, runtime guard distinctions, intermediate-value diagnosis and explicit deprecated-versus-current AOTI APIs.

**Limitations:** Narrow compiled-inference scope, no training transition or JAX semantics. Several diagnostic flags materially change execution and must not leak into performance measurement.

**Inspected evidence:** Full initial checks, CUDA diagnosis, API notes and compile-time/runtime flag table.

### 13. dgrauet/claude-skill-mlx-porting / mlx-porting/SKILL.md

Score: **48 / 100**. [Pinned source](https://github.com/dgrauet/claude-skill-mlx-porting/blob/44d92c094bd9ca2812e222a222254149a936c9dc/mlx-porting/SKILL.md).
Revision `44d92c094bd9ca2812e222a222254149a936c9dc`; entry SHA256 `66aaedbe50ed907acb8c3ca20ea29b93c06ecd1f670822fcf831834d23c50f15`.

**Strengths:** Detailed model/config, QKV, dtype, wrapper and RNG traps; layer localization, conversion helpers and progressive references make inference ports actionable.

**Limitations:** MLX inference scope excludes JAX and training. Fixed absolute-error rules, mandatory structural copying and project-specific arsenal dependencies are not universal. Reported prior production experience was not independently rerun.

**Inspected evidence:** Full entrypoint, especially reference-reading, parity, end-to-end and framework-constraint sections; supporting pitfalls and parity material inspected for relevant warnings.

### 14. modular/skills / profile-model/SKILL.md

Score: **48 / 100**. [Pinned source](https://github.com/modular/skills/blob/d55e84c7e2a6d954b5a38eee751a75237dee1b7d/profile-model/SKILL.md).
Revision `d55e84c7e2a6d954b5a38eee751a75237dee1b7d`; entry SHA256 `e80f5545faac337baab5ac439d5547e8ead99a8ca3ec7bae33257f3a7c348b31`.

**Strengths:** Cheapest-first utilization, kernel breakdown and scoped deep profiling; good warmup, artifact checks, topology context and cleanup of owned processes only.

**Limitations:** No native conversion or training-parity workflow. Low utilization is a clue, not conclusive proof of a host bottleneck. MAX/nightly tooling and confirmation rules reduce portability.

**Inspected evidence:** Full decision ladder, kernel-mix interpretation and durable-algorithm sections.

### 15. tensormux/kernel-skills / skills/cuda/debug-cuda-kernel-correctness/SKILL.md

Score: **47 / 100**. [Pinned source](https://github.com/tensormux/kernel-skills/blob/7b7337a123f8711aa8e3d0452351d8fd30dde4b7/skills/cuda/debug-cuda-kernel-correctness/SKILL.md).
Revision `7b7337a123f8711aa8e3d0452351d8fd30dde4b7`; entry SHA256 `f5df8c37153aa895806e188dcdbe502b4a0b6539bea083ad5fbb12a0b6b9c0b2`.

**Strengths:** Minimal repro, error-pattern localization, index/stride derivation, reduction and barrier auditing, sanitizer escalation and regression cases.

**Limitations:** CUDA-only and no model-state/gradient parity. Symptom-to-cause heuristics sometimes sound stronger than evidence; changing fast math and occupancy together confounds diagnosis.

**Inspected evidence:** Full ten-step diagnosis, kernel rules and post-fix validation checklist.

### 16. tensormux/kernel-skills / skills/patterns/write-kernel-test-plan/SKILL.md

Score: **46 / 100**. [Pinned source](https://github.com/tensormux/kernel-skills/blob/7b7337a123f8711aa8e3d0452351d8fd30dde4b7/skills/patterns/write-kernel-test-plan/SKILL.md).
Revision `7b7337a123f8711aa8e3d0452351d8fd30dde4b7`; entry SHA256 `522b7196dbd083fc71febe8b9bee250048b507d5e20b26031b94fcb872c93a19`.

**Strengths:** Strong boundary, layout, precision, sanitizer and adversarial coverage with independent references and explicit failure detection.

**Limitations:** Fixed tolerance tables and a generic 10% slowdown allowance are not task-calibrated. Limited training semantics; missing standard YAML metadata lowers installation usability.

**Inspected evidence:** Full numbered test-design workflow, correctness/performance requirements and review checklist.

### 17. Orchestra-Research/AI-Research-SKILLs / 10-optimization/flash-attention/SKILL.md

Score: **46 / 100**. [Pinned source](https://github.com/Orchestra-Research/AI-Research-SKILLs/blob/773a52944ba4747a18bd4ae9ade53fff041adcbc/10-optimization/flash-attention/SKILL.md).
Revision `773a52944ba4747a18bd4ae9ade53fff041adcbc`; entry SHA256 `e0c1da83766332e0f3db4bcca954c3a70be399632758dcec2680aa82e236438d`.

**Strengths:** Useful attention layout, backend availability and memory/performance orientation with concrete framework calls.

**Limitations:** Fixed thresholds and speed claims, input generation in an example timing boundary, print-only comparison and precision-changing remedies weaken port validation. No JAX or optimizer state audit.

**Inspected evidence:** Full API, benchmark, verification and troubleshooting examples; claims are not accepted as measured results in this review.

### 18. mflux-community/mflux / .cursor/skills/mflux-model-tiny-test/SKILL.md

Score: **44 / 100**. [Pinned source](https://github.com/mflux-community/mflux/blob/051ba9ff25c9a8a8703356053a012a0dfad3fe39/.cursor/skills/mflux-model-tiny-test/SKILL.md).
Revision `051ba9ff25c9a8a8703356053a012a0dfad3fe39`; entry SHA256 `de0b9d53e7ed57c7d8a6601ba6cf6b836e1589cf200fedd223c48c036ed4257a`.

**Strengths:** Hermetic real checkpoint seams, different initialization seeds, forced shard boundaries and explicit dimension constraints. Identifies a helper defect instead of weakening assertions.

**Limitations:** Checkpoint specialist only. VAE stand-ins do not validate actual VAE math; no forward/training performance or bidirectional porting coverage.

**Inspected evidence:** Full tiny-test construction, seed/shard constraints and troubleshooting sections; cited helper behavior is documented, not rerun here.

### 19. mflux-community/mflux / .cursor/skills/mflux-debugging/SKILL.md

Score: **43 / 100**. [Pinned source](https://github.com/mflux-community/mflux/blob/051ba9ff25c9a8a8703356053a012a0dfad3fe39/.cursor/skills/mflux-debugging/SKILL.md).
Revision `051ba9ff25c9a8a8703356053a012a0dfad3fe39`; entry SHA256 `c5419364d57c3b95a7486b9a7e7f721c04077b340c4849bac1fc3c901d21af53`.

**Strengths:** Export-then-compare fixtures, exact noise injection, coarse-to-fine intermediate localization and protected golden outputs.

**Limitations:** MPS-only comparison instruction conflicts with CPU-first portable development. Visual quality and fixed starting tolerances cannot alone establish semantic parity. No gradients or optimizer state.

**Inspected evidence:** Full export workflow, divergence causes, latent injection and artifact rules.

### 20. tensormux/kernel-skills / skills/portability/port-cuda-kernel-to-triton/SKILL.md

Score: **42 / 100**. [Pinned source](https://github.com/tensormux/kernel-skills/blob/7b7337a123f8711aa8e3d0452351d8fd30dde4b7/skills/portability/port-cuda-kernel-to-triton/SKILL.md).
Revision `7b7337a123f8711aa8e3d0452351d8fd30dde4b7`; entry SHA256 `b394d1c8c580b5b55cf6f2e0afc48c39741298bfb8a64b36b81f2d20f9b1de87`.

**Strengths:** Explicit tile/index/stride mapping, masked loads and stores, accumulation choices and post-correctness tuning.

**Limitations:** Different conversion direction and abstraction level. Generic 80-100% performance expectation, small fixed test matrix and broad synchronization simplifications need qualification.

**Inspected evidence:** Full mapping workflow, correctness/performance sections and boundary checklist.

### 21. modular/skills / benchmark-model/SKILL.md

Score: **41 / 100**. [Pinned source](https://github.com/modular/skills/blob/d55e84c7e2a6d954b5a38eee751a75237dee1b7d/benchmark-model/SKILL.md).
Revision `d55e84c7e2a6d954b5a38eee751a75237dee1b7d`; entry SHA256 `8fa6f97b801ece1ef2c15083394608a4886c3e0a04705ff27997ebabaa01a222`.

**Strengths:** Explicit workload/concurrency choice, readiness checks, tokenizer/model identity, saved result metadata and latency/throughput interpretation.

**Limitations:** Served MAX workload specialist, not a cross-framework equivalence protocol. No gradient/state checks and limited uncertainty/anti-contamination methodology.

**Inspected evidence:** Full workload selection, run/save, metric and troubleshooting sections.

### 22. tensormux/kernel-skills / skills/portability/write-backend-agnostic-kernel-plan/SKILL.md

Score: **40 / 100**. [Pinned source](https://github.com/tensormux/kernel-skills/blob/7b7337a123f8711aa8e3d0452351d8fd30dde4b7/skills/portability/write-backend-agnostic-kernel-plan/SKILL.md).
Revision `7b7337a123f8711aa8e3d0452351d8fd30dde4b7`; entry SHA256 `abba92db0598d45589a43c3443a3bff30bc074754e91f5b39d72beddece1b59b`.

**Strengths:** Capability matrix, warp-size risks, backend boundaries, CPU fallback and real secondary-backend CI requirements.

**Limitations:** Hardware-kernel portability is not JAX/PyTorch model parity. Fixed percentage expectations and broad backend support statements require current verification; no optimizer or transformation validation.

**Inspected evidence:** Full backend decision, risk register, CPU fallback and CI sections.

### 23. obra/superpowers / skills/systematic-debugging/SKILL.md

Score: **39 / 100**. [Pinned source](https://github.com/obra/superpowers/blob/b36e0829c6d0140e93cfef2ca599b1b07d4a7797/skills/systematic-debugging/SKILL.md).
Revision `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`; entry SHA256 `808fc5717aa88ad65efff312b11c186294d3e6ee301afb584e2f86599b137787`.

**Strengths:** Root-cause-first minimal reproduction, boundary evidence, one hypothesis per change, regression test and three-fix escalation discipline.

**Limitations:** No numerical/model-porting specifics; universal human stops and broad environment dumps need privacy and autonomy adaptation. Anecdotal success percentages are not controlled evidence.

**Inspected evidence:** Full investigation, pattern analysis, hypothesis, implementation and escalation phases.

### 24. tensormux/kernel-skills / skills/patterns/write-numerically-stable-kernel/SKILL.md

Score: **35 / 100**. [Pinned source](https://github.com/tensormux/kernel-skills/blob/7b7337a123f8711aa8e3d0452351d8fd30dde4b7/skills/patterns/write-numerically-stable-kernel/SKILL.md).
Revision `7b7337a123f8711aa8e3d0452351d8fd30dde4b7`; entry SHA256 `402ec2a63a6acaba3d9f8ce867f2b55613b6f4e9d0ebfc28521ef33fcdefd46a`.

**Strengths:** Risk-based discussion of cancellation, accumulation and stable softmax/variance; adversarial value distributions and higher-precision references.

**Limitations:** Reviewer judgment, not independently verified against upstream: some FP32 subnormal and low-precision digit statements may be inaccurate and need checking; unconditional logit clipping and variance replacement can change source semantics. Several uncalibrated numeric rules; no standard frontmatter.

**Inspected evidence:** Full risk taxonomy and common-failure sections; numerical claims cross-checked against framework type information and official precision documentation.

### 25. mindrally/skills / jax-best-practices/SKILL.md

Score: **17 / 100**. [Pinned source](https://github.com/mindrally/skills/blob/97184105b5daa3a6860a2aeb8e7e7fd1c42da40a/jax-best-practices/SKILL.md).
Revision `97184105b5daa3a6860a2aeb8e7e7fd1c42da40a`; entry SHA256 `636555b9bc87159b6b50b86e5bafb8bac7afaf159bff0cb6dda7b16872df8031`.

**Strengths:** Very small context footprint; reminds agents about pure functions, keys, transformations and pytrees.

**Limitations:** Mostly generic reminders, without runnable validation, state/checkpoint mapping, stopping rules, optimizer parity or supporting evidence. Not an executable porting workflow.

**Inspected evidence:** Entire 53-line entrypoint; unmentioned criteria receive zero rather than inferred parent-repository credit.

## Our score evidence

| Criterion | Score | Evidence / limitation |
|---|---:|---|
| C01 JAX | 4 | references/semantics.md |
| C02 PyTorch | 4 | references/semantics.md |
| C03 Bidirectional | 5 | validation/results/resnet-training.json and rnn-training.json |
| C04 Inference | 4 | validation/results/resnet-forward.json and rnn-training.json |
| C05 Training | 4 | references/training.md and both real training reports |
| C06 Numerics | 4 | scripts/parity.py and tests/test_parity.py |
| C07 Gradients | 4 | both real training reports and semantic-probes.json |
| C08 Optimizer | 4 | semantic-probes.json AdamW and ResNet momentum checks |
| C09 Performance | 4 | references/performance.md and three CPU benchmark reports |
| C10 Accelerators | 3 | references/execution.md; GPU/TPU/distributed not run |
| C11 Localization | 4 | references/validation.md and agent-traces.md |
| C12 Recovery | 4 | references/recovery.md and actual interruption recovery |
| C13 Fast fail | 4 | SKILL.md staged gates and negative tests |
| C14 Adversarial | 4 | tests/test_parity.py, semantic-probes.json, adversarial-review.md |
| C15 Regression | 4 | code fingerprints and repeated model suites |
| C16 Harness | 3 | INSTALL.md and installation-smoke.json; no full harness agent trials |
| C17 Context | 4 | Compact core with eight on-demand references; measured package footprint |
| C18 Documentation | 4 | README.md, source index and provenance |
| C19 Usability | 4 | native bounded ports, tested helper, package checks |
| C20 Evidence | 4 | pinned source ledger, traces, iteration log and explicit negative findings |

## Interpretation and missing evaluation

A high aggregate favors broad bidirectional coverage. A CUDA microbenchmark or compiler-debugging specialist can be stronger in its own domain despite a lower total. Rows were scored against their inspected skill scope, not parent-repository popularity, stars or unrelated tooling.

No matched agent trials were run on these 25 skills. A defensible success-rate ranking would require equal models, budgets, tasks, hidden test sets and independent judges. Small ordinal score differences are not meaningful confidence intervals. Our negative ResNet performance result remains a failed development performance gate, regardless of this documentation score.

Reproduce the table with `python tools/render_comparison.py`. The renderer validates totals and ordering; it does not validate subjective judgments. A new rubric version requires rescoring every row.
