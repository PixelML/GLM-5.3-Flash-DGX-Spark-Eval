# Third-Party Notices

This repository adapts and references third-party open-source work. The
following notices identify the exact sources and licenses.

## exllamav3 (runtime dependency, not committed)

- Source: https://github.com/turboderp-org/exllamav3/tree/0c49587a7c235e6303a6bbedc8b665272ad3a2ea
- Commit: `0c49587a7c235e6303a6bbedc8b665272ad3a2ea` (v1.4.4)
- License: MIT, Copyright (c) 2026 turboderp-org
- Use: cloned into git-ignored `third_party/exllamav3` by `bootstrap.sh` and
  rewritten in-place on aarch64 by `patches/patch_exl3_ext_aarch64.py` before
  building the extension. Never committed to this repository.

## patches/patch_exl3_ext_aarch64.py (adapted code, committed here)

- Source: https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/blob/688b7ab61d549f0f6450981b1f1afbda16c5142f/overlay/patch_exl3_ext_aarch64.py
- Commit: `688b7ab61d549f0f6450981b1f1afbda16c5142f`
- License: MIT, Copyright (c) 2026 Mia's AI Lab
- Adapted for exllamav3 v1.4.4 at the commit above. Retains the original
  stubs for `avx2_target.cpp`, `avx512_target.cpp`,
  `parallel/all_reduce_cpu_avx{2,512}.cpp`, and `cpu/moe_mul1.cpp`; adds
  stubbing/edits for `cpu/moe_handoff.cu` and `parallel/all_reduce_cpu.cu`
  (`__builtin_ia32_pause` to `std::this_thread::yield()`). Disabled paths:
  CPU tensor-parallel all-reduce and CPU-MoE offload abort at runtime;
  single-GPU CUDA kernels are unchanged.

## glm-simple-evals (later-phase dependency, not vendored)

- Source: https://github.com/zai-org/glm-simple-evals/tree/b67cb0bf655f8e08c19b1811a1d3c51e3bfea096
- Commit: `b67cb0bf655f8e08c19b1811a1d3c51e3bfea096`
- License: MIT, zai-org
- Use: capability evals in a later phase; not cloned or committed in Phase 0.
