#!/usr/bin/env python3
"""Stub x86-only AVX CPU targets so exllamav3_ext compiles on aarch64/GB10.

Adapted for exllamav3 rev 0c49587a (v1.4.4) from the proven arm64 build stub in
MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks (overlay/patch_exl3_ext_aarch64.py,
commit 688b7ab61d549f0f6450981b1f1afbda16c5142f, MIT license; full notice below),
which targets the older rev c5d9c657 (v0.0.43). At our pinned rev the CPU-MoE GEMM (cpu/moe_mul1.cpp)
and the fused CUDA handoff kernels additionally use x86-only intrinsics
(immintrin.h, __builtin_cpu_supports, __builtin_ia32_pause), so those two .cu files
and moe_mul1.cpp are stubbed too. All removed paths are CPU-only optimizations:
single-GPU CUDA kernels (qbench's path) are untouched.

Symbols removed by stubbing stay bound in bindings.cpp / moe_handoff.cu /
all_reduce_cpu.cu, so this file defines them as no-ops or abort() stubs to keep the
extension linkable. AVX-family capability probes report false; CPU tensor-parallel
all-reduce and CPU-MoE dispatch abort if actually invoked on this platform.

Adapted from MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks,
overlay/patch_exl3_ext_aarch64.py @ 688b7ab61d549f0f6450981b1f1afbda16c5142f.
Copyright (c) 2026 Mia's AI Lab. MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Full license text: LICENSE and THIRD-PARTY-NOTICES.md in this repository.
"""

import sys
from pathlib import Path

import hashlib

root = Path(sys.argv[1] if len(sys.argv) > 1 else "third_party/exllamav3/exllamav3/exllamav3_ext")
root = root.resolve()
assert root.is_dir(), root

# Fail closed: every file this patch touches must match its exact sha256 preimage
# at exllamav3 rev 0c49587a7c235e6303a6bbedc8b665272ad3a2ea. Any drift (wrong rev,
# dirty tree, prior partial patch that altered content) aborts before rewriting.
PINNED_PREIMAGE_SHA256 = {
    "avx2_target.cpp": "4ffa7fdba36f9f3d7787e7a8a5f78388e88ce5447c59728019470ca7cb302dd5",
    "avx2_target.h": "957e8343952d97437c94a9b6e4a5368122a78c6c876980988965c5212473fa96",
    "avx512_target.cpp": "4f9b51075f769c34884e2b1510328b814b71fa0bb71fb0fdd92b72ebc9ba447a",
    "avx512_target.h": "fa43f1957011b77d5f7b67732817ab5916aa3d6d375c6389f4ef26680e3887d4",
    "parallel/all_reduce_cpu_avx2.cpp": "a6c184ad5a35b7d16ba77461576ae9b70a4539160887f0e675207e7ae13dccec",
    "parallel/all_reduce_cpu_avx512.cpp": "9f02819e86f4617356e9dd282e877b0ba24afb87fc10fe1601adc8e8d092cc6a",
    "cpu/moe_mul1.cpp": "da4f6a50da619af4b055d57f900733f404f669abdc71c0d95b2b9ecaae32a24e",
    "cpu/moe_handoff.cu": "a635bb717c5c1fa4ba44d08215cc6b0da8c21faa91499f01bff5308610055acd",
    "parallel/all_reduce_cpu.cu": "064e7e56206333ef95e42abc2c2cdd55fb564b606ac5155150e8b58755195fe3",
}

for rel, expected in PINNED_PREIMAGE_SHA256.items():
    p = root / rel
    if not p.is_file():
        print(f"ERROR: expected pinned source missing: {rel}", file=sys.stderr)
        sys.exit(2)
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    # idempotency: already-patched content is accepted too (hash of the exact
    # bytes this script writes for that file, computed after first apply)
    import json as _json
    _postfile = Path(__file__).with_name("patched_postimages.json")
    _post = _json.loads(_postfile.read_text()) if _postfile.is_file() else {}
    post = _post.get(rel)
    if post is None:
        print(f"ERROR: patched_postimages.json missing entry for {rel}", file=sys.stderr)
        sys.exit(4)
    if actual not in (expected, post):
        print(
            f"ERROR: {rel} does not match the pinned upstream preimage.\n"
            f"       expected sha256 {expected}\n"
            f"       found    sha256 {actual}\n"
            f"       Refusing to patch a drifted or unexpected source tree.",
            file=sys.stderr,
        )
        sys.exit(3)

# --- avx2_target / avx512_target ------------------------------------------------------
(root / "avx2_target.cpp").write_text(
    '#include "avx2_target.h"\n'
    'bool is_avx2_supported() { return false; }\n'
    'bool is_f16c_supported() { return false; }\n'
)
(root / "avx512_target.cpp").write_text(
    '#include "avx512_target.h"\n'
    'bool is_avx512_supported() { return false; }\n'
)
# Headers: strip Linux target attributes / target_clones that GCC still rejects on
# aarch64 even with stub definitions.
for name, guard, extra in (
    ("avx2_target.h", "AVX2", "bool is_f16c_supported();\n"),
    ("avx512_target.h", "AVX512", ""),
):
    (root / name).write_text(
        "#pragma once\n"
        f"bool is_{name.split('_')[0]}_supported();\n"
        + extra
        + f"#define {guard}_TARGET\n"
        + f"#define {guard}_TARGET_OPTIONAL\n"
    )

# --- CPU all-reduce AVX implementations ------------------------------------------------
# Stubs for the AVX2/AVX-512 CPU all-reduce workers. cpu_reduce_parallel (declared in
# all_reduce_cpu_avx2.h, defined in the stubbed .cpp) is used by perform_cpu_reduce in
# all_reduce_cpu.cu, which itself only runs for multi-GPU tensor-parallel CPU reduce.
# Keep the declaration-available signatures linkable and aborting.
(root / "parallel" / "all_reduce_cpu_avx2.cpp").write_text(
    '#include "../avx2_target.h"\n'
    '#include "all_reduce_cpu_avx2.h"\n'
    '#include "all_reduce_cpu_avx512.h"\n'
    '#include <cstdlib>\n'
    'void enable_fast_fp() {}\n'
    'void enable_fast_fp_avx2() {}\n'
    'void perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }\n'
    'void perform_cpu_reduce_avx2(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }\n'
    'void cpu_reduce_parallel(void (*)(uint16_t*, const uint16_t*, const uint16_t*, size_t),\n'
    '                        void (*)(uint16_t*, const uint16_t*, size_t),\n'
    '                        uint16_t*, const uint16_t*, const uint16_t*, size_t, int)\n'
    '{ std::abort(); }\n'
)
(root / "parallel" / "all_reduce_cpu_avx512.cpp").write_text(
    '#include "all_reduce_cpu_avx512.h"\n'
    '#include <cstdlib>\n'
    'void enable_fast_fp_avx512() {}\n'
    'void bf16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) { std::abort(); }\n'
    'void bf16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }\n'
    'void fp16_add_inplace_avx512(uint16_t*, const uint16_t*, size_t) { std::abort(); }\n'
    'void fp16_add_twosrc_avx512(uint16_t*, const uint16_t*, const uint16_t*, size_t) { std::abort(); }\n'
    'void perform_cpu_reduce_avx512(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }\n'
)

# --- CPU MoE expert GEMM (mul1) + handoff worker ----------------------------------------
# moe_mul1.cpp is entirely AVX2/AVX-512 VNNI. moe_handoff.cu (CUDA host code) drives it
# but contains no x86 intrinsics after the pause swap below, so it stays real and its
# own definitions (exl3_moe_cpu_set_memops, exl3_moe_cpu_worker_run) remain in place.
# Only the moe_mul1.cpp symbols are stubbed here; abort if invoked: this is an opt-in
# CPU-offload path, never used by qbench's single-GPU CUDA flow.
(root / "cpu" / "moe_mul1.cpp").write_text(
    '#include "moe_mul1.h"\n'
    '#include <cstdlib>\n'
    'int64_t exl3_moe_cpu_make_layer(const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,\n'
    '                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,\n'
    '                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,\n'
    '                                const std::vector<at::Tensor>&, const std::vector<at::Tensor>&, const std::vector<at::Tensor>&,\n'
    '                                int64_t, double, int64_t) { std::abort(); }\n'
    'void exl3_moe_cpu_free_layer(int64_t) { std::abort(); }\n'
    'void exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) { std::abort(); }\n'
    'void exl3_moe_cpu_forward_raw(int64_t, const at::Half*, const int32_t*, const at::Half*, float*, int, int, int) { std::abort(); }\n'
    'void exl3_moe_cpu_stage_experts(int64_t, const uint32_t*, int, uint8_t*, int) { std::abort(); }\n'
    'void exl3_moe_cpu_set_prof(bool) {}\n'
    'bool exl3_moe_cpu_has_avx2() { return false; }\n'
    'bool exl3_moe_cpu_has_avx512_vnni() { return false; }\n'
    'bool exl3_moe_cpu_has_avx512_vbmi() { return false; }\n'
)

# --- __builtin_ia32_pause in CUDA host code --------------------------------------------
# Both spin loops are on CPU worker threads; std::this_thread::yield() is the portable
# equivalent. Patch by text replacement so the diff stays minimal.
for rel in ("cpu/moe_handoff.cu", "parallel/all_reduce_cpu.cu"):
    p = root / rel
    src = p.read_text()
    patched = src.replace("__builtin_ia32_pause();", "std::this_thread::yield();")
    if patched != src:
        p.write_text(patched)
        print(f"patched pause -> yield: {p}")

# --- final guard: no x86-only intrinsics remain anywhere in the extension ---------------
leftovers = []
for pat in ("immintrin.h", "__builtin_ia32_pause", "__builtin_cpu_supports"):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".cpp", ".cu", ".cuh", ".h", ".c"):
            try:
                if pat in p.read_text():
                    leftovers.append((pat, p))
            except OSError:
                pass
if leftovers:
    for pat, p in leftovers:
        print(f"WARNING: {pat} still present in {p}", file=sys.stderr)
    sys.exit(1)
print(f"aarch64 EXL3 x86-target stubs written in {root}")
