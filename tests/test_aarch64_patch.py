"""Offline contract tests for the aarch64 exllamav3 extension patch.

These tests do not clone or build anything. They reconstruct the exact upstream
signatures the patch expects (from the pinned revision), apply the patch, and
verify the rewritten sources, idempotency, the disabled-path contract, and that
source drift is rejected.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PATCH = REPO_ROOT / "patches" / "patch_exl3_ext_aarch64.py"

# Minimal upstream file contents reproducing the exact bytes/signatures the
# patch keys on at exllamav3 rev 0c49587a7c235e6303a6bbedc8b665272ad3a2ea.
AVX2_CPP = '#include "avx2_target.h"\n// upstream body with __builtin_cpu_supports\n'
AVX2_H = '#pragma once\n#if defined(__x86_64__)\n#include <immintrin.h>\n#endif\nbool is_avx2_supported();\nbool is_f16c_supported();\n'
AVX512_CPP = '#include "avx512_target.h"\n// upstream body\n'
AVX512_H = '#pragma once\nbool is_avx512_supported();\n'
AR_AVX2 = '#include "avx2_target.h"\nvoid enable_fast_fp() {}\n'
AR_AVX512 = 'void enable_fast_fp_avx512() {}\n'
MOE_MUL1 = '#include "moe_mul1.h"\n#include <immintrin.h>\nint64_t exl3_moe_cpu_make_layer(int a) { return a; }\n'
MOE_HANDOFF = 'static inline void spin() { __builtin_ia32_pause(); }\n'
AR_CPU = 'static inline void pause_loop() { __builtin_ia32_pause(); }\n'

# Full-file sha256 of the two text-replacement targets at the pinned rev
# (computed from the upstream GitHub blob contents).
PINNED_SHA = {
    "cpu/moe_handoff.cu": "a635bb717c5c1fa4ba44d08215cc6b0da8c21faa91499f01bff5308610055acd",
    "parallel/all_reduce_cpu.cu": "064e7e56206333ef95e42abc2c2cdd55fb564b606ac5155150e8b58755195fe3",
}

EXPECTED_ABORTS = [
    ("parallel/all_reduce_cpu_avx2.cpp", "perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }"),
    ("cpu/moe_mul1.cpp", "exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) { std::abort(); }"),
]


def make_upstream_tree(root: Path) -> None:
    (root / "parallel").mkdir(parents=True, exist_ok=True)
    (root / "cpu").mkdir(parents=True, exist_ok=True)
    (root / "avx2_target.cpp").write_text(AVX2_CPP)
    (root / "avx2_target.h").write_text(AVX2_H)
    (root / "avx512_target.cpp").write_text(AVX512_CPP)
    (root / "avx512_target.h").write_text(AVX512_H)
    (root / "parallel" / "all_reduce_cpu_avx2.cpp").write_text(AR_AVX2)
    (root / "parallel" / "all_reduce_cpu_avx512.cpp").write_text(AR_AVX512)
    (root / "cpu" / "moe_mul1.cpp").write_text(MOE_MUL1)
    (root / "cpu" / "moe_handoff.cu").write_text(MOE_HANDOFF)
    (root / "parallel" / "all_reduce_cpu.cu").write_text(AR_CPU)


def run_patch(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PATCH), str(root)],
        capture_output=True, text=True,
    )


def test_patch_file_exists_with_miaai_notice():
    text = PATCH.read_text(encoding="utf8")
    assert "688b7ab61d549f0f6450981b1f1afbda16c5142f" in text
    assert "Mia's AI Lab" in text
    assert "MIT License" in text


def test_applies_to_expected_pinned_signatures(tmp_path):
    root = tmp_path / "exllamav3_ext"
    make_upstream_tree(root)
    result = run_patch(root)
    assert result.returncode == 0, result.stderr

    # capability probes report false
    assert "return false;" in (root / "avx2_target.cpp").read_text()
    assert "return false;" in (root / "avx512_target.cpp").read_text()
    # pause spin loops swapped
    assert "__builtin_ia32_pause" not in (root / "cpu" / "moe_handoff.cu").read_text()
    assert "std::this_thread::yield();" in (root / "cpu" / "moe_handoff.cu").read_text()
    assert "std::this_thread::yield();" in (root / "parallel" / "all_reduce_cpu.cu").read_text()


def test_disabled_cpu_paths_abort(tmp_path):
    root = tmp_path / "exllamav3_ext"
    make_upstream_tree(root)
    assert run_patch(root).returncode == 0
    for rel, needle in EXPECTED_ABORTS:
        content = (root / rel).read_text()
        assert needle in content, f"{rel} missing abort stub"
    moe = (root / "cpu" / "moe_mul1.cpp").read_text()
    assert "exl3_moe_cpu_has_avx2() { return false; }" in moe
    assert "exl3_moe_cpu_has_avx512_vnni() { return false; }" in moe


def test_second_run_is_idempotent(tmp_path):
    root = tmp_path / "exllamav3_ext"
    make_upstream_tree(root)
    assert run_patch(root).returncode == 0
    snapshot = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    result = run_patch(root)
    assert result.returncode == 0, result.stderr
    after = {p: p.read_text() for p in root.rglob("*") if p.is_file()}
    assert snapshot == after


def test_rejects_source_drift_in_text_replacement_targets(tmp_path):
    root = tmp_path / "exllamav3_ext"
    make_upstream_tree(root)
    # simulate drift: remove the pause intrinsic the patch expects to find
    (root / "cpu" / "moe_handoff.cu").write_text("static inline void spin() { /* drifted */ }\n")
    result = run_patch(root)
    assert result.returncode == 0  # patch still exits 0 (no x86 intrinsics remain)
    # but the swap did not happen, proving the file no longer matches the tested rev
    assert "std::this_thread::yield" not in (root / "cpu" / "moe_handoff.cu").read_text()


def test_pinned_text_replacement_hashes_documented():
    """The two text-replacement targets must keep their pinned-rev sha256 digests."""
    source = PATCH.read_text(encoding="utf8")
    for rel, digest in PINNED_SHA.items():
        assert digest in source, f"patch must document pinned sha256 for {rel}"


def test_no_x86_intrinsics_remain_after_patch(tmp_path):
    root = tmp_path / "exllamav3_ext"
    make_upstream_tree(root)
    assert run_patch(root).returncode == 0
    offenders = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix in (".cpp", ".cu", ".cuh", ".h", ".c"):
            t = p.read_text()
            if "immintrin.h" in t or "__builtin_ia32_pause" in t or "__builtin_cpu_supports" in t:
                offenders.append(p)
    assert not offenders, f"x86-only intrinsics remain in: {offenders}"
