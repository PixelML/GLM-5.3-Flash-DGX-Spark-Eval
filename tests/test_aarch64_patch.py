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

# Exact upstream file contents at exllamav3 rev 0c49587a7c235e6303a6bbedc8b665272ad3a2ea
# (full files are fetched lazily from raw.githubusercontent.com on first use and cached
# in the repo-ignored .pytest_cache; tests are skipped offline unless the cache exists).
UPSTREAM_BASE = "https://raw.githubusercontent.com/turboderp-org/exllamav3/0c49587a7c235e6303a6bbedc8b665272ad3a2ea/exllamav3/exllamav3_ext"
UPSTREAM_FILES = [
    "avx2_target.cpp", "avx2_target.h",
    "avx512_target.cpp", "avx512_target.h",
    "parallel/all_reduce_cpu_avx2.cpp", "parallel/all_reduce_cpu_avx512.cpp",
    "cpu/moe_mul1.cpp", "cpu/moe_handoff.cu", "parallel/all_reduce_cpu.cu",
]
CACHE_DIR = REPO_ROOT / ".pytest_cache" / "upstream_exl3"


def fetch_upstream(rel: str) -> str:
    """Return the exact pinned upstream file content, caching it locally."""
    cached = CACHE_DIR / rel
    if cached.is_file():
        return cached.read_text(encoding="utf8")
    import urllib.request
    url = f"{UPSTREAM_BASE}/{rel}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)
    return data.decode("utf8")


def upstream_available() -> bool:
    try:
        fetch_upstream(UPSTREAM_FILES[0])
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not upstream_available(),
    reason="pinned upstream sources not cached and network unavailable",
)

EXPECTED_ABORTS = [
    ("parallel/all_reduce_cpu_avx2.cpp", "perform_cpu_reduce(PGContext*, size_t, uint32_t, uint32_t, uint8_t*, size_t) { std::abort(); }"),
    ("cpu/moe_mul1.cpp", "exl3_moe_cpu_forward(int64_t, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t) { std::abort(); }"),
]


def make_upstream_tree(root: Path) -> None:
    (root / "parallel").mkdir(parents=True, exist_ok=True)
    (root / "cpu").mkdir(parents=True, exist_ok=True)
    for rel in UPSTREAM_FILES:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fetch_upstream(rel), encoding="utf8")


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
    # simulate one-byte drift in a text-replacement target
    original = (root / "cpu" / "moe_handoff.cu").read_text()
    drifted = original.replace("__builtin_ia32_pause", "__builtin_ia32_pause ", 1)
    (root / "cpu" / "moe_handoff.cu").write_text(drifted)
    result = run_patch(root)
    assert result.returncode != 0, "patch must fail closed on source drift"
    assert "does not match the pinned upstream preimage" in result.stderr


def test_rejects_drift_in_any_touched_file(tmp_path):
    for rel in UPSTREAM_FILES:
        root = tmp_path / rel.replace("/", "_")
        make_upstream_tree(root)
        original = (root / rel).read_text()
        (root / rel).write_text(original + "\n// drift\n")
        result = run_patch(root)
        assert result.returncode != 0, f"patch must reject drift in {rel}"
        assert "does not match the pinned upstream preimage" in result.stderr


def test_pinned_text_replacement_hashes_documented():
    """Every touched file must keep its pinned-rev sha256 digest in the patch."""
    source = PATCH.read_text(encoding="utf8")
    for rel in UPSTREAM_FILES:
        import hashlib
        content = fetch_upstream(rel).encode("utf8")
        digest = hashlib.sha256(content).hexdigest()
        assert digest in source, f"patch must pin sha256 {digest} for {rel}"


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
