"""
Offline repo-integrity tests for SparkQuant-Lab (Phase 0 scaffolding).

No network access; no model downloads; no qbench execution. These tests only check
that the repo's configs, manifests, data rows, schema, and scripts are internally
consistent and well-formed.
"""

import json
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

JSON_FILES = [
    REPO_ROOT / "configs" / "first_decisive_run.json",
    REPO_ROOT / "manifests" / "spark-hardware.json",
    REPO_ROOT / "manifests" / "runtime-pins.json",
    REPO_ROOT / "results" / "schema" / "fidelity_result.schema.json",
]

EXPECTED_DOMAINS = {"code", "reasoning", "tools", "long-context"}
TOKEN_BUDGET = 512

SCHEMA_TOP_LEVEL_REQUIRED = [
    "run_id",
    "created_utc",
    "git_rev",
    "qbench_rev",
    "models",
    "rows_used",
    "metrics",
    "raw_output_path",
]
SCHEMA_METRICS_REQUIRED = [
    "kld_median",
    "kld_p90",
    "kld_high_confidence",
    "excess_kld_over_noise_floor",
    "ppl_delta",
    "stored_bpw",
    "model_bytes",
]


# --------------------------------------------------------------------------- helpers

def all_json_files():
    """Every owned JSON file, including the data/fidelity_rows glob."""
    files = list(JSON_FILES)
    files.extend(sorted((REPO_ROOT / "data" / "fidelity_rows").glob("*.json")))
    return files


def fidelity_rows():
    return sorted((REPO_ROOT / "data" / "fidelity_rows").glob("*.json"))


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- tests

@pytest.mark.parametrize("path", [p for p in all_json_files()])
def test_json_files_parse(path):
    assert path.is_file(), f"missing JSON file: {path}"
    load_json(path)  # raises on invalid JSON


def test_runtime_pins_structure():
    pins = load_json(REPO_ROOT / "manifests" / "runtime-pins.json")
    assert pins["python"] == "3.12.3"
    assert pins["exllamav3"]["rev"]
    assert pins["glm_simple_evals"]["rev"]
    for key in ("torch", "cuda"):
        assert pins[key]["version"], f"{key} version must now be measured, not null"
        assert pins[key]["status"] == "verified_on_node"


def test_model_revision_pins_present_and_full_length():
    pins = load_json(REPO_ROOT / "manifests" / "runtime-pins.json")
    for key in ("reference_model", "candidate_model"):
        entry = pins[key]
        assert entry["hf_id"], f"{key} missing hf_id"
        assert re.fullmatch(r"[0-9a-f]{40}", entry["rev"]), (
            f"{key} rev must be a full 40-char commit SHA, got {entry['rev']!r}"
        )


def test_manifests_match_readme_pinned_revs():
    pins = load_json(REPO_ROOT / "manifests" / "runtime-pins.json")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf8")

    exl3_rev = pins["exllamav3"]["rev"]
    glm_rev = pins["glm_simple_evals"]["rev"]

    assert exl3_rev in readme, "exllamav3 rev in runtime-pins.json is not pinned in README.md"
    assert glm_rev in readme, "glm-simple-evals rev in runtime-pins.json is not pinned in README.md"


def test_config_qbench_rev_matches_pins():
    pins = load_json(REPO_ROOT / "manifests" / "runtime-pins.json")
    cfg = load_json(REPO_ROOT / "configs" / "first_decisive_run.json")
    assert cfg["qbench_rev"] == pins["exllamav3"]["rev"], (
        "configs/first_decisive_run.json qbench_rev must match "
        "manifests/runtime-pins.json exllamav3 rev"
    )
    assert cfg["token_budget"] == TOKEN_BUDGET
    assert cfg["noise_floor"] is True
    assert cfg["estimate_minutes"] == [60, 180]


def test_exactly_four_fidelity_rows_cover_four_domains():
    rows_paths = fidelity_rows()
    assert len(rows_paths) == 4, f"expected exactly 4 fidelity rows, found {len(rows_paths)}"

    rows = [load_json(p) for p in rows_paths]
    ids = [r["id"] for r in rows]
    domains = [r["domain"] for r in rows]

    assert len(set(ids)) == 4, "fidelity row ids must be unique"
    assert set(domains) == EXPECTED_DOMAINS, (
        f"rows must cover domains {sorted(EXPECTED_DOMAINS)}, got {sorted(domains)}"
    )

    for row in rows:
        for key in ("id", "domain", "prompt"):
            assert key in row, f"row {row.get('id')!r} missing key {key!r}"
        assert row["prompt"].strip(), f"row {row['id']!r} has empty prompt"


def test_fidelity_rows_within_token_budget():
    for path in fidelity_rows():
        row = load_json(path)
        n_tokens = len(row["prompt"].split())
        assert n_tokens <= TOKEN_BUDGET, (
            f"row {row['id']!r} is ~{n_tokens} whitespace tokens, "
            f"over the {TOKEN_BUDGET}-token budget"
        )


def test_schema_is_valid_json_schema():
    schema = load_json(REPO_ROOT / "results" / "schema" / "fidelity_result.schema.json")
    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"

    assert schema.get("type") == "object"
    for key in SCHEMA_TOP_LEVEL_REQUIRED:
        assert key in schema.get("required", []), f"schema missing required key {key!r}"

    metrics = schema.get("properties", {}).get("metrics", {})
    for key in SCHEMA_METRICS_REQUIRED:
        assert key in metrics.get("required", []), f"schema metrics missing required key {key!r}"
    assert "kld_median" in metrics.get("properties", {})


def test_scripts_exist_and_are_executable():
    for rel in ("bootstrap.sh", "scripts/run_fidelity_smoke.sh", "scripts/build_self_trace.py"):
        path = REPO_ROOT / rel
        assert path.is_file(), f"missing script: {rel}"
        assert os.access(path, os.X_OK), f"script not executable: {rel}"


def test_build_self_trace_compiles():
    import py_compile
    path = REPO_ROOT / "scripts" / "build_self_trace.py"
    assert path.is_file(), "missing scripts/build_self_trace.py"
    py_compile.compile(str(path), doraise=True)


def test_engines_are_transformers_with_streaming():
    cfg = load_json(REPO_ROOT / "configs" / "first_decisive_run.json")
    for key in ("reference", "candidate"):
        model = cfg[key]
        assert model["engine"] == "transformers", f"{key} engine must be transformers"
        opts = model.get("options", {})
        assert opts.get("streaming") is True, f"{key} missing options.streaming=true"
        assert opts.get("trust_remote_code") is True, f"{key} missing options.trust_remote_code=true"


def test_config_trace_provenance_block():
    cfg = load_json(REPO_ROOT / "configs" / "first_decisive_run.json")
    prov = cfg["trace_provenance"]
    assert prov["method"] == "self-sampled via OpenAI-compatible endpoint"
    for key in ("temperature", "top_p", "max_tokens", "seed"):
        assert key in prov["sampling_params"], f"trace_provenance.sampling_params missing {key!r}"


def test_gitignore_covers_required_paths():
    text = (REPO_ROOT / ".gitignore").read_text(encoding="utf8")
    for entry in ("third_party/", "results/runs/", "__pycache__", ".pytest_cache", "*.pt", "*.safetensors"):
        assert entry in text, f".gitignore missing entry {entry!r}"
