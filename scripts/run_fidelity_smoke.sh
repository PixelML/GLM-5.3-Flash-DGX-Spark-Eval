#!/usr/bin/env bash
#
# run_fidelity_smoke.sh — wire the first decisive fidelity run to UPSTREAM exllamav3 qbench.
#
# qbench CLI flags verified against the pinned source (exllamav3 @
# 0c49587a7c235e6303a6bbedc8b665272ad3a2ea, eval/qbench.py, argparse allow_abbrev=False):
#   - positional <project.yml>   (the qbench YAML project file; required)
#   - -d / --device <int>        (CUDA device index; default 0)
#   There are NO other CLI flags. Every other input (test set, models, noise floor,
#   outputs) lives in the project file, which this script GENERATES from
#   configs/first_decisive_run.json.
#
# qbench has no upstream mode for "score arbitrary text rows": test_data supports only
# wiki2/openwebtext, and test_trace must be pre-tokenized model-sampled responses
# (normally produced by qbench_prompts.py self-sampling). This script therefore never
# static-text tokenizes the fidelity rows. It obtains a test_trace one of two ways:
#   (1) TRACE_FILE=/path/to/existing/test_trace.json — use it as-is; or
#   (2) TRACE_ENDPOINT=http://localhost:30000/v1 — run scripts/build_self_trace.py to
#       self-sample the rows from the candidate's serving endpoint (needs BASE_MODEL_DIR
#       for the tokenizer and TRACE_MODEL for the served model name).
# There is no fallback to static-text tokenization. qbench.py is never patched or forked.
#
# Usage:
#   DRY_RUN=1 ./scripts/run_fidelity_smoke.sh        # print pipeline + qbench command; no models needed
#   BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash \
#   QUANT_MODEL_DIR=/path/to/LibertAIDAI/GLM-5.3-Flash-NVFP4 \
#   TRACE_ENDPOINT=http://localhost:30000/v1 \
#   TRACE_MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 \
#     ./scripts/run_fidelity_smoke.sh                 # self-sample trace + run qbench
#   ... TRACE_FILE=/path/to/test_trace.json ./scripts/run_fidelity_smoke.sh  # use existing trace
#
# Env: CONFIG (config JSON), RUN_DIR (output dir), DEVICE (CUDA index),
#      BASE_MODEL_DIR / QUANT_MODEL_DIR (checkpoint paths), DRY_RUN (1 to preview),
#      TRACE_FILE / TRACE_ENDPOINT / TRACE_MODEL (trace source).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="${CONFIG:-configs/first_decisive_run.json}"
EXL3_DIR="third_party/exllamav3"
QBENCH="$EXL3_DIR/eval/qbench.py"
BUILD_TRACE="$REPO_ROOT/scripts/build_self_trace.py"

# Normalize DRY_RUN to 0/1
if [[ "${DRY_RUN:-0}" =~ ^(1|true|TRUE|yes|YES)$ ]]; then
  DRY_RUN=1
else
  DRY_RUN=0
fi

# qbench itself is only needed for a real run; DRY_RUN=1 must work on a fresh clone
# (before ./bootstrap.sh) so the exact intended pipeline can be previewed anywhere.
if [[ "$DRY_RUN" != "1" && ! -f "$QBENCH" ]]; then
  echo "ERROR: $QBENCH not found. Run ./bootstrap.sh first (clones exllamav3 at the pinned rev)." >&2
  exit 1
fi

# Resolve the output dir (RUN_DIR env or config output_dir or results/runs/<run_id>).
RUN_DIR="$(python3 -c 'import json, sys, os
c = json.load(open(sys.argv[1]))
print(os.path.abspath(os.environ.get("RUN_DIR") or c.get("output_dir") or os.path.join("results/runs", c["run_id"])))' "$CONFIG")"
mkdir -p "$RUN_DIR"

# --- Resolve the test_trace source ------------------------------------------------
# Priority: (1) TRACE_FILE env (existing trace), (2) TRACE_ENDPOINT env (self-sample via
# build_self_trace.py), (3) error. There is NO static-text tokenization fallback.
TRACE_PATH=""
if [[ "$DRY_RUN" == "1" ]]; then
  TRACE_PATH="<TRACE_FILE>"
elif [[ -n "${TRACE_FILE:-}" ]]; then
  if [[ -f "$TRACE_FILE" ]]; then
    TRACE_PATH="$(cd "$(dirname "$TRACE_FILE")" && pwd)/$(basename "$TRACE_FILE")"
  else
    echo "ERROR: TRACE_FILE=$TRACE_FILE does not exist." >&2
    exit 1
  fi
elif [[ -n "${TRACE_ENDPOINT:-}" ]]; then
  if [[ -z "${BASE_MODEL_DIR:-}" ]]; then
    echo "ERROR: TRACE_ENDPOINT is set, so build_self_trace.py needs BASE_MODEL_DIR for the tokenizer." >&2
    echo "       Pass BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash (or --tokenizer-dir)." >&2
    exit 1
  fi
  if [[ -z "${TRACE_MODEL:-}" ]]; then
    echo "ERROR: TRACE_ENDPOINT is set, so TRACE_MODEL (the served model name) is required." >&2
    exit 1
  fi
  TRACE_PATH="$RUN_DIR/test_trace.json"
  python3 "$BUILD_TRACE" --config "$CONFIG" --trace-out "$TRACE_PATH"
else
  echo "ERROR: no test_trace source. Provide one of:" >&2
  echo "  TRACE_FILE=/path/to/existing/test_trace.json" >&2
  echo "  TRACE_ENDPOINT=http://localhost:30000/v1 (+ BASE_MODEL_DIR for the tokenizer, TRACE_MODEL for the served model)" >&2
  echo "See: python3 scripts/build_self_trace.py --help" >&2
  exit 1
fi

# Build the qbench project. The python process prints exactly one line to stdout: the
# absolute path of the generated project file.
PROJECT_YML="$(python3 - "$CONFIG" "$DRY_RUN" "$TRACE_PATH" <<'PY'
import json, os, sys

cfg_path, dry_run, trace_path = sys.argv[1], sys.argv[2], sys.argv[3]
dry_run = (dry_run == "1")

with open(cfg_path, "r", encoding="utf8") as f:
    cfg = json.load(f)

run_dir = os.path.abspath(os.environ.get("RUN_DIR") or cfg.get("output_dir")
                          or os.path.join("results/runs", cfg["run_id"]))
os.makedirs(run_dir, exist_ok=True)

def model_dir(mspec):
    d = os.environ.get(mspec.get("env_override")) or mspec.get("local_dir")
    return os.path.abspath(d) if d else None

def model_source(mspec):
    if dry_run:
        return "<%s>" % mspec.get("env_override", "MODEL_DIR")
    d = model_dir(mspec)
    if not d or not os.path.isdir(d):
        env = mspec.get("env_override")
        print(f"ERROR: {env} unset/absent and {mspec['label']} local_dir is null; "
              f"pass {env}=/path/to/checkpoint (see {cfg_path}).", file=sys.stderr)
        sys.exit(1)
    return d

base_src = model_source(cfg["reference"])
quant_src = model_source(cfg["candidate"])


def yaml_scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)  # JSON strings are valid YAML scalars
    raise TypeError(f"unsupported YAML scalar type: {type(v)}")


def yaml_lines(obj, indent=0):
    pad = " " * indent
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, dict):
                out.append(f"{pad}{key}:")
                out.extend(yaml_lines(v, indent + 2))
            elif isinstance(v, list):
                if not v:
                    out.append(f"{pad}{key}: []")
                else:
                    out.append(f"{pad}{key}:")
                    out.extend(yaml_lines(v, indent + 2))
            else:
                out.append(f"{pad}{key}: {yaml_scalar(v)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                out.append(f"{pad}-")
                out.extend(yaml_lines(item, indent + 2))
            elif isinstance(item, list):
                out.append(f"{pad}-")
                out.extend(yaml_lines(item, indent + 2))
            else:
                out.append(f"{pad}- {yaml_scalar(item)}")
    else:
        out.append(f"{pad}{yaml_scalar(obj)}")
    return out


def model_entry(mspec, group, source):
    entry = {
        "label": mspec["label"],
        "group": group,
        "engine": mspec["engine"],
        "source": source,
    }
    if mspec.get("options"):
        entry["options"] = mspec["options"]
    return entry


project = {
    "title": cfg["title"],
    "test_trace": trace_path,
    "logit_cache": {
        "dir": os.path.join(run_dir, "_logit_cache"),
        "max_size_gb": 50,
    },
    "models": [
        model_entry(cfg["reference"], "reference", base_src),
        model_entry(cfg["candidate"], "candidate", quant_src),
    ],
    "noise_floor": bool(cfg.get("noise_floor", True)),
    "output": {
        "results": os.path.join(run_dir, "qb_results.json"),
        "plot_kld_spread": os.path.join(run_dir, "qb_kld_spread.png"),
        "plot_kld_hist": os.path.join(run_dir, "qb_kld_hist.png"),
        "plot_kld_hist_combined": {
            "file": os.path.join(run_dir, "qb_kld_hist_combined.png"),
            "labels": [cfg["candidate"]["label"]],
        },
    },
}

project_path = os.path.join(run_dir, "project.yml")
with open(project_path, "w", encoding="utf8") as f:
    f.write("\n".join(yaml_lines(project)) + "\n")

# The only stdout line: the project path (captured by the shell).
print(project_path)
PY
)"

CMD=(python3 "$QBENCH" "$PROJECT_YML" -d "${DEVICE:-0}")

echo "qbench command (cwd=$REPO_ROOT):"
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN=1: not executing. Intended pipeline for a real run:"
  echo "  1. Self-sample the test_trace from the candidate's serving endpoint"
  echo "     (no static-text tokenization):"
  echo "       TRACE_ENDPOINT=http://localhost:30000/v1 \\"
  echo "       TRACE_MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 \\"
  echo "       BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash \\"
  echo "         python3 scripts/build_self_trace.py --config $CONFIG --trace-out $RUN_DIR/test_trace.json"
  echo "     (or supply an existing trace via TRACE_FILE=/path/to/test_trace.json)"
  echo "  2. Then run qbench with the transformers+streaming engines:"
  echo "       BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash \\"
  echo "       QUANT_MODEL_DIR=/path/to/LibertAIDAI/GLM-5.3-Flash-NVFP4 \\"
  echo "       TRACE_ENDPOINT=http://localhost:30000/v1 \\"
  echo "       TRACE_MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 \\"
  echo "         $0"
  exit 0
fi

exec "${CMD[@]}"
