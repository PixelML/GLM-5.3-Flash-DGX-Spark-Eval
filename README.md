# SparkQuant-Lab

On one or two DGX Sparks, which quantization + runtime stack preserves the most
original-model intelligence while keeping usable speed, context, and stability?

- **Primary model:** GLM-5.3-Flash
- **Validation model:** Qwen3.8-Flash-Next (later phase)
- **First experiment:** the current NVFP4 routed-expert checkpoint
  (`LibertAIDAI/GLM-5.3-Flash-NVFP4`, ~97% of params NVFP4 routed experts, rest BF16)
  vs the official streamed FP8 (e4m3 dynamic) reference (`zai-org/GLM-5.3-Flash`, rev
  `84c6a6aa…`, byte-identical to `04c4e9e9…`), measured with the **upstream,
  unmodified** exllamav3 `qbench` harness.

No benchmark numbers here are real yet. This repo is Phase 0 scaffolding: the configs,
draft data, and wiring to run the first decisive comparison. Numbers get filled in only
after the run completes on the DGX Spark node.

## Repo layout

```
README.md                        this file
AGENTS.md                        project-local working rules (routing, pins, qbench policy)
bootstrap.sh                     idempotent fetch of exllamav3 at the pinned rev
configs/first_decisive_run.json  the first decisive run spec (models, rows, budget, metrics)
manifests/spark-hardware.json    DGX Spark (GB10) advertised specs + measured fields
manifests/runtime-pins.json      single source of truth for pinned revisions / runtime
data/fidelity_rows/              4 draft domain rows (code, reasoning, tools, long-context)
scripts/build_self_trace.py      self-samples test_trace rows from a serving endpoint
scripts/run_fidelity_smoke.sh    resolves/self-samples test_trace, builds qbench project, runs qbench unchanged
results/schema/fidelity_result.schema.json   JSON Schema (2020-12) for a run result
tests/test_repo_integrity.py     offline integrity tests (pytest, no network)
third_party/                     git-ignored; exllamav3 lands here via bootstrap.sh
results/runs/                    git-ignored; qbench projects, traces, caches, results, plots
```

## Quickstart

```bash
# 1. fetch the pinned eval harness (no model weights are downloaded here, ever)
./bootstrap.sh

# 2. offline integrity checks (all JSON parses, pins match, rows valid, scripts executable)
python3 -m pytest tests/ -q

# 3. copy this repo to the node (code + configs only — never weights through the shared tree)
#    then on the DGX Spark node:
BASE_MODEL_DIR=/path/to/zai-org/GLM-5.3-Flash \
QUANT_MODEL_DIR=/path/to/LibertAIDAI/GLM-5.3-Flash-NVFP4 \
TRACE_ENDPOINT=http://localhost:30000/v1 \
TRACE_MODEL=LibertAIDAI/GLM-5.3-Flash-NVFP4 \
  ./scripts/run_fidelity_smoke.sh
#    (or reuse an existing self-sampled trace: TRACE_FILE=/path/to/test_trace.json)

# 4. preview the exact qbench command without touching any model:
DRY_RUN=1 ./scripts/run_fidelity_smoke.sh
```

`bootstrap.sh` only clones `exllamav3` into `third_party/exllamav3` at the pinned rev
(see `manifests/runtime-pins.json`). It never downloads model weights; weight fetching is
an node-side step outside this repo.

## DGX Spark execution notes

- Runs happen on a **DGX Spark node** (arm64, 128 GB unified LPDDR5x). This shared tree
  holds code + configs only — model checkpoints live on the DGX Spark node's local disk, never in git
  or in this shared mount.
- Populate `manifests/spark-hardware.json` `measured` block on first boot (nvidia-smi,
  driver/CUDA versions, allocatable VRAM). Treat the advertised GB10 specs as unverified
  until then.
- exllamav3 must be built on the DGX Spark node (its CUDA extensions compile on first import). The
  pinned `requirements.txt` needs `torch>=2.6.0`; exact torch/CUDA versions are recorded in
  `manifests/runtime-pins.json` once resolved on the DGX Spark node (`resolve_on_node`).
- `scripts/run_fidelity_smoke.sh` requires the two checkpoint dirs via
  `BASE_MODEL_DIR` / `QUANT_MODEL_DIR`, and a `test_trace` via `TRACE_FILE` (existing
  trace) or `TRACE_ENDPOINT` + `TRACE_MODEL` (self-sample). Both models use the
  `transformers` engine with `{streaming: true, trust_remote_code: true}` (see the
  weight-only-fidelity rationale under "Known gaps").
- `scripts/build_self_trace.py` passes `SAMPLE_SEED` through as the request `seed` field,
  but not all servers honor it; record the serving stack via `TRACE_SERVER` (or
  `--server-label`) so runs stay attributable. Sampling params (`temperature`, `top_p`,
  `max_tokens`, `seed`) are recorded in `configs/first_decisive_run.json`
  `trace_provenance` and echoed into the trace's `meta`.

## qbench CLI — what was verified

Cloned `exllamav3` at `0c49587a7c235e6303a6bbedc8b665272ad3a2ea` and read
`eval/qbench.py`'s argparse directly. The **only** CLI surface is:

- `eval/qbench.py <project.yml>` — positional, the qbench YAML project file (required)
- `-d, --device <int>` — primary CUDA device index (default 0)

`allow_abbrev=False`; there are no other flags. Everything else (test set, tokenizer,
models, noise floor, outputs) is expressed in the project file, which
`scripts/run_fidelity_smoke.sh` generates from `configs/first_decisive_run.json`.

Verified project-file keys (read from `eval/qbench.py`, `eval/qbench/data.py`,
`eval/qbench/engines.py`, `eval/qbench/measure.py`): `test_data`, `tokenizer`, `test_trace`,
`logit_cache`, `models[]` (`label`, `group` exactly one `reference`, `engine`, `source`,
`options`), `noise_floor`, `reference_quant`, and `output` (`results`, the `plot_*` keys).

## Known gaps / TODO for later phases

1. **No upstream "score arbitrary text" mode.** `qbench`'s `test_data` supports only
   `wiki2`/`openwebtext`; `test_trace` must be pre-tokenized (normally produced by
   `qbench_prompts.py` self-sampling). To run our 4 curated domain rows,
   `scripts/build_self_trace.py` self-samples responses from the candidate's deployed
   runtime via an OpenAI-compatible `chat/completions` endpoint, then tokenizes them with
   the reference tokenizer's chat template into a valid `test_trace` before invoking
   `qbench.py` unchanged. `qbench.py` itself is never patched or forked. W4A4 activation
   noise affects trace selection only, not the KLD reference — the reference is the
   official FP8 (e4m3 dynamic) checkpoint streamed with bf16 compute.
2. **Model bytes are derived, not emitted.** qbench reports `vram_gb` (and `bpw_layer` /
   `bpw_head`); the schema's `model_bytes` is `vram_gb * 2^30`. `ppl_delta` is likewise
   derived (candidate ppl / reference ppl). Both are documented in
   `configs/first_decisive_run.json` `expected_outputs`.
3. **transformers + NVFP4 dequant path is unverified on GB10.** Both models use the
   `transformers` engine with `{streaming: true, trust_remote_code: true}` — the streaming
   backend dequantizes modelopt + compressed-tensors nvfp4-pack-quantized checkpoints on
   the fly, so evaluation is weight-only: results are the checkpoint's weight fidelity
   under bf16 compute, i.e. it measures NVFP4-A16 weight damage — the first-decisive-run
   question. This also removes dependence on an unverified exllamav3 loader path for the
   HF NVFP4 format. Remaining risk: transformers support for the GLM-5.3 arch + NVFP4
   dequant path on GB10; verify on first node run. If it can't, that's a fork point, not
   something to fake here.
4. **Draft rows.** `data/fidelity_rows/*` are scaffolding drafts pending Sean's curation;
   token counts are approximate (whitespace). Long-context held-out documents are not yet
   included.
5. **Capability evals (later phase).** `glm-simple-evals` (pinned in
   `manifests/runtime-pins.json`, not yet cloned) covers capability-style evals that
   qbench's logit-divergence metrics don't.
6. **Perf matrix (later phase).** Speed/context/stability matrix across candidate stacks is
   out of scope for the first decisive run; `eval/perf.py` exists upstream for that later.
7. **Agentic multi-node set (later phase).** Orchestrating multiple runs / the validation model
   on the DGX Spark node is not wired yet.

## Pins

Single source of truth: `manifests/runtime-pins.json`. Exllamav3 at
`0c49587a7c235e6303a6bbedc8b665272ad3a2ea`; glm-simple-evals at
`b67cb0bf655f8e08c19b1811a1d3c51e3bfea096` (later phase). Change pins only in the manifest
(and the matching `configs/*.json`), then rerun `tests/test_repo_integrity.py`.
