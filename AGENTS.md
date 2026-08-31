# SparkQuant-Lab — project-local AGENTS rules

These rules are specific to this repo. They refine, not replace, the shared-tree
`AGENTS.md` at the mount root.

## Routing rule (where work runs)

- **Heavy runs (qbench, model loading, anything touching weights or VRAM) run on the DGX Spark node**
  (DGX Spark, arm64, its own GPU). This shared tree holds **code + configs only** — never
  model weights, never logit caches, never run artifacts.
- The build/orchestration workstation (shared mount) is for scaffolding, editing, and the
  offline integrity tests. It has no VRAM and not enough disk for checkpoints; do not
  attempt to run qbench or download weights here.
- Checkpoints are copied to the node's local disk out-of-band (not via this repo, not via
  this shared tree). The repo only references them by env var
  (`BASE_MODEL_DIR`, `QUANT_MODEL_DIR`).

## Pin policy

- Pinned revisions live in `manifests/runtime-pins.json` (and the matching fields in
  `configs/first_decisive_run.json`). Change pins **only** there, then rerun
  `python3 -m pytest tests/ -q` — the tests assert manifests and configs agree with each
  other and with the revs documented in `README.md`.
- Runtime versions (python/torch/CUDA) are placeholders (`resolve_on_node`) until
  measured on the DGX Spark node; fill them in via the manifest, not ad hoc.

## qbench-unchanged policy

- `eval/qbench.py` from the pinned exllamav3 rev is used **unchanged**. Do not patch, fork,
  or vendor-modify it in Phase 0.
- The smoke script feeds qbench a generated YAML project + a pre-tokenized `test_trace`
  (self-sampled from the candidate's serving endpoint by `scripts/build_self_trace.py`,
  split with the HF `AutoTokenizer` chat template). If a needed capability is missing
  upstream, record it in `README.md` "Known gaps" instead of faking it or patching qbench.
- A later-phase fork is explicitly out of scope for now; if it happens, it becomes a
  `third_party/` entry with its own pin in `manifests/runtime-pins.json`.

## Data hygiene

- `third_party/`, `results/runs/`, `__pycache__/`, `.pytest_cache/`, `*.pt`,
  `*.safetensors` are git-ignored. Never `git add` model weights or run artifacts.
- `data/fidelity_rows/*` are DRAFT scaffolding rows pending Sean's curation; treat them as
  replaceable, not final.
