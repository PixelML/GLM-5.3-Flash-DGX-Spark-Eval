#!/usr/bin/env python3
"""build_self_trace.py — self-sample qbench test_trace rows from an OpenAI-compatible
serving endpoint, then tokenize them with the reference tokenizer's chat template.

Why this exists: qbench has no "score arbitrary text" mode. `test_data` supports only
wiki2/openwebtext, and `test_trace` must be pre-tokenized model-sampled responses
(normally produced by `qbench_prompts.py` self-sampling). Tokenizing static prompts and
scoring them as if they were responses inflates the noise floor with out-of-distribution
text. This script instead:

  1. POSTs each `data/fidelity_rows/*.json` prompt to `{TRACE_ENDPOINT}/chat/completions`
     and stores the assistant's text.
  2. Loads the HF tokenizer from BASE_MODEL_DIR (tokenizer only — NO weights).
  3. Applies the tokenizer's chat template to (user prompt, assistant response), splits
     the sampled response tokens off a generation-prompt prefix, and writes a qbench
     `test_trace` JSON that `qbench/data.py` consumes unchanged.

Determinism note: SAMPLE_SEED is passed through as the request `seed` field, but not all
servers honor it. Record the server identity via TRACE_SERVER / --server-label so runs
remain attributable. TRACE_TIMEOUT (default 600s) bounds each request.

Exit status: 0 on full success (or partial success with --allow-partial and >=2 rows);
nonzero otherwise (e.g. all rows failed, or <2 rows succeeded with --allow-partial).
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 600
DEFAULT_SEED = 1730
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9


def to_list(ids):
    """Normalize apply_chat_template/encode output to a list of ints."""
    if hasattr(ids, "tolist"):
        return [int(x) for x in ids.tolist()]
    return [int(x) for x in ids]


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)


def sample_assistant(endpoint, model, prompt, *, temperature, top_p, max_tokens,
                     seed, api_key, timeout):
    url = endpoint.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf8"), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf8"))
    content = payload["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise ValueError("server returned empty assistant content")
    return content.strip()


def fallback_tokens(tokenizer, response):
    """Tokenize the response alone; require >= 2 tokens for a one-token context prefix."""
    tok = to_list(tokenizer.encode(response, add_special_tokens=False))
    if len(tok) < 2:
        raise ValueError(
            f"response tokenized to {len(tok)} tokens (need >= 2 for a context prefix)"
        )
    return tok


def split_template(tokenizer, prompt, response):
    """Split (prompt, response) into (input_ids, response_ids) via the chat template.

    Returns (input_ids, response_ids, split_method, fallback_reason). Falls back to
    tokenizing the response alone with a one-token context prefix when the generation
    prompt prefix does not verify as a prefix of the full templated ids.
    """
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    prefix_messages = [{"role": "user", "content": prompt}]
    try:
        full = to_list(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        ))
        prefix = to_list(tokenizer.apply_chat_template(
            prefix_messages, tokenize=True, add_generation_prompt=True
        ))
    except Exception as exc:  # noqa: BLE001 — chat template missing/unsupported
        tok = fallback_tokens(tokenizer, response)
        return tok[:1], tok[1:], "fallback", f"chat_template_error: {exc}"

    if len(prefix) < len(full) and full[:len(prefix)] == prefix:
        return prefix, full[len(prefix):], "chat_template_prefix", None

    tok = fallback_tokens(tokenizer, response)
    return tok[:1], tok[1:], "fallback", "prefix_not_verified"


def main():
    parser = argparse.ArgumentParser(
        description="Self-sample qbench test_trace rows from an OpenAI-compatible endpoint.",
        epilog=(
            "Determinism: SAMPLE_SEED is passed through as the request 'seed' field, but "
            "not all servers honor it. Set TRACE_SERVER or --server-label so runs are "
            "attributable to a specific serving stack."
        ),
    )
    parser.add_argument("--config", default="configs/first_decisive_run.json",
                        help="run config JSON (default: configs/first_decisive_run.json)")
    parser.add_argument("--trace-out", required=True,
                        help="output path for the qbench test_trace JSON")
    parser.add_argument("--tokenizer-dir", default=None,
                        help="override BASE_MODEL_DIR for the tokenizer directory")
    parser.add_argument("--server-label", default=None,
                        help="record this server identity in trace meta")
    parser.add_argument("--allow-partial", action="store_true",
                        help="proceed and write a partial trace if >=2 rows succeeded")
    parser.add_argument("--timeout", type=float, default=None,
                        help="per-request timeout in seconds (default: TRACE_TIMEOUT or 600)")
    args = parser.parse_args()

    endpoint = os.environ.get("TRACE_ENDPOINT")
    if not endpoint:
        sys.exit("ERROR: TRACE_ENDPOINT is required "
                 "(OpenAI-compatible base URL, e.g. http://localhost:30000/v1).")
    model = os.environ.get("TRACE_MODEL")
    if not model:
        sys.exit("ERROR: TRACE_MODEL is required (served model name).")

    tokenizer_dir = args.tokenizer_dir or os.environ.get("BASE_MODEL_DIR")
    if not tokenizer_dir:
        sys.exit("ERROR: BASE_MODEL_DIR (or --tokenizer-dir) is required for the tokenizer.")
    if not os.path.isdir(tokenizer_dir):
        sys.exit(f"ERROR: tokenizer dir not found: {tokenizer_dir}")

    with open(args.config, "r", encoding="utf8") as f:
        cfg = json.load(f)

    max_tokens = int(os.environ.get("MAX_TOKENS") or cfg.get("token_budget", 512))
    seed = int(os.environ.get("SAMPLE_SEED") or DEFAULT_SEED)
    temperature = float(os.environ.get("TEMPERATURE") or DEFAULT_TEMPERATURE)
    top_p = float(os.environ.get("TOP_P") or DEFAULT_TOP_P)
    timeout = args.timeout or float(os.environ.get("TRACE_TIMEOUT") or DEFAULT_TIMEOUT)
    api_key = os.environ.get("TRACE_API_KEY")
    server = args.server_label or os.environ.get("TRACE_SERVER") or "unknown"

    rows_dir = cfg.get("rows_dir", "data/fidelity_rows")
    row_files = sorted(
        os.path.join(rows_dir, f) for f in os.listdir(rows_dir) if f.endswith(".json")
    )
    if not row_files:
        sys.exit(f"ERROR: no fidelity rows found in {rows_dir}.")

    tokenizer = load_tokenizer(tokenizer_dir)
    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))

    trace_rows = []
    failures = []
    for rf in row_files:
        with open(rf, "r", encoding="utf8") as f:
            row_meta = json.load(f)
        row_id = row_meta["id"]
        prompt = row_meta["prompt"]
        try:
            response = sample_assistant(
                endpoint, model, prompt,
                temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                seed=seed, api_key=api_key, timeout=timeout,
            )
            input_ids, response_ids, split_method, fallback_reason = split_template(
                tokenizer, prompt, response
            )
            if not response_ids:
                raise ValueError("empty response_ids after split")
            entry = {
                "input_ids": input_ids,
                "response_ids": response_ids,
                "domain": row_meta["domain"],
                "source_row_id": row_id,
                "prefix_split": split_method,
            }
            if fallback_reason:
                entry["fallback_reason"] = fallback_reason
            trace_rows.append(entry)
        except Exception as exc:  # noqa: BLE001 — one bad row must not sink the rest
            failures.append({
                "source_row_id": row_id,
                "error": f"{type(exc).__name__}: {exc}",
            })

    n_total = len(row_files)
    n_ok = len(trace_rows)

    print(f"Sampled {n_ok}/{n_total} rows from {endpoint} "
          f"(model={model}, seed={seed}, temperature={temperature}, top_p={top_p}, "
          f"max_tokens={max_tokens}).")
    for f in failures:
        print(f"  FAILED {f['source_row_id']}: {f['error']}", file=sys.stderr)

    if n_ok == 0:
        sys.exit("ERROR: all rows failed; no test_trace written.")
    if n_ok < n_total:
        if args.allow_partial and n_ok >= 2:
            print(f"WARNING: {n_total - n_ok} row(s) failed; "
                  f"--allow-partial given, writing partial trace.", file=sys.stderr)
        else:
            sys.exit(
                f"ERROR: {n_total - n_ok} row(s) failed and no --allow-partial "
                f"(or <2 rows succeeded); no test_trace written."
            )

    trace = {
        "model": model,
        "vocab_size": vocab_size,
        "tokenizer_dir": tokenizer_dir,
        "template_provenance": {
            "method": "apply_chat_template (user+assistant, add_generation_prompt=False); "
                      "prefix split via add_generation_prompt=True",
            "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        },
        "meta": {
            "rows": n_ok,
            "input_tokens": sum(len(r["input_ids"]) for r in trace_rows),
            "output_tokens": sum(len(r["response_ids"]) for r in trace_rows),
            "endpoint": endpoint,
            "seed": seed,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "server": server,
            "source_row_ids": [r["source_row_id"] for r in trace_rows],
        },
        "rows": trace_rows,
    }

    out_path = os.path.abspath(args.trace_out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as f:
        json.dump(trace, f, indent=2)
    print(f"Wrote test_trace: {out_path} "
          f"({n_ok} rows, {trace['meta']['output_tokens']} scored tokens, "
          f"vocab_size={vocab_size}).")


if __name__ == "__main__":
    main()
