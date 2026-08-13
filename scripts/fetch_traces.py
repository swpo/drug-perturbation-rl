"""Pull inference traces (prompt + completion + reward) from a list of evals,
filtered to interesting examples. Saves to a JSON dump for use in reports.

Interesting examples we want to surface:
  - Correct cycle_phase prediction on v2+3 with strong reasoning
  - Incorrect cycle_phase prediction (failure mode)
  - Same perturbation across multiple (model, mask) cells — see how reasoning changes

Usage:
    uv run python scripts/fetch_traces.py EVAL_ID1 [EVAL_ID2 ...] [--out PATH]
"""

import argparse
import json
import subprocess
from pathlib import Path


def fetch_samples(eval_id: str, page_size: int = 200) -> list[dict]:
    samples = []
    page = 1
    while True:
        out = subprocess.run(
            ["prime", "--plain", "eval", "samples", eval_id,
             "--output", "json", "-p", str(page), "-n", str(page_size)],
            capture_output=True, text=True, timeout=120,
        )
        if out.returncode != 0:
            raise RuntimeError(f"prime eval samples failed for {eval_id}: {out.stderr}")
        data = json.loads(out.stdout, strict=False)
        batch = data.get("samples", [])
        if not batch:
            break
        samples.extend(batch)
        total = data.get("total")
        if total and len(samples) >= total:
            break
        page += 1
        if page > 100:
            break
    return samples


def fetch_model_name(eval_id: str) -> str:
    out = subprocess.run(
        ["prime", "--plain", "eval", "get", eval_id],
        capture_output=True, text=True, timeout=30,
    )
    try:
        return json.loads(out.stdout).get("model_name", eval_id)
    except Exception:
        return eval_id


def to_trace(s: dict, model: str) -> dict:
    info = s.get("info", {})
    completion = s.get("completion") or []
    content = ""
    if completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content") or ""
    user_prompt = ""
    for p in s.get("prompt", []) or []:
        if isinstance(p, dict) and p.get("role") == "user":
            user_prompt = p.get("content", "")
            break
    return {
        "model": model,
        "perturbation": info.get("perturbation"),
        "scale_mask": info.get("scale_mask"),
        "query_component": info.get("query_component"),
        "answer": s.get("answer"),
        "reward": s.get("reward"),
        "format_reward": s.get("format_reward"),
        "user_prompt": user_prompt,
        "completion": content,
        "completion_len": len(content),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_ids", nargs="+")
    p.add_argument("--out", default="/tmp/bioreasoning_phenotype_data/traces.json")
    args = p.parse_args()

    traces = []
    for eval_id in args.eval_ids:
        model = fetch_model_name(eval_id)
        print(f"Fetching {eval_id} ({model})...", flush=True)
        samples = fetch_samples(eval_id)
        for s in samples:
            traces.append(to_trace(s, model))
        print(f"  {len(samples)} traces", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(traces, indent=2))
    print(f"\nWrote {out_path}: {len(traces)} traces total")


if __name__ == "__main__":
    main()
