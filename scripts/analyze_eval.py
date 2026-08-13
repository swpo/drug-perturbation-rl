"""Pull rollout-level samples from one or more Prime evaluations and produce
ablation matrices.

For each (model, scale_mask, query_component) cell, reports:
  - pass_at_1, pass_at_k  — accuracy (mean reward / any-correct across rollouts)
  - macro_f1              — per-class F1 averaged, immune to class imbalance
  - binary_f1 (cycle only) — cycle-perturbing-vs-balanced F1
  - format_compliance, n_examples
Also prints analytical baselines (always-majority, uniform-random) so accuracy
numbers can be sanity-checked.

Usage:
    uv run python scripts/analyze_eval.py EVAL_ID1 EVAL_ID2 ...
"""

import argparse
import json
import re
import subprocess

import pandas as pd

_ANSWER_RE = re.compile(r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>", re.IGNORECASE | re.DOTALL)


# Class definitions for F1 computation
QUERY_CLASSES = {
    "cycle_phase": ["balanced", "s_dominant", "g1_dominant", "g2m_dominant"],
    "essentiality_bucket": ["neutral", "moderate_essential", "essential", "growth_advantage"],
    # moa_class is open-set; skip categorical F1
}

# For cycle_phase, the natural binary collapse:
CYCLE_PERTURBING_CLASSES = {"s_dominant", "g1_dominant", "g2m_dominant"}


def extract_pred(completion_text: str) -> str | None:
    if not isinstance(completion_text, str):
        return None
    m = _ANSWER_RE.search(completion_text)
    return m.group(1).strip().lower() if m else None


def per_class_f1(preds: list[str], gts: list[str], classes: list[str]) -> dict:
    """Return per-class precision/recall/F1 + macro F1."""
    out = {}
    f1s = []
    for c in classes:
        tp = sum(1 for p, g in zip(preds, gts) if p == c and g == c)
        fp = sum(1 for p, g in zip(preds, gts) if p == c and g != c)
        fn = sum(1 for p, g in zip(preds, gts) if p != c and g == c)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[c] = {"prec": prec, "rec": rec, "f1": f1, "n_gt": sum(1 for g in gts if g == c)}
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s) if f1s else 0.0
    return out


def cycle_binary_f1(preds: list[str], gts: list[str]) -> dict:
    """Binary cycle-perturbing-vs-balanced F1 (positive class = any *_dominant)."""
    bin_pred = [p in CYCLE_PERTURBING_CLASSES for p in preds]
    bin_gt = [g in CYCLE_PERTURBING_CLASSES for g in gts]
    tp = sum(1 for p, g in zip(bin_pred, bin_gt) if p and g)
    fp = sum(1 for p, g in zip(bin_pred, bin_gt) if p and not g)
    fn = sum(1 for p, g in zip(bin_pred, bin_gt) if not p and g)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"prec": prec, "rec": rec, "f1": f1}


def analytical_baselines(gts: list[str], classes: list[str]) -> dict:
    """Compute always-majority and uniform-random baselines analytically."""
    n = len(gts)
    if n == 0:
        return {"always_majority_acc": 0.0, "uniform_random_acc": 0.0}
    # Class distribution
    dist = {c: sum(1 for g in gts if g == c) / n for c in classes}
    majority_class = max(dist, key=dist.get)
    always_acc = dist[majority_class]
    # Uniform random: expected accuracy = avg over classes of P(c_true == c) * 1/|C|
    uniform_acc = sum(dist.values()) / len(classes)
    return {
        "always_majority_class": majority_class,
        "always_majority_acc": always_acc,
        "uniform_random_acc": uniform_acc,
        "class_dist": dist,
    }


def fetch_samples(eval_id: str, page_size: int = 200) -> list[dict]:
    """Page through all samples for an eval."""
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
        try:
            # strict=False allows unescaped control characters in strings,
            # which Prime CLI's --output json emits in prompt content fields.
            data = json.loads(out.stdout, strict=False)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"bad JSON for {eval_id} page {page}: {e}\n{out.stdout[:500]}")
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
        d = json.loads(out.stdout)
        return d.get("model_name", eval_id)
    except json.JSONDecodeError:
        return eval_id


def samples_to_df(samples: list[dict], model_name: str) -> pd.DataFrame:
    rows = []
    for s in samples:
        info = s.get("info", {})
        # completion field may be None or [] (e.g. empty completions on errors)
        completion = s.get("completion") or []
        content = ""
        if completion:
            last = completion[-1]
            if isinstance(last, dict):
                content = last.get("content") or ""
        rows.append({
            "model": model_name,
            "example_id": s.get("example_id"),
            "sample_id": s.get("sample_id"),
            "perturbation": info.get("perturbation"),
            "scale_mask": info.get("scale_mask"),
            "query_component": info.get("query_component"),
            "reward": s.get("reward"),
            "format_reward": s.get("format_reward"),
            "answer": (s.get("answer") or "").lower(),
            "pred": extract_pred(content),
            "completion_len": len(content),
        })
    return pd.DataFrame(rows)


def compute_ablation_matrix(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """For each model, compute per-(mask, query) cell metrics with multiple metrics."""
    out = {}
    for model, mdf in df.groupby("model"):
        grp = mdf.groupby(["scale_mask", "query_component", "perturbation"])
        per_example = grp.agg(
            n_rollouts=("reward", "count"),
            mean_reward=("reward", "mean"),
            any_correct=("reward", "max"),
            mean_format=("format_reward", "mean"),
        ).reset_index()

        cells = []
        for (mask, query), cdf in mdf.groupby(["scale_mask", "query_component"]):
            preds = cdf["pred"].fillna("null").tolist()
            gts = cdf["answer"].tolist()

            row = {
                "scale_mask": mask,
                "query_component": query,
                "n_examples": cdf["perturbation"].nunique(),
                "n_rollouts": len(cdf),
                "pass_at_1": cdf["reward"].mean(),
                "format_compliance": cdf["format_reward"].mean(),
            }
            # pass_at_k from per_example
            sub_pe = per_example[(per_example["scale_mask"] == mask) & (per_example["query_component"] == query)]
            row["pass_at_k"] = sub_pe["any_correct"].mean() if len(sub_pe) else 0.0
            row["headroom"] = row["pass_at_k"] - row["pass_at_1"]

            # Macro F1 (if class set known)
            classes = QUERY_CLASSES.get(query)
            if classes:
                f1_info = per_class_f1(preds, gts, classes)
                row["macro_f1"] = f1_info["macro_f1"]
                # Per-class recall for non-majority classes
                for c in classes:
                    row[f"recall_{c}"] = f1_info[c]["rec"]
                # Analytical baselines
                base = analytical_baselines(gts, classes)
                row["always_majority_acc"] = base["always_majority_acc"]
                row["uniform_random_acc"] = base["uniform_random_acc"]

            # Cycle-binary F1 (only meaningful for cycle_phase)
            if query == "cycle_phase":
                bin_info = cycle_binary_f1(preds, gts)
                row["cycle_binary_f1"] = bin_info["f1"]
                row["cycle_binary_prec"] = bin_info["prec"]
                row["cycle_binary_rec"] = bin_info["rec"]

            cells.append(row)
        out[model] = pd.DataFrame(cells).sort_values(["query_component", "scale_mask"]).reset_index(drop=True)
    return out


def print_per_model(cell_df: pd.DataFrame, model: str):
    print(f"\n=== {model} ===")
    order = ["v0", "v2", "v3", "v4", "v2+3", "v2+4", "v2+3+4"]
    metrics_to_show = [
        "pass_at_1", "macro_f1", "cycle_binary_f1",
        "cycle_binary_rec", "cycle_binary_prec",
        "pass_at_k", "headroom", "format_compliance",
    ]
    for metric in metrics_to_show:
        if metric not in cell_df.columns:
            continue
        pivot = cell_df.pivot(index="scale_mask", columns="query_component", values=metric)
        pivot = pivot.reindex([m for m in order if m in pivot.index])
        if pivot.dropna(how="all").empty:
            continue
        print(f"\n{metric}:")
        print(pivot.round(3).to_string())

    # Baselines (per query)
    baseline_rows = cell_df.drop_duplicates("query_component")[
        ["query_component", "always_majority_acc", "uniform_random_acc"]
    ].dropna()
    if not baseline_rows.empty:
        print(f"\nAnalytical baselines for this model's test slice:")
        print(baseline_rows.round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eval_ids", nargs="+", help="Eval IDs to analyze")
    p.add_argument("-o", "--output", default=None, help="Optional CSV output path")
    args = p.parse_args()

    all_dfs = []
    for eval_id in args.eval_ids:
        print(f"Fetching samples for {eval_id}...", flush=True)
        model = fetch_model_name(eval_id)
        samples = fetch_samples(eval_id)
        print(f"  {model}: {len(samples)} samples", flush=True)
        all_dfs.append(samples_to_df(samples, model))

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nTotal rollouts: {len(df)}")
    print(f"Models: {sorted(df.model.unique())}")

    matrices = compute_ablation_matrix(df)
    for model, cell_df in matrices.items():
        print_per_model(cell_df, model)

    # Cross-model comparison
    if len(matrices) > 1:
        print(f"\n\n=== Cross-model comparison (pass_at_1) ===")
        combined = pd.concat([
            cdf.assign(model=m) for m, cdf in matrices.items()
        ])
        for q in sorted(combined.query_component.unique()):
            sub = combined[combined.query_component == q]
            pivot = sub.pivot(index="scale_mask", columns="model", values="pass_at_1")
            order = ["v0", "v2", "v3", "v4", "v2+3", "v2+4", "v2+3+4"]
            pivot = pivot.reindex([m for m in order if m in pivot.index])
            print(f"\nquery_component = {q}")
            print(pivot.round(3).to_string())

    if args.output:
        all_rows = []
        for m, cdf in matrices.items():
            cdf2 = cdf.copy()
            cdf2["model"] = m
            all_rows.append(cdf2)
        out_df = pd.concat(all_rows, ignore_index=True)
        out_df.to_csv(args.output, index=False)
        print(f"\nWrote {args.output}: {out_df.shape}")


if __name__ == "__main__":
    main()
