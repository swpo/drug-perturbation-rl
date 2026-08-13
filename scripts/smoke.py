"""Smoke test for the small-molecule phenotype chain env.

No API calls. This checks the packaged data, prompt assembly, tools, and core
rubric helpers used by local eval / Prime eval.
"""

from __future__ import annotations

import sys

import pandas as pd

from bioreasoning_phenotype import load_environment
from bioreasoning_phenotype.env import (
    DATA_PATH,
    _extract_tag,
    _score_class,
    _score_moa,
    _score_pathway_direction_accuracy,
    _score_pathway_name_f1,
    _score_pathway_name_validity,
    _score_pathways,
    _score_target,
    _score_viability,
)
from bioreasoning_phenotype.hallmarks import describe_hallmark, hallmark_names
from bioreasoning_phenotype.prompts import (
    CYCLE_CLASSES,
    CYCLE_TAG,
    MOA_TAG,
    PATHWAYS_TAG,
    TARGET_TAG,
    VIABILITY_TAG,
    build_chain_prompt,
)


def _check(condition: bool, message: str) -> int:
    if condition:
        return 0
    print(f"  FAIL: {message}")
    return 1


def check_env_loads() -> int:
    fail = 0
    env = load_environment(entry_points=["smiles_only"], num_eval_examples=8)
    fail += _check(len(env.eval_dataset) == 8, "smiles_only eval downsample did not load 8 rows")
    row = env.eval_dataset[0]
    fail += _check(row["info"]["entry_point"] == "smiles_only", "entry_point info missing")
    fail += _check(row["info"]["phenotype"] in {"viability", "cell_cycle", "stress", "magnitude"},
                   "phenotype info missing")

    tool_env = load_environment(
        entry_points=["smiles_only"],
        num_eval_examples=1,
        tools=True,
        hallmark_tools=True,
    )
    fail += _check("ToolEnv" in type(tool_env).__name__, "tools did not create a ToolEnv")

    target_env = load_environment(entry_points=["target_from_smiles"], num_eval_examples=4)
    fail += _check(len(target_env.eval_dataset) == 4,
                   "target_from_smiles eval downsample did not load 4 rows")
    target_row = target_env.eval_dataset[0]
    fail += _check(target_row["info"]["phenotype"] == "upstream",
                   "upstream-only curriculum task should be labeled phenotype=upstream")
    fail += _check("target" in target_row["answer"] and "phenotype" not in target_row["answer"],
                   "target_from_smiles answer should grade target only")
    return fail


def check_prompt_and_packaged_data() -> int:
    fail = 0
    prompt = build_chain_prompt(
        smiles="CCO",
        cell_line="MCF7",
        dose_um=3.16227766,
        timepoint_h=6,
        phenotype="viability",
        prism_dose_um=2.5,
        prism_duration_h=120,
    )
    fail += _check("LINCS expression assay: 3.16228 µM, 6h treatment" in prompt,
                   "dynamic LINCS dose/time formatting missing from generated prompt")
    fail += _check("PRISM viability assay: 2.5 µM, 5-day endpoint" in prompt,
                   "PRISM viability protocol context missing from generated prompt")
    fail += _check("at 10 µM, 24h" not in prompt,
                   "viability prompt still contains hardcoded 10 µM / 24h condition")
    fail += _check("under this treatment condition" not in prompt,
                   "viability prompt still conflates LINCS and PRISM contexts")
    fail += _check("Use only these canonical Hallmark pathway names:" not in prompt,
                   "Hallmark canonical-name menu should not be in v0.10 prompt")
    fail += _check("Use the HALLMARK_ prefix and ':direction' suffix." in prompt,
                   "v0.10 prompt should preserve hypo3-style pathway instruction")

    moa_prompt = build_chain_prompt(
        smiles="CCO",
        cell_line="MCF7",
        dose_um=3.16227766,
        timepoint_h=6,
        phenotype="viability",
        given_scales={"target": "ESR1"},
        requested_steps=["moa"],
    )
    fail += _check("Known protein target(s): ESR1" in moa_prompt,
                   "standalone MoA prompt should include known target")
    fail += _check("<MOA>...</MOA>" in moa_prompt,
                   "standalone MoA prompt should request MOA tag")
    fail += _check("<TARGET>...</TARGET>" not in moa_prompt,
                   "standalone MoA prompt should not request target tag")

    df = pd.read_parquet(DATA_PATH)
    fail += _check(len(df) > 100_000, f"packaged examples unexpectedly small: {len(df)}")
    fail += _check("target_from_smiles" in set(df["entry_point"]),
                   "packaged examples missing target_from_smiles curriculum rows")
    fail += _check("phenotype_from_pathways" in set(df["entry_point"]),
                   "packaged examples missing phenotype_from_pathways curriculum rows")
    prompts = df["user_prompt"].astype(str)
    fail += _check(not prompts.str.contains("at 10 µM, 24h", regex=False).any(),
                   "packaged prompts contain stale hardcoded viability condition")
    fail += _check(prompts.str.contains("6h treatment", regex=False).any(),
                   "packaged prompts do not include any 6h L1000 conditions")
    fail += _check(prompts.str.contains("LINCS expression assay:", regex=False).any(),
                   "packaged prompts do not include explicit LINCS assay context")
    fail += _check(prompts.str.contains("PRISM viability assay:", regex=False).any(),
                   "packaged prompts do not include explicit PRISM viability context")
    fail += _check(not prompts.str.contains("under this treatment condition", regex=False).any(),
                   "packaged prompts still conflate LINCS and PRISM contexts")
    fail += _check(not prompts.str.contains("Use only these canonical Hallmark pathway names:", regex=False).any(),
                   "packaged prompts unexpectedly include Hallmark menu")
    fail += _check(prompts.str.contains("Use the HALLMARK_ prefix and ':direction' suffix.", regex=False).any(),
                   "packaged prompts do not include hypo3-style pathway instruction")
    return fail


def check_hallmark_tool() -> int:
    fail = 0
    names = hallmark_names()
    fail += _check(len(names) == 50, f"expected 50 Hallmark gene sets, got {len(names)}")
    fail += _check("HALLMARK_ANDROGEN_RESPONSE" in names,
                   "canonical androgen Hallmark name missing")

    response = describe_hallmark("androgen signaling", max_genes=3)
    fail += _check(response.get("canonical_name") == "HALLMARK_ANDROGEN_RESPONSE",
                   f"androgen alias did not canonicalize: {response}")
    fail += _check(response.get("gene_count", 0) > 3, "Hallmark gene_count missing")
    fail += _check(len(response.get("member_genes_sample", [])) == 3,
                   "max_genes was not honored")
    fail += _check("pathway_scores" not in response and "observed_direction" not in response,
                   "describe_hallmark leaked perturbation outcomes")
    return fail


def check_rubric_helpers() -> int:
    fail = 0
    text = (
        f"<{TARGET_TAG}>EGFR | ERBB2</{TARGET_TAG}> "
        f"<{MOA_TAG}>EGFR inhibitor</{MOA_TAG}> "
        f"<{PATHWAYS_TAG}>HALLMARK_E2F_TARGETS:down, HALLMARK_APOPTOSIS:up</{PATHWAYS_TAG}> "
        f"<{VIABILITY_TAG}>-0.45</{VIABILITY_TAG}> "
        f"<{CYCLE_TAG}>arrest</{CYCLE_TAG}>"
    )
    gt_pairs = [("HALLMARK_E2F_TARGETS", "down"), ("HALLMARK_APOPTOSIS", "up")]

    fail += _check(_score_target(_extract_tag(text, TARGET_TAG), ["EGFR", "ERBB2"]) == 1.0,
                   "target F1 perfect case failed")
    fail += _check(_score_moa(_extract_tag(text, MOA_TAG), "EGFR inhibitor") == 1.0,
                   "MoA exact-match perfect case failed")
    fail += _check(_score_pathways(_extract_tag(text, PATHWAYS_TAG), gt_pairs) == 1.0,
                   "signed pathway F1 perfect case failed")
    fail += _check(_score_pathway_name_validity(_extract_tag(text, PATHWAYS_TAG)) == 1.0,
                   "pathway name validity perfect case failed")
    fail += _check(_score_pathway_name_f1(_extract_tag(text, PATHWAYS_TAG), gt_pairs) == 1.0,
                   "pathway name F1 perfect case failed")
    fail += _check(_score_pathway_direction_accuracy(_extract_tag(text, PATHWAYS_TAG), gt_pairs) == 1.0,
                   "pathway direction perfect case failed")
    fail += _check(_score_viability(_extract_tag(text, VIABILITY_TAG), -0.50) == 1.0,
                   "viability tolerance perfect case failed")
    fail += _check(_score_class(_extract_tag(text, CYCLE_TAG), "arrest", CYCLE_CLASSES) == 1.0,
                   "cell-cycle class perfect case failed")

    bad_pathways = "HALLMARK_ANDROGEN_SIGNALING:down, HALLMARK_E2F_TARGETS:up"
    fail += _check(_score_pathway_name_validity(bad_pathways) == 0.5,
                   "invalid Hallmark synonym should lower name validity")
    fail += _check(_score_pathway_direction_accuracy(bad_pathways, gt_pairs) == 0.0,
                   "wrong direction on shared pathway should score 0 direction accuracy")
    return fail


def main() -> int:
    print("--- Phenotype chain env smoke test ---\n")
    checks = [
        ("Env loads + tool flags", check_env_loads),
        ("Prompt/data sanity", check_prompt_and_packaged_data),
        ("Hallmark ontology tool", check_hallmark_tool),
        ("Rubric helper scoring", check_rubric_helpers),
    ]

    total_fail = 0
    for label, fn in checks:
        failures = fn()
        total_fail += failures
        print(f"{label}: {'OK' if failures == 0 else 'FAIL'}")

    print()
    if total_fail == 0:
        print("PASS")
        return 0
    print(f"FAIL: {total_fail} check(s) failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
