"""load_environment entry point for the small-molecule chain env."""

from typing import Any

import verifiers as vf

from bioreasoning_phenotype.env import load_examples, make_rubric
from bioreasoning_phenotype.hallmarks import describe_hallmark
from bioreasoning_phenotype.tools import identify_compound


def load_environment(
    entry_points: list[str] | None = None,
    phenotypes: list[str] | None = None,
    cell_lines: list[str] | None = None,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    reward_weights: dict[str, float] | None = None,
    tools: bool = False,
    hallmark_tools: bool = False,
    max_tool_turns: int = 5,
    **kwargs: Any,
) -> vf.Environment:
    """Build the small-molecule reasoning-chain env.

    Skinny chain: each example asks for target → MoA → pathways+direction → ONE
    phenotype prediction. The phenotype varies per example.

    Args:
        entry_points: subset of full-chain, ablation, and standalone curriculum
            tasks. Common values include {"smiles_only", "from_target",
            "from_moa", "from_pathways", "phenotype_direct",
            "target_from_smiles", "moa_from_target", "pathways_from_smiles",
            "pathways_from_moa", "phenotype_from_pathways"}. Default None = all
            available entry points.
        phenotypes: subset of {"viability", "cell_cycle", "stress", "magnitude"}.
            Default None = all four.
        cell_lines: subset of the 6 LINCS core lines
            {A375, A549, HEPG2, HT29, MCF7, PC3}. Default None = all.
        num_train_examples / num_eval_examples: -1 for all; else downsample.
        reward_weights: aggregate-reward weights for the chain steps. Keys:
            target / moa / pathways / phenotype. Default 0.15/0.15/0.25/0.45.
        tools: if True, expose `identify_compound(smiles)` so the model can
            look up compound identity and chemistry metadata during reasoning.
        hallmark_tools: if True, expose `describe_hallmark(pathway)` so the
            model can look up Hallmark ontology metadata during reasoning.
        max_tool_turns: max number of tool-call rounds when any tool is enabled.
        kwargs: absorbed for forward-compat with test harnesses.
    """
    train_ds, eval_ds = load_examples(
        entry_points=entry_points,
        phenotypes=phenotypes,
        cell_lines=cell_lines,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
    )
    rubric = make_rubric(weights=reward_weights)

    tool_fns = []
    if tools:
        tool_fns.append(identify_compound)
    if hallmark_tools:
        tool_fns.append(describe_hallmark)

    if tool_fns:
        return vf.ToolEnv(
            dataset=train_ds,
            eval_dataset=eval_ds,
            rubric=rubric,
            tools=tool_fns,
            max_turns=max_tool_turns,
        )

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        rubric=rubric,
    )
