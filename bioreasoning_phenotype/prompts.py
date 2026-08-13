"""Prompt templates + answer-space menus for the small-molecule reasoning chain.

Chain shape (skinny: one phenotype prediction per example):
  Step 1: target gene(s) the compound binds
  Step 2: MoA (mechanism of action) — categorical from Drug Repurposing Hub
  Step 3: pathways + signed direction — top-5 Hallmark pathways with up/down
  Step 4: ONE phenotype prediction, varies per example:
            - viability  (continuous LFC)
            - cell_cycle (3-class)
            - stress     (4-class)
            - magnitude  (3-class)

Entry points include full-chain prompts, scaffolding ablations, and standalone
curriculum subtasks:
  smiles_only / from_target / from_moa / from_pathways / phenotype_direct
  target_from_smiles / moa_from_smiles / moa_from_target
  pathways_from_smiles / pathways_from_moa
  phenotype_from_moa / phenotype_from_pathways

v0.6 note: prior versions (v0.4/0.5) required explicit per-step
<X_REASONING>...</X_REASONING> tags before each answer tag. Reverted in v0.6:
modern instruction-tuned models already emit thinking-trace content via the
`reasoning_content` channel before producing the answer, so the extra tag
ceremony added prompt complexity without measurable phenotype lift in the v2
RL run. Reasoning is welcomed but not enforced — the grader only looks at the
answer tags.
"""

# Answer tags (model output)
TARGET_TAG = "TARGET"
MOA_TAG = "MOA"
PATHWAYS_TAG = "PATHWAYS"
VIABILITY_TAG = "VIABILITY"
CYCLE_TAG = "CELL_CYCLE"
STRESS_TAG = "STRESS"
MAGNITUDE_TAG = "MAGNITUDE"


SYSTEM_PROMPT = (
    "You are an expert pharmacologist analyzing a small-molecule compound's "
    "effect on a cancer cell line. Reason through the compound's biology in "
    "sequential steps, then predict the requested phenotype. Wrap each step's "
    "answer in the specified tags so the grader can extract it. Reasoning "
    "between tags is welcomed."
)


# Phenotype answer spaces
CYCLE_CLASSES = ["arrest", "no_effect", "proliferation"]
STRESS_CLASSES = ["none", "apoptosis", "UPR", "DNA_damage"]
MAGNITUDE_CLASSES = ["inert", "moderate", "strong"]


def _format_measurement(value: float | int | str) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _format_duration_h(value: float | int | str) -> str:
    try:
        hours = float(value)
    except (TypeError, ValueError):
        return str(value)
    if hours % 24 == 0:
        days = hours / 24
        return f"{_format_measurement(days)}-day"
    return f"{_format_measurement(hours)}h"


PHENOTYPE_PROMPTS = {
    "viability": (
        f"Predict the viability outcome as a continuous PRISM log2-fold-change "
        f"(LFC) for the PRISM viability assay described above. Negative LFC = cells "
        f"die, positive LFC = cells grow more than control. Typical range: −3.0 to +1.0.\n"
        f"Wrap your answer as a single number in <{VIABILITY_TAG}>...</{VIABILITY_TAG}> "
        f"(e.g. <{VIABILITY_TAG}>-0.45</{VIABILITY_TAG}>)."
    ),
    "cell_cycle": (
        f"Predict the cell-cycle effect in this cell line. Categories:\n"
        f"  - arrest: cells stop dividing (E2F/G2M/MYC programs downregulated)\n"
        f"  - no_effect: cell cycle unchanged\n"
        f"  - proliferation: cells divide more (E2F/G2M/MYC upregulated)\n"
        f"Wrap your answer in <{CYCLE_TAG}>...</{CYCLE_TAG}>."
    ),
    "stress": (
        f"Predict the dominant stress / death pathway activated by this compound. Categories:\n"
        f"  - none: no significant stress pathway activation\n"
        f"  - apoptosis: intrinsic apoptotic pathway activated\n"
        f"  - UPR: unfolded protein response (ER stress) activated\n"
        f"  - DNA_damage: DNA-repair pathway activated\n"
        f"Wrap your answer in <{STRESS_TAG}>...</{STRESS_TAG}>."
    ),
    "magnitude": (
        f"Predict the transcriptional impact magnitude — how strongly the compound "
        f"perturbs gene expression. Categories:\n"
        f"  - inert: little or no transcriptional response\n"
        f"  - moderate: typical perturbation\n"
        f"  - strong: large transcriptional response\n"
        f"Wrap your answer in <{MAGNITUDE_TAG}>...</{MAGNITUDE_TAG}>."
    ),
}


def build_chain_prompt(
    smiles: str,
    cell_line: str,
    dose_um: float,
    timepoint_h: int,
    phenotype: str,
    given_scales: dict[str, str] | None = None,
    skip_chain: bool = False,
    requested_steps: list[str] | None = None,
    prism_dose_um: float | int | str | None = None,
    prism_duration_h: float | int | str | None = None,
) -> str:
    """Build the chain prompt for one skinny example.

    Args:
        smiles: canonical SMILES of the compound
        cell_line: LINCS cell line code (e.g. "MCF7")
        dose_um, timepoint_h: experimental condition
        phenotype: which downstream phenotype to predict, one of
            {"viability", "cell_cycle", "stress", "magnitude"}.
        given_scales: pre-revealed upstream-step answers for entry-point ablations.
            Keys: "target", "moa", "pathways".
        skip_chain: if True, omit all upstream step requests — model is asked
            for the phenotype directly with no chain scaffolding (the
            `phenotype_direct` entry point).
        requested_steps: optional explicit list from
            {"target", "moa", "pathways", "phenotype"}. When set, only those
            outputs are requested and graded, which supports standalone
            curriculum subtasks.
        prism_dose_um, prism_duration_h: optional PRISM viability assay context.
            These are shown only when the requested phenotype is viability.

    Returns:
        The user-content string. The system prompt is separate.
    """
    if given_scales is None:
        given_scales = {}

    requests_phenotype = (
        skip_chain
        or requested_steps is None
        or "phenotype" in requested_steps
    )

    def append_step(step_num: int, step: str) -> None:
        if step == "target":
            sections.append(
                f"Step {step_num} — Target: Identify the primary protein target(s) "
                f"this compound binds. Provide gene symbol(s) separated by '|'.\n"
                f"Wrap your answer in <{TARGET_TAG}>...</{TARGET_TAG}>"
            )
            return
        if step == "moa":
            sections.append(
                f"Step {step_num} — MoA: State the mechanism of action of this compound "
                f"in the canonical pharmacology phrasing (e.g. 'HDAC inhibitor', "
                f"'serotonin receptor antagonist').\n"
                f"Wrap your answer in <{MOA_TAG}>...</{MOA_TAG}>"
            )
            return
        if step == "pathways":
            sections.append(
                f"Step {step_num} — Pathways: Predict the top 5 Hallmark pathways most "
                f"affected by treatment, each annotated with direction ('up' or 'down'). "
                f"Use the HALLMARK_ prefix and ':direction' suffix.\n"
                f"Wrap your answer in <{PATHWAYS_TAG}>HALLMARK_X:up, HALLMARK_Y:down, "
                f"...</{PATHWAYS_TAG}>"
            )
            return
        if step == "phenotype":
            sections.append(f"Step {step_num} — Phenotype: {PHENOTYPE_PROMPTS[phenotype]}")
            return
        raise ValueError(f"Unknown requested step: {step}")

    sections = [
        f"Compound (SMILES): {smiles}",
        f"Cell line: {cell_line}",
        f"LINCS expression assay: {_format_measurement(dose_um)} µM, "
        f"{_format_measurement(timepoint_h)}h treatment "
        f"(used for pathway, cell-cycle, stress, and transcriptional-magnitude labels)",
    ]
    if phenotype == "viability" and requests_phenotype:
        prism_parts = []
        if prism_dose_um is not None:
            prism_parts.append(f"{_format_measurement(prism_dose_um)} µM")
        if prism_duration_h is not None:
            prism_parts.append(f"{_format_duration_h(prism_duration_h)} endpoint")
        if prism_parts:
            prism_context = ", ".join(prism_parts)
        else:
            prism_context = "separate PRISM viability assay"
        sections.append(
            f"PRISM viability assay: {prism_context}; separate assay for the same "
            f"compound and cell line; LFC is relative to DMSO controls"
        )

    if skip_chain:
        # Direct-ask mode: no upstream prefill, no chain instructions, just the phenotype prompt.
        sections.append(f"\n{PHENOTYPE_PROMPTS[phenotype]}")
        return "\n".join(sections)

    if "target" in given_scales:
        sections.append(f"\nKnown protein target(s): {given_scales['target']}")
    if "moa" in given_scales:
        sections.append(f"Known mechanism of action: {given_scales['moa']}")
    if "pathways" in given_scales:
        sections.append(f"Observed pathway changes: {given_scales['pathways']}")

    if requested_steps is not None:
        sections.append(
            "\nAnswer the requested task. Reasoning is welcome, but include only "
            "the requested answer tag(s).\n"
        )
        for i, step in enumerate(requested_steps, start=1):
            append_step(i, step)
        return "\n".join(sections)

    sections.append("\nReason through the following steps:\n")

    step_num = 1
    if "target" not in given_scales:
        append_step(step_num, "target")
        step_num += 1

    if "moa" not in given_scales:
        append_step(step_num, "moa")
        step_num += 1

    if "pathways" not in given_scales:
        append_step(step_num, "pathways")
        step_num += 1

    append_step(step_num, "phenotype")
    return "\n".join(sections)


# Entry-point ablation templates.
#   "given"      = upstream steps pre-filled into the prompt (and removed from grading)
#   "skip_chain" = if True, omit upstream step REQUESTS entirely; model is asked
#                  directly for the phenotype (head-to-head test of chain reasoning).
#   "asks"       = optional explicit standalone outputs to request/grade.
ENTRY_POINTS = {
    "smiles_only":      {"given": [], "skip_chain": False},
    "from_target":      {"given": ["target"], "skip_chain": False},
    "from_moa":         {"given": ["target", "moa"], "skip_chain": False},
    "from_pathways":    {"given": ["target", "moa", "pathways"], "skip_chain": False},
    "phenotype_direct": {"given": [], "skip_chain": True},
    "target_from_smiles":      {"given": [], "asks": ["target"]},
    "moa_from_smiles":         {"given": [], "asks": ["moa"]},
    "moa_from_target":         {"given": ["target"], "asks": ["moa"]},
    "pathways_from_smiles":    {"given": [], "asks": ["pathways"]},
    "pathways_from_moa":       {"given": ["target", "moa"], "asks": ["pathways"]},
    "phenotype_from_moa":      {"given": ["target", "moa"], "asks": ["phenotype"]},
    "phenotype_from_pathways": {"given": ["pathways"], "asks": ["phenotype"]},
}

# All four downstream phenotypes (skinny: one per example)
PHENOTYPES = ["viability", "cell_cycle", "stress", "magnitude"]
