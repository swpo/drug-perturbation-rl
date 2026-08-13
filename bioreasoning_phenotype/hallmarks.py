"""MSigDB Hallmark ontology helpers.

This module exposes answer-vocabulary support for the pathway step. It contains
only pathway names, URLs, aliases, and gene membership. It deliberately does
not expose compound-specific L1000 scores or observed pathway labels.
"""

from __future__ import annotations

import difflib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


DATA_PATH = Path(__file__).parent / "data" / "hallmarks.json"

HALLMARK_ALIASES = {
    "ANDROGEN_RECEPTOR": "HALLMARK_ANDROGEN_RESPONSE",
    "ANDROGEN_RECEPTOR_SIGNALING": "HALLMARK_ANDROGEN_RESPONSE",
    "ANDROGEN_RESPONSE": "HALLMARK_ANDROGEN_RESPONSE",
    "ANDROGEN_SIGNALING": "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_ANDROGEN_RECEPTOR": "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_ANDROGEN_RECEPTOR_SIGNALING": "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_ANDROGEN_SIGNALING": "HALLMARK_ANDROGEN_RESPONSE",
    "HALLMARK_KRAS_SIGNALING_DOWN": "HALLMARK_KRAS_SIGNALING_DN",
    "HALLMARK_KRAS_SIGNALING_UPPER": "HALLMARK_KRAS_SIGNALING_UP",
    "HALLMARK_MYC_V1": "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_V1_TARGETS": "HALLMARK_MYC_TARGETS_V1",
    "HALLMARK_MYC_V2": "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_MYC_V2_TARGETS": "HALLMARK_MYC_TARGETS_V2",
    "HALLMARK_TGFBETA": "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_TGFBETA_SIGNALING": "HALLMARK_TGF_BETA_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKAPPA_B": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_TNFA_VIA_NFKB": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_WNT_BETA_CATENIN_PATHWAY": "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "KRAS_SIGNALING_DN": "HALLMARK_KRAS_SIGNALING_DN",
    "KRAS_SIGNALING_DOWN": "HALLMARK_KRAS_SIGNALING_DN",
    "KRAS_SIGNALING_UP": "HALLMARK_KRAS_SIGNALING_UP",
    "MYC_TARGETS": "HALLMARK_MYC_TARGETS_V1",
    "MYC_TARGETS_V1": "HALLMARK_MYC_TARGETS_V1",
    "MYC_TARGETS_V2": "HALLMARK_MYC_TARGETS_V2",
    "TGF_BETA": "HALLMARK_TGF_BETA_SIGNALING",
    "TGF_BETA_SIGNALING": "HALLMARK_TGF_BETA_SIGNALING",
    "TGFBETA": "HALLMARK_TGF_BETA_SIGNALING",
    "TGFBETA_SIGNALING": "HALLMARK_TGF_BETA_SIGNALING",
    "TNFA_SIGNALING_VIA_NFKAPPA_B": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "TNFA_VIA_NFKB": "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "WNT_BETA_CATENIN": "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
    "WNT_BETA_CATENIN_PATHWAY": "HALLMARK_WNT_BETA_CATENIN_SIGNALING",
}


def _normalize_key(text: str) -> str:
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


@lru_cache(maxsize=1)
def load_hallmarks() -> list[dict[str, Any]]:
    """Load packaged Hallmark metadata."""
    with open(DATA_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def hallmark_by_name() -> dict[str, dict[str, Any]]:
    """Return Hallmark metadata keyed by canonical name."""
    return {row["name"].upper(): row for row in load_hallmarks()}


def hallmark_names() -> list[str]:
    """Canonical MSigDB Hallmark names used by this env."""
    return sorted(hallmark_by_name())


def format_hallmark_menu(names_per_line: int = 5) -> str:
    """Format the canonical Hallmark answer vocabulary for prompts."""
    names = hallmark_names()
    lines = []
    for i in range(0, len(names), names_per_line):
        lines.append(", ".join(names[i:i + names_per_line]))
    return "\n".join(lines)


def canonicalize_hallmark(pathway: str | None) -> str | None:
    """Map a Hallmark-ish name or alias to the canonical name if possible."""
    if not pathway:
        return None
    raw = pathway.strip()
    raw = re.sub(r"^<PATHWAYS>", "", raw, flags=re.IGNORECASE).strip()
    raw = raw.split(":", 1)[0].strip()
    key = _normalize_key(raw)

    names = hallmark_by_name()
    if key in names:
        return key
    if not key.startswith("HALLMARK_") and f"HALLMARK_{key}" in names:
        return f"HALLMARK_{key}"
    if key in HALLMARK_ALIASES:
        return HALLMARK_ALIASES[key]
    if key.startswith("HALLMARK_"):
        no_prefix = key.removeprefix("HALLMARK_")
        if no_prefix in HALLMARK_ALIASES:
            return HALLMARK_ALIASES[no_prefix]
    return None


def _aliases_for(canonical_name: str) -> list[str]:
    aliases = []
    for alias, canonical in HALLMARK_ALIASES.items():
        if canonical == canonical_name:
            aliases.append(alias)
    return sorted(aliases)


def describe_hallmark(pathway: str, max_genes: int = 25) -> dict[str, Any]:
    """Describe a canonical MSigDB Hallmark pathway.

    Args:
        pathway: Hallmark name or common alias, e.g. "androgen signaling" or
            "HALLMARK_ANDROGEN_RESPONSE".
        max_genes: Maximum number of member genes to include. Defaults to 25
            and is capped at 100.

    Returns:
        Ontology metadata: canonical name, MSigDB URL, gene count, a sample of
        member genes, and known aliases. Does not return compound-specific
        pathway scores or observed perturbation outcomes.
    """
    canonical = canonicalize_hallmark(pathway)
    if canonical is None:
        query = _normalize_key(pathway or "")
        choices = hallmark_names() + sorted(HALLMARK_ALIASES)
        suggestions = difflib.get_close_matches(query, choices, n=5, cutoff=0.55)
        return {
            "error": "unknown Hallmark pathway",
            "query": pathway,
            "suggestions": suggestions,
        }

    try:
        max_genes = int(max_genes)
    except (TypeError, ValueError):
        max_genes = 25
    max_genes = max(0, min(max_genes, 100))
    row = hallmark_by_name()[canonical]
    genes = list(row["genes"])
    return {
        "canonical_name": canonical,
        "matched_from": pathway,
        "msigdb_url": row["url"],
        "gene_count": len(genes),
        "member_genes_sample": genes[:max_genes],
        "aliases": _aliases_for(canonical),
    }
