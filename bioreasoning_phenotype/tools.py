"""Retrieval tool: identify_compound(smiles).

Given a SMILES string, returns structured metadata that helps the model
identify the compound and reason about its chemistry — without leaking
the graded fields (target, MoA, pathways, phenotype).

Returns:
- `exact_match`: drug name if the canonical InChIKey is in our compound set
- `descriptors`: physicochemical properties (MW, logP, TPSA, etc.)
- `scaffold`: Bemis-Murcko scaffold SMILES (the chemotype)
- `nearest_neighbors`: top-K most similar compounds from our 1046-compound set
  by Morgan-fingerprint Tanimoto, each with name + similarity score

No external IDs (pubchem_cid, inchi_key) are returned to the model — those would
just be hallucination fodder. The drug name + similar compounds are the high-
signal pieces; descriptors and scaffold give chemistry context for cases where
the model needs to reason from structure rather than retrieve by name.
"""
from pathlib import Path
from typing import Any
import functools

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, Lipinski, MolFromSmiles
from rdkit.Chem.Scaffolds import MurckoScaffold

# Suppress noisy rdkit warnings
RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path(__file__).parent / "data"


@functools.lru_cache(maxsize=1)
def _load_compound_db() -> tuple[pd.DataFrame, list]:
    """Load compound table + pre-compute Morgan fingerprints."""
    # First try the env-bundled compound table; fall back to /tmp cache location
    candidates = [
        DATA_DIR / "compound_table.parquet",
        Path("/tmp/bioreasoning_smallmol_data/compound_table.parquet"),
    ]
    for p in candidates:
        if p.exists():
            df = pd.read_parquet(p)
            break
    else:
        raise FileNotFoundError(
            "compound_table.parquet not found. Run scripts/build_compound_table.py."
        )

    # Compute Morgan fingerprints for all compounds
    fps: list = []
    for smi in df["smiles"]:
        mol = MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    return df, fps


def _descriptors(mol) -> dict[str, float]:
    """Compute basic physicochemical descriptors via rdkit."""
    return {
        "molecular_weight": round(Descriptors.MolWt(mol), 1),
        "logp": round(Descriptors.MolLogP(mol), 2),
        "tpsa": round(Descriptors.TPSA(mol), 1),
        "num_h_bond_donors": Lipinski.NumHDonors(mol),
        "num_h_bond_acceptors": Lipinski.NumHAcceptors(mol),
        "num_rings": Lipinski.RingCount(mol),
        "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
        "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
    }


def _murcko_scaffold(mol) -> str:
    """Return the Bemis-Murcko scaffold as a canonical SMILES."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True)


def identify_compound(smiles: str, top_k: int = 5) -> dict[str, Any]:
    """Look up structured metadata for a compound by its SMILES string.

    Args:
        smiles: SMILES of the query compound
        top_k: how many nearest neighbors to return (default 5)

    Returns:
        Dict with `exact_match`, `descriptors`, `scaffold`, `nearest_neighbors`.
        If SMILES is unparseable, returns {"error": "invalid SMILES"}.
    """
    mol = MolFromSmiles(smiles)
    if mol is None:
        return {"error": "invalid SMILES"}

    df, fps = _load_compound_db()

    # Exact match via canonical InChIKey
    try:
        query_inchi_key = Chem.MolToInchiKey(mol)
    except Exception:
        query_inchi_key = None

    exact_match = None
    if query_inchi_key:
        hit = df[df["inchi_key"] == query_inchi_key]
        if not hit.empty:
            exact_match = {"name": hit.iloc[0]["pert_iname"]}

    # Compute fingerprint and Tanimoto similarity to all compounds
    query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    sims = []
    for i, fp in enumerate(fps):
        if fp is None:
            continue
        s = DataStructs.TanimotoSimilarity(query_fp, fp)
        sims.append((s, i))

    sims.sort(reverse=True)

    # Skip neighbor #1 if it's the exact match (Tanimoto 1.0 to itself)
    neighbors = []
    for s, i in sims:
        if exact_match is not None and df.iloc[i]["inchi_key"] == query_inchi_key:
            continue
        neighbors.append({
            "name": df.iloc[i]["pert_iname"],
            "similarity": round(s, 3),
        })
        if len(neighbors) >= top_k:
            break

    return {
        "exact_match": exact_match,
        "descriptors": _descriptors(mol),
        "scaffold": _murcko_scaffold(mol),
        "nearest_neighbors": neighbors,
    }


if __name__ == "__main__":
    # Quick smoke test
    import json
    test_smiles = [
        ("CC(C)NCC(O)COc1cccc2ccccc12", "propranolol"),  # exact known
        ("CC(CCc1ccc(O)cc1)NCCc1ccc(O)c(O)c1", "dobutamine"),  # exact known
        ("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "gefitinib"),  # exact known
        ("CC(C)Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "novel-ish (adenosine analog)"),
    ]
    for smi, label in test_smiles:
        print(f"=== {label} ===")
        print(f"SMILES: {smi}")
        result = identify_compound(smi)
        print(json.dumps(result, indent=2))
        print()
