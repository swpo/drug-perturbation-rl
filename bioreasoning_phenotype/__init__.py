"""Small-molecule phenotype reasoning-chain env.

Given a SMILES + cell line + assay context, the model reasons through:
target gene(s) -> MoA -> signed Hallmark pathways -> one downstream phenotype.
"""

from bioreasoning_phenotype.main import load_environment

__all__ = ["load_environment"]
