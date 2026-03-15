# src/features/interaction.py
#
# Interaction feature block — FIXED design.
#
# Previous bug: joint PCA on [ESM_640d || ECFP_1024d]
#   ESM: dense floats, semantic meaning, range ~[-3,3]
#   ECFP: sparse binary {0,1}
#   Joint PCA on incompatible spaces → garbage projections
#
# Fixed design: SEPARATE PCA per modality, THEN interact
#   Protein:  ESM_multi [N, n_layers*480] → PCA → [N, 128]
#   Ligand:   ECFP+MACCS+AtomPair+Torsion [N, 5287] → PCA → [N, 128]
#   Interact: [P⊙L, |P-L|] → [N, 256]
#   Final:    [P_proj, L_proj, P⊙L, |P-L|] → [N, 512]
#
# Biological rationale for hadamard + diff:
#   P⊙L captures co-activation: which latent dimensions
#        are simultaneously high in both protein and ligand
#   |P-L| captures complementarity: which dimensions differ
#        (shape complementarity is about what doesn't match)

import numpy as np
from sklearn.decomposition import PCA
import joblib
from pathlib import Path


def build_interaction_features(prot_emb, lig_concat,
                                dim=128,
                                prot_pca=None, lig_pca=None,
                                fit=False):
    """
    Args:
        prot_emb:   [N, prot_dim]  — ESM multi+attention concatenated
        lig_concat: [N, lig_dim]   — ECFP+MACCS+AtomPair+Torsion concatenated
                                     (NOT including scaled RDKit phys)
        dim:        projection dimension
        fit:        if True, fit PCA on this data

    Returns:
        interaction [N, 4*dim]
        prot_pca, lig_pca
    """
    if fit:
        prot_pca = PCA(n_components=min(dim, prot_emb.shape[1]),
                       random_state=42)
        lig_pca  = PCA(n_components=min(dim, lig_concat.shape[1]),
                       random_state=42)
        p_proj = prot_pca.fit_transform(prot_emb)
        l_proj = lig_pca.fit_transform(lig_concat)
    else:
        p_proj = prot_pca.transform(prot_emb)
        l_proj = lig_pca.transform(lig_concat)

    # Pad if PCA gave fewer components than dim (small datasets)
    if p_proj.shape[1] < dim:
        p_proj = np.pad(p_proj, ((0,0),(0, dim - p_proj.shape[1])))
    if l_proj.shape[1] < dim:
        l_proj = np.pad(l_proj, ((0,0),(0, dim - l_proj.shape[1])))

    hadamard = p_proj * l_proj
    diff     = np.abs(p_proj - l_proj)

    interaction = np.concatenate([p_proj, l_proj, hadamard, diff], axis=1)
    return interaction, prot_pca, lig_pca


def save_pcas(prot_pca, lig_pca, out_dir):
    joblib.dump(prot_pca, out_dir / "prot_pca.pkl")
    joblib.dump(lig_pca,  out_dir / "lig_pca.pkl")


def load_pcas(out_dir):
    return (joblib.load(out_dir / "prot_pca.pkl"),
            joblib.load(out_dir / "lig_pca.pkl"))
