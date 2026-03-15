# src/applicability_domain.py
#
# Three-layer applicability domain check.
# Flags garbage inputs before prediction is shown to the user.
#
# Layer 1 — Sequence sanity  (catches poly-A, random chars, empty)
# Layer 2 — Ligand sanity    (catches invalid SMILES, non-drug-like)
# Layer 3 — Embedding AD     (catches proteins far from training dist)

import numpy as np
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

STANDARD_AA = set('ACDEFGHIKLMNPQRSTVWY')


# ── Layer 1: Sequence ─────────────────────────────────────────────────

def check_sequence(seq: str) -> tuple:
    """Returns (score 0-100, list of warning strings)."""
    seq  = seq.strip().upper()
    warn = []

    if not seq:
        return 0.0, ["EMPTY_SEQUENCE"]

    invalid_frac = sum(1 for c in seq if c not in STANDARD_AA) / len(seq)
    if invalid_frac > 0.30:
        return 0.0, [f"NOT_A_PROTEIN: {invalid_frac:.0%} non-standard characters"]

    clean = ''.join(c for c in seq if c in STANDARD_AA)
    if not clean:
        return 0.0, ["NOT_A_PROTEIN: no standard amino acids found"]

    score = 100.0

    # Low complexity (poly-X)
    counts     = Counter(clean)
    top_frac   = counts.most_common(1)[0][1] / len(clean)
    if top_frac > 0.40:
        warn.append(f"LOW_COMPLEXITY: single AA = {top_frac:.0%} of sequence "
                    f"(likely poly-X repeat — prediction unreliable)")
        score -= 50

    # Shannon entropy
    freqs   = np.array(list(counts.values())) / len(clean)
    entropy = -np.sum(freqs * np.log2(freqs + 1e-10))
    if entropy < 2.5:
        warn.append(f"LOW_ENTROPY: {entropy:.2f} bits — low complexity sequence")
        score -= 25

    # Unique AAs
    if len(counts) < 5:
        warn.append(f"LOW_DIVERSITY: only {len(counts)} unique amino acids")
        score -= 25

    if len(seq) < 50:
        warn.append(f"SHORT: length {len(seq)} — may be a peptide, not a drug target")
        score -= 15

    if invalid_frac > 0:
        warn.append(f"NON_STANDARD: {invalid_frac:.0%} non-standard characters present")
        score -= 10

    return max(0.0, score), warn


# ── Layer 2: Ligand ───────────────────────────────────────────────────

def check_ligand(smiles: str) -> tuple:
    """Returns (score 0-100, list of warning strings)."""
    warn = []
    if not smiles or not smiles.strip():
        return 0.0, ["EMPTY_SMILES"]

    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return 0.0, ["INVALID_SMILES: RDKit could not parse this string"]

    score = 100.0

    mw = Descriptors.MolWt(mol)
    if mw < 100:
        warn.append(f"LOW_MW: {mw:.1f} Da — likely a fragment or solvent")
        score -= 40
    elif mw > 1000:
        warn.append(f"HIGH_MW: {mw:.1f} Da — may be outside training distribution")
        score -= 20

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy < 5:
        warn.append(f"TOO_SMALL: only {n_heavy} heavy atoms")
        score -= 40

    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53}
    exotic  = {a.GetSymbol() for a in mol.GetAtoms()
               if a.GetAtomicNum() not in allowed and a.GetAtomicNum() != 0}
    if exotic:
        warn.append(f"EXOTIC_ATOMS: {', '.join(sorted(exotic))} — rare in training data")
        score -= 20

    return max(0.0, score), warn


# ── Layer 3: Embedding AD ─────────────────────────────────────────────

class EmbeddingAD:
    """
    kNN applicability domain in ESM embedding space.
    Flags proteins far from the training distribution.
    """
    def __init__(self, k: int = 5, percentile: float = 95.0):
        self.k          = k
        self.percentile = percentile
        self.fitted     = False

    def fit(self, train_embeddings: np.ndarray):
        from sklearn.neighbors import NearestNeighbors
        print(f"Fitting Embedding AD on {len(train_embeddings)} proteins...")
        self.nn = NearestNeighbors(n_neighbors=self.k + 1,
                                   metric='cosine', n_jobs=-1)
        self.nn.fit(train_embeddings.astype(np.float32))
        dists, _ = self.nn.kneighbors(train_embeddings.astype(np.float32))
        knn_dists = dists[:, 1:].mean(axis=1)
        self.threshold = np.percentile(knn_dists, self.percentile)
        self.fitted = True
        print(f"  AD threshold ({self.percentile}th pct): {self.threshold:.4f}")
        return self

    def score(self, embedding: np.ndarray) -> tuple:
        """Returns (distance, score 0-100, in_domain bool)."""
        if not self.fitted:
            return None, 100.0, True
        dists, _ = self.nn.kneighbors(
            embedding.reshape(1, -1).astype(np.float32), n_neighbors=self.k
        )
        dist      = float(dists[0].mean())
        in_domain = dist <= self.threshold
        if dist <= self.threshold:
            ad_score = 50 + 50 * (1 - dist / self.threshold)
        else:
            ad_score = max(0.0, 50 * (1 - (dist - self.threshold) / self.threshold))
        return dist, ad_score, in_domain


# ── Combined report ───────────────────────────────────────────────────

def confidence_report(seq: str, smiles: str,
                      embedding: np.ndarray = None,
                      ad_model: EmbeddingAD = None) -> dict:
    seq_score, seq_warn = check_sequence(seq)
    lig_score, lig_warn = check_ligand(smiles)

    ad_score, ad_dist, in_domain = 100.0, None, True
    if embedding is not None and ad_model is not None:
        ad_dist, ad_score, in_domain = ad_model.score(embedding)
        if not in_domain:
            seq_warn.append(
                f"OUT_OF_DOMAIN: protein distance={ad_dist:.3f} "
                f"(threshold={ad_model.threshold:.3f})"
            )

    all_warn = seq_warn + lig_warn
    w_ad     = 0.20 if ad_model else 0.0
    w_seq    = 0.55 if not ad_model else 0.45
    w_lig    = 1.0 - w_seq - w_ad
    overall  = w_seq * seq_score + w_lig * lig_score + w_ad * ad_score

    if overall >= 70 and seq_score >= 60 and lig_score >= 60:
        flag   = 'RELIABLE'
    elif overall >= 40:
        flag   = 'UNCERTAIN'
    else:
        flag   = 'UNRELIABLE'

    return {
        'flag':               flag,
        'overall':            round(overall, 1),
        'seq_score':          round(seq_score, 1),
        'lig_score':          round(lig_score, 1),
        'ad_score':           round(ad_score, 1),
        'in_domain':          in_domain,
        'warnings':           all_warn,
    }
