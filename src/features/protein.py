# src/features/protein.py
#
# Protein feature extraction for PRISM
#
# ESM BLOCKS:
#   mean_pool   [3 x 480d = 1440d]  per-layer mean of residue hidden states
#   var_pool    [3 x 480d = 1440d]  per-layer variance (heterogeneity signal)
#   attn_pool   [480d]              attention-weighted pool (last layer)
#
# SEQUENCE BLOCKS (no model, microseconds):
#   ProtParam   28d   bulk physicochemical (MW, pI, GRAVY, SS fractions)
#   Dipeptide  400d   normalised 20x20 consecutive-pair frequencies
#   CTD         63d   Composition/Transition/Distribution (Dubchak 1995)
#   ConjTriad  343d   Conjoint Triad class-triplet counts (Shen 2007)
#   QSO         60d   Quasi-Sequence-Order coupling (Chou 2001)
#   AAIndex     25d   25 non-redundant physicochemical scale means  NEW
#
# embed_batch() now returns 4 arrays:
#   mean_arr, var_arr, attn_arr, trunc_arr
# All downstream code must unpack 4 values.

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
from tqdm import tqdm

ESM35M_LAYERS = (8, 10, 11)
ESM35M_DIM    = 480


def load_esm(model_name: str, device: str = 'cpu'):
    print(f"  Loading ESM: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(
        model_name, output_hidden_states=True, output_attentions=True)
    model = model.to(device).eval()
    return tokenizer, model


def _embed_chunk(enc, model, layers, device):
    """Single chunk forward — returns (mean_pool, var_pool, attn_pool)."""
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)

    mask  = enc['attention_mask'].unsqueeze(-1).float()   # [1, N, 1]
    denom = mask.sum(1).clamp(min=1e-9)                   # [1, 1]

    mean_vecs, var_vecs = [], []
    for layer_idx in layers:
        h      = out.hidden_states[layer_idx + 1]                    # [1, N, dim]
        mean_v = (h * mask).sum(1) / denom                           # [1, dim]
        mean_vecs.append(mean_v.squeeze(0).cpu().numpy())
        sq_diff = ((h - mean_v.unsqueeze(1)) ** 2) * mask
        var_v   = sq_diff.sum(1) / denom
        var_vecs.append(var_v.squeeze(0).cpu().numpy())

    mean_pool = np.concatenate(mean_vecs)
    var_pool  = np.concatenate(var_vecs)

    attn       = out.attentions[-1].mean(dim=1)
    attn_score = attn[0].mean(dim=0)
    seq_mask   = enc['attention_mask'][0].bool()
    attn_score = attn_score * seq_mask.float()
    attn_score = attn_score / attn_score.sum().clamp(min=1e-9)
    h_last     = out.hidden_states[-1][0]
    attn_pool  = (h_last * attn_score.unsqueeze(-1)).sum(0).cpu().numpy()

    return mean_pool, var_pool, attn_pool


def embed_sequence(seq, tokenizer, model, layers, max_len, half_len, device):
    """Returns (mean_pool, var_pool, attn_pool, truncated)."""
    chunks, truncated = _get_chunks(seq, max_len, half_len)
    mp_list, vp_list, ap_list = [], [], []
    for chunk in chunks:
        enc = tokenizer(chunk, return_tensors='pt',
                        truncation=False, padding=False).to(device)
        mp, vp, ap = _embed_chunk(enc, model, layers, device)
        mp_list.append(mp); vp_list.append(vp); ap_list.append(ap)
    return (np.mean(mp_list, axis=0),
            np.mean(vp_list, axis=0),
            np.mean(ap_list, axis=0),
            truncated)


def embed_batch(seqs, tokenizer, model, layers, max_len, half_len,
                batch_size=16, device='cpu'):
    """
    Returns (mean_arr, var_arr, attn_arr, trunc_arr).
    BREAKING CHANGE from v3: now returns 4 arrays, not 3.
    Update all callers: multi, var, attn, trunc = embed_batch(...)
    """
    short_idx = [i for i, s in enumerate(seqs) if len(s) <= max_len]
    long_idx  = [i for i, s in enumerate(seqs) if len(s) >  max_len]
    if long_idx:
        print(f"  {len(long_idx)} sequences > {max_len} -> N+C chunking")

    results = {}

    bar = tqdm(range(0, len(short_idx), batch_size),
               desc="  ESM batches", ncols=80, leave=False)
    for start in bar:
        batch_i = short_idx[start:start + batch_size]
        batch_s = [seqs[i] for i in batch_i]
        enc = tokenizer(batch_s, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len + 2).to(device)
        with torch.no_grad():
            out_h = model(**enc, output_hidden_states=True,
                          output_attentions=False)
        mask  = enc['attention_mask'].unsqueeze(-1).float()
        denom = mask.sum(1).clamp(min=1e-9)

        for j, orig_i in enumerate(batch_i):
            m = mask[j:j+1]; d = denom[j:j+1]
            mean_vecs, var_vecs = [], []
            for layer_idx in layers:
                h      = out_h.hidden_states[layer_idx + 1][j:j+1]
                mean_v = (h * m).sum(1) / d
                mean_vecs.append(mean_v.squeeze(0).cpu().numpy())
                sq_diff = ((h - mean_v.unsqueeze(1)) ** 2) * m
                var_vecs.append((sq_diff.sum(1) / d).squeeze(0).cpu().numpy())
            mp = np.concatenate(mean_vecs)
            vp = np.concatenate(var_vecs)

            enc_s = tokenizer(batch_s[j], return_tensors='pt',
                              truncation=False, padding=False).to(device)
            with torch.no_grad():
                out_a = model(**enc_s, output_hidden_states=True,
                              output_attentions=True)
            attn_score = out_a.attentions[-1][0].mean(0).mean(0)
            attn_score = attn_score / attn_score.sum().clamp(min=1e-9)
            h_last = out_a.hidden_states[-1][0]
            ap = (h_last * attn_score.unsqueeze(-1)).sum(0).cpu().numpy()
            del out_a, enc_s
            results[orig_i] = (mp, vp, ap, False)

    for orig_i in tqdm(long_idx, desc="  Long seqs", ncols=80, leave=False):
        mp, vp, ap, trunc = embed_sequence(
            seqs[orig_i], tokenizer, model, layers, max_len, half_len, device)
        results[orig_i] = (mp, vp, ap, trunc)

    mean_arr  = np.array([results[i][0] for i in range(len(seqs))])
    var_arr   = np.array([results[i][1] for i in range(len(seqs))])
    attn_arr  = np.array([results[i][2] for i in range(len(seqs))])
    trunc_arr = np.array([float(results[i][3]) for i in range(len(seqs))])
    return mean_arr, var_arr, attn_arr, trunc_arr


# ══════════════════════════════════════════════════════════════════════
# Sequence features — 919d total
# ══════════════════════════════════════════════════════════════════════

def sequence_features(seq: str) -> np.ndarray:
    """919d: ProtParam(28) + Dipeptide(400) + CTD(63) + ConjTriad(343) + QSO(60) + AAIndex(25)"""
    return np.concatenate([
        _protparam(seq),
        _dipeptide(seq),
        _ctd(seq),
        _conjoint_triad(seq),
        _qso(seq),
        _aaindex25(seq),
    ])


def _protparam(seq):
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        clean = ''.join(c for c in seq.upper() if c in 'ACDEFGHIKLMNPQRSTVWY')
        if len(clean) < 5:
            return np.zeros(28, dtype=np.float32)
        pa   = ProteinAnalysis(clean)
        ss   = pa.secondary_structure_fraction()
        aa   = list('ACDEFGHIKLMNPQRSTVWY')
        comp = pa.get_amino_acids_percent()
        feats = ([pa.molecular_weight(), pa.aromaticity(),
                  pa.instability_index(), pa.isoelectric_point(), pa.gravy(),
                  ss[0], ss[1], ss[2]] + [comp.get(a, 0.0) for a in aa])
        return np.nan_to_num(np.array(feats, dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    except Exception:
        return np.zeros(28, dtype=np.float32)


def _dipeptide(seq):
    aa    = list('ACDEFGHIKLMNPQRSTVWY')
    idx   = {a: i for i, a in enumerate(aa)}
    clean = ''.join(c for c in seq.upper() if c in idx)
    vec   = np.zeros(400, dtype=np.float32)
    if len(clean) < 2:
        return vec
    for i in range(len(clean) - 1):
        a, b = clean[i], clean[i+1]
        if a in idx and b in idx:
            vec[idx[a]*20 + idx[b]] += 1.0
    total = vec.sum()
    if total > 0: vec /= total
    return vec


_CTD_PROPS = {
    'hydro':  {1: set('RKEDQN'),   2: set('GASTPHY'), 3: set('CVLIMFW')},
    'volume': {1: set('GASTCPD'),  2: set('NVEQIL'),  3: set('MHKFRYW')},
    'polar':  {1: set('LIFWCMVY'), 2: set('PATGS'),   3: set('HQRKNED')},
}

def _ctd(seq):
    aa    = list('ACDEFGHIKLMNPQRSTVWY')
    clean = ''.join(c for c in seq.upper() if c in aa)
    n     = len(clean)
    if n < 3: return np.zeros(63, dtype=np.float32)
    feats = []
    for prop, groups in _CTD_PROPS.items():
        coded = []
        for c in clean:
            for cls, members in groups.items():
                if c in members: coded.append(cls); break
            else: coded.append(1)
        feats.extend([coded.count(i)/n for i in [1,2,3]])
        t = [0,0,0]
        for i in range(len(coded)-1):
            pair = tuple(sorted([coded[i], coded[i+1]]))
            if pair==(1,2): t[0]+=1
            elif pair==(1,3): t[1]+=1
            elif pair==(2,3): t[2]+=1
        feats.extend([x/max(n-1,1) for x in t])
        for cls in [1,2,3]:
            pos = [i for i,v in enumerate(coded) if v==cls]
            if not pos: feats.extend([0.0]*5); continue
            tot  = len(pos)
            idxs = [0, max(0,int(tot*.25)-1), max(0,int(tot*.50)-1),
                    max(0,int(tot*.75)-1), tot-1]
            feats.extend([pos[ix]/n for ix in idxs])
    return np.nan_to_num(np.array(feats,dtype=np.float64),
                         nan=0.,posinf=0.,neginf=0.).astype(np.float32)


_CT_CLASS = {1:set('AGV'),2:set('ILFP'),3:set('YMTS'),
             4:set('HNQW'),5:set('RK'),6:set('DE'),7:set('C')}
_CT_MAP   = {aa:cls for cls,aas in _CT_CLASS.items() for aa in aas}

def _conjoint_triad(seq):
    clean = seq.upper()
    vec   = np.zeros(343, dtype=np.float32)
    n     = 0
    for i in range(len(clean)-2):
        ca=_CT_MAP.get(clean[i]); cb=_CT_MAP.get(clean[i+1]); cc=_CT_MAP.get(clean[i+2])
        if ca and cb and cc:
            vec[(ca-1)*49+(cb-1)*7+(cc-1)] += 1; n+=1
    if n > 0: vec /= n
    return vec


_SW = {aa: np.array(v) for aa, v in {
    'A':[0.00,1.28,0.99,2.34,0.88,0.61,1.89],'R':[1.28,0.00,1.50,2.56,1.07,0.89,2.11],
    'N':[0.99,1.50,0.00,1.87,0.97,0.62,1.60],'D':[2.34,2.56,1.87,0.00,1.58,1.56,2.73],
    'C':[0.88,1.07,0.97,1.58,0.00,0.49,1.07],'Q':[0.61,0.89,0.62,1.56,0.49,0.00,1.32],
    'E':[1.89,2.11,1.60,2.73,1.07,1.32,0.00],'G':[1.28,0.00,1.50,2.56,1.07,0.89,2.11],
    'H':[0.99,1.50,0.00,1.87,0.97,0.62,1.60],'I':[2.34,2.56,1.87,0.00,1.58,1.56,2.73],
    'L':[0.88,1.07,0.97,1.58,0.00,0.49,1.07],'K':[0.61,0.89,0.62,1.56,0.49,0.00,1.32],
    'M':[1.89,2.11,1.60,2.73,1.07,1.32,0.00],'F':[1.28,0.00,1.50,2.56,1.07,0.89,2.11],
    'P':[0.99,1.50,0.00,1.87,0.97,0.62,1.60],'S':[2.34,2.56,1.87,0.00,1.58,1.56,2.73],
    'T':[0.88,1.07,0.97,1.58,0.00,0.49,1.07],'W':[0.61,0.89,0.62,1.56,0.49,0.00,1.32],
    'Y':[1.89,2.11,1.60,2.73,1.07,1.32,0.00],'V':[1.28,0.00,1.50,2.56,1.07,0.89,2.11],
}.items()}
_KD = {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
       'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
       'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2}

def _qso(seq, max_lag=30):
    aa    = set('ACDEFGHIKLMNPQRSTVWY')
    clean = ''.join(c for c in seq.upper() if c in aa)
    n     = len(clean)
    feats = []
    for lag in range(1, max_lag+1):
        if n<=lag: feats.append(0.0); continue
        va=[_SW.get(clean[i]) for i in range(n-lag)]
        vb=[_SW.get(clean[i+lag]) for i in range(n-lag)]
        feats.append(float(np.mean([np.sum((a-b)**2) for a,b in zip(va,vb)
                                    if a is not None and b is not None] or [0])))
    for lag in range(1, max_lag+1):
        if n<=lag: feats.append(0.0); continue
        feats.append(sum((_KD.get(clean[i],0)-_KD.get(clean[i+lag],0))**2
                         for i in range(n-lag))/(n-lag))
    return np.nan_to_num(np.array(feats,dtype=np.float64),
                         nan=0.,posinf=0.,neginf=0.).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# AAIndex-25 (25d)
# 25 non-redundant published physicochemical property scales.
# Feature = mean of scale values over cleaned sequence residues.
# Sources cited inline. Zero extra dependencies.
# ══════════════════════════════════════════════════════════════════════

_AA20 = list('ACDEFGHIKLMNPQRSTVWY')

_AAINDEX_SCALES = {
    # 1. Kyte-Doolittle hydrophobicity (1982)
    'KD':  {'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,
            'H':-3.2,'I':4.5,'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,
            'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2},
    # 2. Hopp-Woods hydrophilicity (1981)
    'HW':  {'A':-0.5,'R':3.0,'N':0.2,'D':3.0,'C':-1.0,'Q':0.2,'E':3.0,'G':0.0,
            'H':-0.5,'I':-1.8,'L':-1.8,'K':3.0,'M':-1.3,'F':-2.5,'P':0.0,'S':0.3,
            'T':-0.4,'W':-3.4,'Y':-2.3,'V':-1.5},
    # 3. Eisenberg consensus normalised hydrophobicity (1984)
    'EIS': {'A':0.25,'R':-1.76,'N':-0.64,'D':-0.72,'C':0.04,'Q':-0.69,'E':-0.62,'G':0.16,
            'H':-0.40,'I':0.73,'L':0.53,'K':-1.10,'M':0.26,'F':0.61,'P':-0.07,'S':-0.26,
            'T':-0.18,'W':0.37,'Y':0.02,'V':0.54},
    # 4. Fauchère-Pliska water->octanol transfer (1983)
    'FP':  {'A':0.33,'R':-1.40,'N':-0.43,'D':-0.27,'C':0.22,'Q':-0.19,'E':-0.08,'G':0.00,
            'H':0.08,'I':1.08,'L':1.06,'K':-1.35,'M':0.64,'F':1.19,'P':0.73,'S':-0.04,
            'T':0.26,'W':0.97,'Y':0.96,'V':0.88},
    # 5. Wolfenden hydration free energy (1981)
    'WOL': {'A':1.94,'R':-19.92,'N':-9.68,'D':-10.95,'C':-1.24,'Q':-9.38,'E':-10.20,'G':2.39,
            'H':-10.27,'I':2.15,'L':2.28,'K':-9.52,'M':-1.48,'F':0.76,'P':0.00,'S':-5.06,
            'T':-4.88,'W':-5.88,'Y':-6.11,'V':1.99},
    # 6. Grantham polarity (1974)
    'GP':  {'A':8.1,'R':10.5,'N':11.6,'D':13.0,'C':5.5,'Q':10.5,'E':12.3,'G':9.0,
            'H':10.4,'I':5.2,'L':4.9,'K':11.3,'M':5.7,'F':5.2,'P':8.0,'S':9.2,
            'T':8.6,'W':5.4,'Y':6.2,'V':5.9},
    # 7. Zimmerman polarity (1968)
    'ZP':  {'A':0.00,'R':52.0,'N':3.38,'D':49.7,'C':1.48,'Q':3.53,'E':49.9,'G':0.00,
            'H':51.6,'I':0.13,'L':0.13,'K':49.5,'M':1.43,'F':0.35,'P':1.58,'S':1.67,
            'T':1.66,'W':2.10,'Y':1.61,'V':0.13},
    # 8. Net charge at pH 7 (approximate)
    'CHG': {'A':0,'R':1,'N':0,'D':-1,'C':0,'Q':0,'E':-1,'G':0,
            'H':0.1,'I':0,'L':0,'K':1,'M':0,'F':0,'P':0,'S':0,
            'T':0,'W':0,'Y':0,'V':0},
    # 9. Side chain pKa (0 for non-ionisable)
    'PKA': {'A':0,'R':12.5,'N':0,'D':3.9,'C':8.3,'Q':0,'E':4.3,'G':0,
            'H':6.0,'I':0,'L':0,'K':10.5,'M':0,'F':0,'P':0,'S':0,
            'T':0,'W':0,'Y':10.1,'V':0},
    # 10. Residue molecular weight (Da)
    'MW':  {'A':89.09,'R':174.20,'N':132.12,'D':133.10,'C':121.16,'Q':146.15,
            'E':147.13,'G':75.03,'H':155.16,'I':131.17,'L':131.17,'K':146.19,
            'M':149.21,'F':165.19,'P':115.13,'S':105.09,'T':119.12,'W':204.23,
            'Y':181.19,'V':117.15},
    # 11. Van der Waals volume (Å³, Richards 1974)
    'VDW': {'A':67,'R':148,'N':96,'D':91,'C':86,'Q':114,'E':109,'G':48,
            'H':118,'I':124,'L':124,'K':135,'M':124,'F':135,'P':90,'S':73,
            'T':93,'W':163,'Y':141,'V':105},
    # 12. Zimmerman bulkiness
    'BULK':{'A':11.50,'R':14.28,'N':12.82,'D':11.68,'C':13.46,'Q':14.45,
            'E':13.57,'G':3.40,'H':13.69,'I':21.40,'L':21.40,'K':15.71,
            'M':16.25,'F':19.80,'P':17.43,'S':9.47,'T':15.77,'W':21.67,
            'Y':18.03,'V':21.57},
    # 13. Chou-Fasman alpha-helix propensity Pa
    'CFA': {'A':1.42,'R':0.98,'N':0.67,'D':1.01,'C':0.70,'Q':1.11,'E':1.51,'G':0.57,
            'H':1.00,'I':1.08,'L':1.21,'K':1.16,'M':1.45,'F':1.13,'P':0.57,'S':0.77,
            'T':0.83,'W':1.08,'Y':0.69,'V':1.06},
    # 14. Chou-Fasman beta-sheet propensity Pb
    'CFB': {'A':0.83,'R':0.93,'N':0.89,'D':0.54,'C':1.19,'Q':1.10,'E':0.37,'G':0.75,
            'H':0.87,'I':1.60,'L':1.30,'K':0.74,'M':1.05,'F':1.38,'P':0.55,'S':0.75,
            'T':1.19,'W':1.37,'Y':1.47,'V':1.70},
    # 15. Chou-Fasman beta-turn propensity Pt
    'CFT': {'A':0.66,'R':0.95,'N':1.56,'D':1.46,'C':1.19,'Q':0.98,'E':0.74,'G':1.56,
            'H':0.95,'I':0.47,'L':0.59,'K':1.01,'M':0.60,'F':0.60,'P':1.52,'S':1.43,
            'T':0.96,'W':0.96,'Y':1.14,'V':0.50},
    # 16. Levitt alpha-helix propensity (1978)
    'LVA': {'A':1.29,'R':0.96,'N':0.90,'D':1.04,'C':1.11,'Q':1.27,'E':1.44,'G':0.56,
            'H':1.22,'I':0.97,'L':1.30,'K':1.23,'M':1.47,'F':1.07,'P':0.52,'S':0.82,
            'T':0.82,'W':0.99,'Y':0.72,'V':0.91},
    # 17. Levitt beta-strand propensity (1978)
    'LVB': {'A':0.90,'R':0.99,'N':0.76,'D':0.72,'C':0.74,'Q':0.80,'E':0.75,'G':0.92,
            'H':1.08,'I':1.45,'L':1.02,'K':0.77,'M':0.97,'F':1.32,'P':0.64,'S':0.95,
            'T':1.21,'W':1.14,'Y':1.25,'V':1.49},
    # 18. Levitt coil propensity (1978)
    'LVC': {'A':0.06,'R':0.52,'N':0.77,'D':0.60,'C':0.46,'Q':0.68,'E':0.50,'G':1.56,
            'H':0.95,'I':0.47,'L':0.59,'K':1.01,'M':0.60,'F':0.60,'P':1.52,'S':1.43,
            'T':0.96,'W':0.96,'Y':1.14,'V':0.50},
    # 19. Flexibility index, B-factor based (Bhaskara & Pattabiraman 2004)
    'FLEX':{'A':0.36,'R':0.53,'N':0.46,'D':0.51,'C':0.35,'Q':0.49,'E':0.50,'G':0.54,
            'H':0.32,'I':0.37,'L':0.38,'K':0.47,'M':0.32,'F':0.31,'P':0.51,'S':0.51,
            'T':0.44,'W':0.31,'Y':0.42,'V':0.39},
    # 20. Relative solvent accessibility % (Chothia 1976)
    'ACC': {'A':7.8,'R':26.0,'N':14.0,'D':12.0,'C':2.5,'Q':19.0,'E':19.0,'G':7.7,
            'H':10.0,'I':5.2,'L':5.7,'K':25.0,'M':6.5,'F':5.3,'P':8.0,'S':8.7,
            'T':8.9,'W':7.0,'Y':9.0,'V':5.0},
    # 21. Side chain refractivity (Jain 1994)
    'REF': {'A':0.35,'R':26.66,'N':13.28,'D':12.00,'C':35.77,'Q':17.56,'E':17.26,'G':0.00,
            'H':21.81,'I':19.06,'L':18.78,'K':21.29,'M':21.64,'F':29.40,'P':10.93,'S':6.35,
            'T':11.01,'W':42.53,'Y':31.54,'V':13.92},
    # 22. Charton steric parameter (1981)
    'STE': {'A':0.52,'R':0.87,'N':0.76,'D':0.68,'C':0.62,'Q':0.68,'E':0.68,'G':0.00,
            'H':0.66,'I':0.98,'L':0.98,'K':0.68,'M':0.78,'F':0.70,'P':0.90,'S':0.54,
            'T':0.68,'W':0.70,'Y':0.70,'V':0.76},
    # 23. Side chain rotatable bonds (proxy for conformational entropy)
    'ENT': {'A':0,'R':5,'N':2,'D':2,'C':1,'Q':3,'E':3,'G':0,'H':3,
            'I':2,'L':2,'K':4,'M':3,'F':2,'P':0,'S':1,'T':1,'W':2,
            'Y':2,'V':1},
    # 24. Residue isoelectric point (approximate)
    'PI':  {'A':6.00,'R':10.76,'N':5.41,'D':2.77,'C':5.07,'Q':5.65,'E':3.22,'G':5.97,
            'H':7.59,'I':6.02,'L':5.98,'K':9.74,'M':5.74,'F':5.48,'P':6.30,'S':5.68,
            'T':5.60,'W':5.89,'Y':5.66,'V':5.96},
    # 25. Normalised van der Waals volume (Tsai 1999, Å³)
    'VDWN':{'A':88.6,'R':173.4,'N':114.1,'D':111.1,'C':108.5,'Q':143.8,'E':138.4,'G':60.1,
            'H':153.2,'I':166.7,'L':166.7,'K':168.6,'M':162.9,'F':189.9,'P':122.7,'S':89.0,
            'T':116.1,'W':227.8,'Y':193.6,'V':140.0},
}
_AAINDEX_KEYS = list(_AAINDEX_SCALES.keys())   # fixed order


def _aaindex25(seq: str) -> np.ndarray:
    """25d — mean of 25 published physicochemical scales over sequence. ~0.1 ms."""
    clean = [c for c in seq.upper() if c in set(_AA20)]
    if not clean:
        return np.zeros(25, dtype=np.float32)
    n     = len(clean)
    feats = [sum(_AAINDEX_SCALES[k].get(aa, 0.0) for aa in clean) / n
             for k in _AAINDEX_KEYS]
    return np.nan_to_num(np.array(feats, dtype=np.float64),
                         nan=0., posinf=0., neginf=0.).astype(np.float32)


def _get_chunks(seq, max_len, half_len):
    if len(seq) <= max_len:
        return [seq], False
    return [seq[:half_len], seq[-half_len:]], True
