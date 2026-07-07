import sys
import time
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.models.conformal import ConformalSelectivePredictor, base_disagreement

MODELS = config.OUTPUT_DIR / "models"
OUTC = config.OUTPUT_DIR / "conformal"
CACHE = config.OUTPUT_DIR / "pdb_class_cache.csv"
MIN_N = 8

CLASS_RULES = [
    ("kinase",            ["kinase", "transferase/transferase", "tyrosine-protein kinase"]),
    ("protease",          ["protease", "peptidase", "proteinase", "hydrolase/hydrolase inhibitor",
                           "aspartic", "serine protease", "metalloprotease"]),
    ("GPCR",              ["g protein-coupled", "gpcr", "rhodopsin", "adrenergic receptor",
                           "adenosine receptor", "membrane receptor"]),
    ("nuclear receptor",  ["nuclear receptor", "estrogen receptor", "androgen receptor",
                           "glucocorticoid", "retinoic acid", "ppar", "thyroid hormone"]),
    ("oxidoreductase",    ["oxidoreductase", "dehydrogenase", "reductase", "oxidase",
                           "cytochrome", "monooxygenase"]),
    ("transferase",       ["transferase", "methyltransferase", "glycosyltransferase",
                           "polymerase", "synthase"]),
    ("hydrolase",         ["hydrolase", "esterase", "lipase", "phosphatase", "phosphodiesterase",
                           "nuclease", "glycosidase"]),
    ("isomerase",         ["isomerase", "topoisomerase", "mutase"]),
    ("lyase",             ["lyase", "carbonic anhydrase", "decarboxylase", "anhydrase"]),
    ("ligase",            ["ligase", "synthetase"]),
    ("transport/binding", ["transport protein", "binding protein", "carrier", "albumin"]),
    ("immune/signaling",  ["immune system", "signaling protein", "cytokine", "chaperone",
                           "viral protein", "structural protein"]),
]


def classify(text):
    t = (text or "").lower()
    for cls, kws in CLASS_RULES:
        if any(k in t for k in kws):
            return cls
    return "other"


def load_cache():
    if CACHE.exists():
        df = pd.read_csv(CACHE, dtype=str)
        return dict(zip(df["pdb_id"].str.lower(), df["pdb_class"]))
    return {}


def save_cache(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"pdb_id": list(cache), "pdb_class": list(cache.values())}).to_csv(CACHE, index=False)


def fetch_class(pdb):
    """Coarse functional class for one PDB id from the RCSB entry record."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb.lower()}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "velobind/1.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            j = json.load(r)
        title = j.get("struct", {}).get("title", "")
        kw = j.get("struct_keywords", {})
        text = " ".join([title, kw.get("pdbx_keywords", ""), kw.get("text", "")])
        return classify(text)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def resolve_classes(pdb_ids, cache, rate=0.1):
    """Fill `cache` for any missing pdb_ids; return class array aligned to pdb_ids."""
    uniq = sorted({p.lower() for p in pdb_ids})
    missing = [p for p in uniq if p not in cache]
    for i, p in enumerate(missing):
        c = fetch_class(p)
        cache[p] = c if c is not None else "other"
        if (i + 1) % 50 == 0:
            print(f"    fetched {i + 1}/{len(missing)}")
            save_cache(cache)
        time.sleep(rate)
    if missing:
        save_cache(cache)
    return np.array([cache.get(p.lower(), "other") for p in pdb_ids])


def honest_oof(oof, y):
    pred = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=42).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        pred[va] = m.predict(oof[va])
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-train", action="store_true",
                    help="skip training-OOF coverage; only CASF-16 per-class R")
    ap.add_argument("--rate", type=float, default=0.1, help="seconds between RCSB calls")
    args = ap.parse_args()
    OUTC.mkdir(parents=True, exist_ok=True)
    cache = load_cache()

    import joblib
    test = np.load(MODELS / "test_matrix.npy")
    meta = joblib.load(MODELS / "meta.pkl")
    c16 = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")
    y16 = c16["label"].values
    p16 = meta.predict(test)
    print(f"Resolving classes for {c16['pdb_id'].nunique()} CASF-2016 proteins...")
    cls16 = resolve_classes(c16["pdb_id"].tolist(), cache, args.rate)

    # per-class scoring R on CASF-16
    casf_r = {}
    for cl in pd.unique(cls16):
        m = cls16 == cl
        if m.sum() >= MIN_N:
            casf_r[cl] = (int(m.sum()), round(float(pearsonr(p16[m], y16[m])[0]), 3))

    # per-class conformal coverage on training OOF
    cov = {}
    if not args.no_train:
        oof = np.load(MODELS / "oof_matrix.npy")
        d = np.load(config.DATA_DIR / "X_train.npz", allow_pickle=True)
        ytr = d["labels"].astype(float)
        pdb_tr = [str(x) for x in d["pdb_ids"]]
        print(f"Resolving classes for {len(set(pdb_tr))} training proteins (cached)...")
        cls_tr = resolve_classes(pdb_tr, cache, args.rate)
        sig = base_disagreement(oof)
        om = honest_oof(oof, ytr)
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(ytr))
        cal, val = idx[: len(ytr) // 2], idx[len(ytr) // 2:]
        cp = ConformalSelectivePredictor(alpha=0.1).fit(om[cal], ytr[cal], cal_sigma=sig[cal])
        cls_val = cls_tr[val]
        for cl in pd.unique(cls_val):
            m = cls_val == cl
            if m.sum() >= MIN_N:
                cv = cp.coverage(om[val][m], ytr[val][m], sigma=sig[val][m])
                cov[cl] = (int(m.sum()), round(cv, 3))

    classes = sorted(set(casf_r) | set(cov), key=lambda c: -(casf_r.get(c, (0,))[0]))
    rows = []
    for cl in classes:
        rows.append({
            "class": cl,
            "CASF16_n": casf_r.get(cl, ("", ""))[0],
            "CASF16_R": casf_r.get(cl, ("", ""))[1] if cl in casf_r else "",
            "train_n": cov.get(cl, ("", ""))[0],
            "coverage90": cov.get(cl, ("", ""))[1] if cl in cov else "",
        })
    df = pd.DataFrame(rows)
    out = OUTC / "family_named.csv"
    df.to_csv(out, index=False)
    print("\n=== Named-family generalization ===")
    print(df.to_string(index=False))
    print(f"\nSaved -> {out}")
    print(f"Class cache: {CACHE}  (hand-editable)")


if __name__ == "__main__":
    main()
