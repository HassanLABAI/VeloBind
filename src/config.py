# src/config.py
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    SEED: int = 42

    ROOT_DIR: Path = Path(__file__).resolve().parent.parent

    def __post_init__(self):
        self.DATA_DIR   = self.ROOT_DIR / "data" / "processed"
        self.RAW_DIR    = self.ROOT_DIR / "data" / "raw"
        self.OUTPUT_DIR = self.ROOT_DIR / "output"
        self.CASF_DIR    = self.ROOT_DIR / "data" / "external" / "CASF-2016"
        self.CASF13_DIR  = self.ROOT_DIR / "data" / "external" / "CASF-2013"
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # LP-PDBBind CSV
    LPPDB_CSV: str = "LP_PDBBind.csv"   # filename inside RAW_DIR

    # ESM — 35M everywhere (train = deploy = benchmark, honest)
    ESM_MODEL:  str   = "facebook/esm2_t12_35M_UR50D"
    ESM_LAYERS: tuple = (8, 10, 11)   # 0-indexed, proportional to 20/26/30 in 150M
    ESM_DIM:    int   = 480           # hidden dim of 35M

    # Long-sequence chunking
    MAX_SEQ_LEN:  int = 1022
    HALF_SEQ_LEN: int = 511

    # Ligand
    ECFP_BITS:   int = 1024
    ECFP_RADIUS: int = 2

    # Interaction projection
    INTERACT_DIM: int = 128

    # GBM training
    # N_TREES=3000 with early_stop=150 is the sweet spot:
    #   - LR=0.02 + 3000 trees → actual stopping usually ~800-1200 trees
    #   - 5 seeds × 4 models × 5 folds = 100 runs × ~6 min = ~10 hrs total
    #   - Reduce SEEDS to (42, 123, 456) for ~6 hrs if needed
    N_FOLDS:    int   = 5
    SEEDS:      tuple = (42, 123, 456)   # 3 seeds ~6 hrs; add 789,1337 if time allows
    LR:         float = 0.02
    N_TREES:    int   = 3000
    EARLY_STOP: int   = 150

    # TTA
    TTA_SCREEN:   int = 5
    TTA_ACCURATE: int = 20

    # Applicability domain
    AD_KNN_K:      int   = 5
    AD_PERCENTILE: float = 95.0
    AD_MAX_MONO:   float = 0.40   # max single-AA fraction before flagging


config = Config()


if __name__ == "__main__":
    print(f"ROOT:   {config.ROOT_DIR}")
    print(f"DATA:   {config.DATA_DIR}   exists={config.DATA_DIR.exists()}")
    print(f"OUTPUT: {config.OUTPUT_DIR}  exists={config.OUTPUT_DIR.exists()}")
    print(f"CASF:   {config.CASF_DIR}   exists={config.CASF_DIR.exists()}")
    print(f"CASF13:   {config.CASF13_DIR}   exists={config.CASF13_DIR.exists()}")
    lp = config.RAW_DIR / config.LPPDB_CSV
    print(f"LP-PDBBind CSV: {lp}   exists={lp.exists()}")

