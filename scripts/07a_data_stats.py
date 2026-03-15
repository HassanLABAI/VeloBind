import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path('data/processed')

def split_stats(csv_path, npz_path, split_name):
    df  = pd.read_csv(csv_path)
    npz = np.load(npz_path, allow_pickle=True)

    pKd        = df['label'].values
    seq_lens   = df['seq'].str.len()
    trunc_pct  = float(npz['truncated'].mean() * 100)

    return {
        'Split':              split_name,
        'N':                  len(df),
        'pKd mean':           f"{pKd.mean():.2f}",
        'pKd SD':             f"{pKd.std():.2f}",
        'pKd min':            f"{pKd.min():.2f}",
        'pKd max':            f"{pKd.max():.2f}",
        'Median seq len (AA)':f"{seq_lens.median():.0f}",
        'Mean seq len (AA)':  f"{seq_lens.mean():.0f}",
        'Max seq len (AA)':   f"{seq_lens.max():.0f}",
        '% truncated':        f"{trunc_pct:.1f}%",
    }

rows = [
    split_stats(DATA/'train_clean.csv',  DATA/'X_train.npz',  'Training'),
    split_stats(DATA/'casf16_clean.csv',   DATA/'X_test.npz',   'CASF-2016'),
    split_stats(DATA/'casf13_clean.csv', DATA/'X_casf13.npz', 'CASF-2013'),
]

table = pd.DataFrame(rows).set_index('Split')

print("\nTable S1 — Dataset Statistics")
print("=" * 70)
print(table.T.to_string())
print()

# Also save as CSV for copy-paste into Word/LaTeX
table.T.to_csv('output/table_s1_dataset_stats.csv')
print("Saved: output/table_s1_dataset_stats.csv")