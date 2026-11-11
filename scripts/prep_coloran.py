"""Build the compact Tier-C KPI bank from the ColO-RAN dataset (wineslab).

Source: https://github.com/wineslab/colosseum-oran-coloran-dataset (rome_static_medium)
  7 base stations x 3 scheduling policies x 28 slice-PRB configurations x 5 repetitions.
Output: data/oran/coloran_kpi.npz with
  X[bs, sched, tr, exp, win, feat]  (7, 3, 28, 5, 8, 9) float32
  feat = per slice s in {eMBB, MTC, URLLC}: [DL throughput sum over UEs (Mbps),
          log1p mean DL buffer (bytes), mean granted PRBs / 1000]
(DL tx error is 0 throughout this dataset, so it is not used.)
Each (exp, win) is one measurement window (1/8 of a run, ~30-60 s of 250 ms KPI reports),
which is what one KPI reporter would observe.

Usage: python scripts/prep_coloran.py /path/to/colosseum-oran-coloran-dataset
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN = 8
COLS = ["slice_id", "tx_brate downlink [Mbps]", "dl_buffer [bytes]", "sum_granted_prbs"]


def one(args):
    d, = args
    acc = np.zeros((WIN, 3, 3)); cnt = np.zeros((WIN, 3))
    for f in sorted(Path(d).glob("slices_bs*/*_metrics.csv")):
        try:
            df = pd.read_csv(f, usecols=COLS)[COLS]          # usecols keeps FILE order; reorder
        except Exception:
            continue
        df = df[df["slice_id"].isin([0, 1, 2])]
        if len(df) < WIN * 4:
            continue
        s = int(df["slice_id"].mode()[0])
        for w, ch in enumerate(np.array_split(df.to_numpy(dtype=float), WIN)):
            acc[w, s, 0] += ch[:, 1].mean()                 # throughput: sum over UEs
            acc[w, s, 1] += np.log1p(ch[:, 2].mean())       # buffer: mean over UEs
            acc[w, s, 2] += ch[:, 3].mean() / 1000          # granted PRBs (x1000): mean over UEs
            cnt[w, s] += 1
    out = acc.copy()
    out[:, :, 1:] /= np.maximum(cnt, 1)[..., None]
    return out.reshape(WIN, 9), int(cnt[0].sum())


def main():
    base = Path(sys.argv[1]) / "rome_static_medium"
    idx, jobs = [], []
    for bs in range(1, 8):
        for sc in range(3):
            for tr in range(28):
                for ex in range(1, 6):
                    idx.append((bs - 1, sc, tr, ex - 1))
                    jobs.append((str(base / f"sched{sc}" / f"tr{tr}" / f"exp{ex}" / f"bs{bs}"),))
    X = np.full((7, 3, 28, 5, WIN, 9), np.nan, dtype=np.float32)
    nue = np.zeros((7, 3, 28, 5), dtype=np.int16)
    with ProcessPoolExecutor() as ex:
        for i, (x, n) in zip(idx, ex.map(one, jobs, chunksize=8)):
            X[i] = x; nue[i] = n
    out = ROOT / "data" / "oran"; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "coloran_kpi.npz", X=X, n_ue=nue)
    print("saved", X.shape, "missing runs:", int((nue == 0).sum()), "of", nue.size)


if __name__ == "__main__":
    main()
