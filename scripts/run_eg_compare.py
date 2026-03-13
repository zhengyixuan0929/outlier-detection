import os
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets import load_dataset, DATASETS
from src.eg_hdiod import (
    knn_distance_score_fixed,
    gated_hdiod_score,
    eg_hdiod_score,
)

# ================= config =================
K_LIST = list(range(5, 101, 5))
# K_LIST = [5, 10, 15, 20, 25, 30]

RESULTS_DIR = "results"
OUT_ALL = os.path.join(RESULTS_DIR, "run_eg_compare_results.csv")
OUT_OVERALL = os.path.join(RESULTS_DIR, "run_eg_compare_overall.csv")

# ----- GATED_HDIOD params -----
GATED_LAM = 0.6
GATED_GAMMA = 2.0

# ----- EG_HDIOD params -----
EG_LAM = 0.6
EG_GAMMA = 2.0
EG_K_EXPAND_MODE = "double"   # "double" or "plus"
EG_EXPAND_OFFSET = 10         # used only when mode == "plus"
# ==========================================


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def add_row(rows: List[Dict], dataset: str, method: str, param: str, k: int, auc: float):
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "param": param,
            "k": k,
            "auc": auc,
        }
    )


def run_one_dataset(name: str) -> pd.DataFrame:
    X, y = load_dataset(name)
    rows: List[Dict] = []

    for k in K_LIST:
        if k >= len(X):
            continue

        # ---------- KNN ----------
        s_knn = knn_distance_score_fixed(X, k=k)
        add_row(rows, name, "KNN", f"k={k}", k, safe_auc(y, s_knn))

        # ---------- GATED_HDIOD ----------
        s_gated = gated_hdiod_score(
            X,
            k=k,
            lam=GATED_LAM,
            gamma=GATED_GAMMA,
        )
        add_row(
            rows,
            name,
            "GATED_HDIOD",
            f"k={k},lam={GATED_LAM},gamma={GATED_GAMMA}",
            k,
            safe_auc(y, s_gated),
        )

        # ---------- EG_HDIOD ----------
        s_eg = eg_hdiod_score(
            X,
            k=k,
            k_expand_mode=EG_K_EXPAND_MODE,
            expand_offset=EG_EXPAND_OFFSET,
            lam=EG_LAM,
            gamma=EG_GAMMA,
        )

        if EG_K_EXPAND_MODE == "double":
            eg_param = f"k={k},k_exp=2k,lam={EG_LAM},gamma={EG_GAMMA}"
        else:
            eg_param = f"k={k},k_exp=k+{EG_EXPAND_OFFSET},lam={EG_LAM},gamma={EG_GAMMA}"

        add_row(rows, name, "EG_HDIOD", eg_param, k, safe_auc(y, s_eg))

    df = pd.DataFrame(rows)
    df["auc"] = df["auc"].astype(float)
    return df


def print_all_results_table(df_one: pd.DataFrame):
    show = df_one.copy()
    show["auc"] = show["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    show = show.sort_values(["k", "auc"], ascending=[True, False]).reset_index(drop=True)
    print(show[["k", "method", "param", "auc"]].to_string(index=False))


def print_dataset_leaderboards(df_one: pd.DataFrame):
    for k in sorted(df_one["k"].dropna().unique()):
        sub = df_one[df_one["k"] == k].copy()
        sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))
        sub["auc"] = sub["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))

        print(f"\n[Leaderboard | fixed k = {int(k)}]")
        print(sub[["rank", "method", "auc"]].to_string(index=False))


def overall_leaderboard(all_df: pd.DataFrame) -> pd.DataFrame:
    out = (
        all_df.groupby(["method", "k"], as_index=False)["auc"]
        .mean()
        .sort_values(["k", "auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    out["auc_rounded"] = out["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return out


def main():
    warnings.filterwarnings("ignore")
    ensure_results_dir()

    print("=" * 70)
    print("RUN EG COMPARE: KNN vs GATED_HDIOD vs EG_HDIOD")
    print("=" * 70)

    print("Config:")
    print(f"  GATED_HDIOD -> lam={GATED_LAM}, gamma={GATED_GAMMA}")
    if EG_K_EXPAND_MODE == "double":
        print(f"  EG_HDIOD    -> k_exp=2k, lam={EG_LAM}, gamma={EG_GAMMA}")
    else:
        print(f"  EG_HDIOD    -> k_exp=k+{EG_EXPAND_OFFSET}, lam={EG_LAM}, gamma={EG_GAMMA}")

    all_results = []

    for name in DATASETS.keys():
        print("\n" + "=" * 70)
        print(f"Dataset: {name}")
        print("=" * 70)

        df_one = run_one_dataset(name)
        all_results.append(df_one)

        print("\n[All results]")
        print_all_results_table(df_one)

        print("\n[Per-dataset leaderboards]")
        print_dataset_leaderboards(df_one)

    all_df = pd.concat(all_results, ignore_index=True)

    # save full
    all_df_out = all_df.copy()
    all_df_out["auc"] = all_df_out["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 6))
    all_df_out.to_csv(OUT_ALL, index=False)
    print("\nSaved ->", OUT_ALL)

    # overall
    overall = overall_leaderboard(all_df)

    print("\n" + "=" * 70)
    print("OVERALL LEADERBOARD (mean AUC across datasets)")
    print("=" * 70)

    for k in sorted(overall["k"].dropna().unique()):
        sub = overall[overall["k"] == k].copy()
        sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))

        print(f"\n[Overall | fixed k = {int(k)}]")
        print(
            sub[["rank", "method", "auc_rounded"]]
            .rename(columns={"auc_rounded": "auc"})
            .to_string(index=False)
        )

    overall.to_csv(OUT_OVERALL, index=False)
    print("\nSaved ->", OUT_OVERALL)


if __name__ == "__main__":
    main()