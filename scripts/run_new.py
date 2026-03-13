import os
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets import load_dataset, DATASETS
from src.baselines import lof_score, cof_score, ldof_score
from src.hdiod import hdiod_score_paper
from src.new_hdiod import (
    knn_distance_score_fixed,
    gated_hdiod_score,
)

warnings.filterwarnings("ignore")

K_LIST = list(range(5, 101, 5))

GATED_LAM = 0.6
GATED_GAMMA = 2.0

RESULTS_DIR = "results"
OUT_ALL = os.path.join(RESULTS_DIR, "run_new_results.csv")
OUT_OVERALL = os.path.join(RESULTS_DIR, "run_new_overall_leaderboard.csv")


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def add_row(rows: List[Dict], dataset: str, method: str, param: str, k: int | None, auc: float):
    rows.append(
        {
            "dataset": dataset,
            "method": method,
            "param": param,
            "k": k if k is not None else np.nan,
            "auc": auc,
        }
    )


def run_one_dataset(name: str) -> pd.DataFrame:
    X, y = load_dataset(name)
    rows: List[Dict] = []

    for k in K_LIST:
        s = knn_distance_score_fixed(X, k=k)
        add_row(rows, name, "KNN", f"k={k}", k, safe_auc(y, s))

    for k in K_LIST:
        if k < 2:
            continue
        s = lof_score(X, k=k)
        add_row(rows, name, "LOF", f"k={k}", k, safe_auc(y, s))

    for k in K_LIST:
        if k < 2:
            continue
        s = cof_score(X, k=k)
        add_row(rows, name, "COF", f"k={k}", k, safe_auc(y, s))

    for k in K_LIST:
        if k < 2:
            continue
        s = ldof_score(X, k=k)
        add_row(rows, name, "LDOF", f"k={k}", k, safe_auc(y, s))

    for k in K_LIST:
        if k < 2:
            continue
        s = hdiod_score_paper(X, k=k)
        add_row(rows, name, "HDIOD", f"k={k}", k, safe_auc(y, s))

    for k in K_LIST:
        if k < 2:
            continue
        s = gated_hdiod_score(X, k=k, lam=GATED_LAM, gamma=GATED_GAMMA)
        add_row(rows, name, "GATED_HDIOD", f"k={k},lam={GATED_LAM},gamma={GATED_GAMMA}", k, safe_auc(y, s))


    df = pd.DataFrame(rows)
    df["auc"] = df["auc"].astype(float)
    return df


def print_overall_leaderboard(df_all: pd.DataFrame):
    print("\n" + "=" * 60)
    print("OVERALL LEADERBOARD (mean AUC across datasets)")
    print("=" * 60)

    grouped = (
        df_all.groupby(["method", "k"], as_index=False)["auc"]
        .mean()
        .sort_values(["k", "auc"], ascending=[True, False])
    )

    for k in sorted(grouped["k"].dropna().unique()):
        sub = grouped[grouped["k"] == k].copy()
        sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)

        print(f"\n[Overall | fixed k = {int(k)}]")
        print(f"{'rank':<5}{'method':<20}{'auc':<10}")
        for i, row in sub.iterrows():
            print(f"{i+1:<5}{row['method']:<20}{row['auc']:.4f}")


def make_overall_csv(df_all: pd.DataFrame) -> pd.DataFrame:
    leaderboard = (
        df_all.groupby(["method", "k"], as_index=False)["auc"]
        .mean()
        .sort_values(["k", "auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return leaderboard


def main():
    ensure_results_dir()

    dataset_names = list(DATASETS.keys())
    all_df_list = []

    print("Datasets to run:")
    for name in dataset_names:
        print(" -", name)

    for i, name in enumerate(dataset_names, 1):
        print(f"\n[{i}/{len(dataset_names)}] Running dataset: {name}")
        try:
            df_one = run_one_dataset(name)
            all_df_list.append(df_one)
        except Exception as e:
            print(f"  Failed on {name}: {e}")

    if not all_df_list:
        print("No results generated.")
        return

    df_all = pd.concat(all_df_list, ignore_index=True)
    df_all.to_csv(OUT_ALL, index=False)

    overall_df = make_overall_csv(df_all)
    overall_df.to_csv(OUT_OVERALL, index=False)

    print_overall_leaderboard(df_all)

    print("\nSaved:")
    print(" -", OUT_ALL)
    print(" -", OUT_OVERALL)


if __name__ == "__main__":
    main()