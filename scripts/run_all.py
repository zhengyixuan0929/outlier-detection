import warnings
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets import load_dataset, DATASETS
from src.baselines import knn_distance_score, lof_score, iforest_score
from src.hdiod import hdiod_score

# =============== config you can tune ===============
K_LIST = [10, 20, 50]
IF_FIXED_TREES = 200
# ====================================================


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def run_one_dataset(name: str) -> pd.DataFrame:
    X, y = load_dataset(name)

    rows = []

    # KNN
    for k in K_LIST:
        s = knn_distance_score(X, k=k)
        rows.append({"dataset": name, "method": "KNN", "param": f"k={k}", "k": k, "auc": safe_auc(y, s)})

    # LOF
    for k in K_LIST:
        if k < 2:
            continue
        s = lof_score(X, k=k)
        rows.append({"dataset": name, "method": "LOF", "param": f"k={k}", "k": k, "auc": safe_auc(y, s)})

    # HDIOD
    for k in K_LIST:
        if k < 2:
            continue
        s = hdiod_score(X, k=k)
        rows.append({"dataset": name, "method": "HDIOD", "param": f"k={k}", "k": k, "auc": safe_auc(y, s)})

    # IForest（固定 trees，方便跟 k 排行榜一起比较）
    s = iforest_score(X, n_estimators=IF_FIXED_TREES, random_state=42)
    rows.append({"dataset": name, "method": "IForest", "param": f"trees={IF_FIXED_TREES}", "k": np.nan, "auc": safe_auc(y, s)})

    df = pd.DataFrame(rows)
    df["auc"] = df["auc"].astype(float)
    return df


def leaderboard_fixed_k(df_one_dataset: pd.DataFrame, k: int) -> pd.DataFrame:
    part = df_one_dataset[(df_one_dataset["k"] == k) & (df_one_dataset["method"].isin(["KNN", "LOF", "HDIOD"]))].copy()
    iforest = df_one_dataset[df_one_dataset["method"] == "IForest"].copy()
    lb = pd.concat([part, iforest], ignore_index=True)

    lb = lb.sort_values("auc", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    lb["auc"] = lb["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return lb[["rank", "method", "param", "auc"]]


def print_full_table(df: pd.DataFrame):
    show = df.copy()
    show["auc"] = show["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    show = show.sort_values(["method", "k"], na_position="last")
    print(show[["method", "param", "auc"]].to_string(index=False))


def main():
    warnings.filterwarnings("ignore")

    all_results = []

    print("=" * 70)
    print("RUN ALL: Full results + Leaderboards (fixed k)")
    print("=" * 70)

    for name in DATASETS.keys():
        print("\n" + "=" * 70)
        print(f"Dataset: {name}")
        print("=" * 70)

        df = run_one_dataset(name)
        all_results.append(df)

        # 1) 全部结果
        print("\n[All results]")
        print_full_table(df)

        # 2) 固定 k 的排行榜（三次）
        for k in K_LIST:
            print("\n" + "-" * 70)
            print(f"[Leaderboard | fixed k = {k}]")
            print("-" * 70)
            lb = leaderboard_fixed_k(df, k=k)
            print(lb.to_string(index=False))

    # 汇总保存
    all_df = pd.concat(all_results, ignore_index=True)

    all_df_out = all_df.copy()
    all_df_out["auc"] = all_df_out["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 6))
    all_df_out.to_csv("results_run_all_full.csv", index=False)

    overall_rows = []
    for k in K_LIST:
        part = all_df[(all_df["k"] == k) & (all_df["method"].isin(["KNN", "LOF", "HDIOD"]))].copy()
        iforest = all_df[all_df["method"] == "IForest"].copy()  # 固定trees
        comb = pd.concat([part, iforest], ignore_index=True)

        g = comb.groupby("method", as_index=False)["auc"].mean()
        g["k"] = k
        overall_rows.append(g)

    overall_df = pd.concat(overall_rows, ignore_index=True)
    overall_df["auc"] = overall_df["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    overall_df = overall_df.sort_values(["k", "auc"], ascending=[True, False])

    print("\n" + "=" * 70)
    print("OVERALL LEADERBOARD (mean AUC across datasets, fixed k)")
    print("=" * 70)
    for k in K_LIST:
        sub = overall_df[overall_df["k"] == k].copy()
        sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))
        print(f"\n[Overall | fixed k = {k}]")
        print(sub[["rank", "method", "auc"]].to_string(index=False))

    overall_df.to_csv("results_run_all_overall_leaderboard.csv", index=False)

    print("\nSaved:")
    print(" - results_run_all_full.csv")
    print(" - results_run_all_overall_leaderboard.csv")


if __name__ == "__main__":
    main()
