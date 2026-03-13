import os
import warnings
from typing import List, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.datasets import load_dataset, DATASETS
from src.hdiod import hdiod_score_paper
from src.baselines import knn_distance_score, lof_score, cof_score, ldof_score, hbos_score
from src.new_hdiod import gated_hdiod_score   # 新增

# =============== config you can tune ===============
K_LIST = [20, 60, 100]
# K_LIST = list(range(5, 101, 5))

RESULTS_DIR = "results"
OUT_ALL = os.path.join(RESULTS_DIR, "run_all_results.csv")
OUT_OVERALL = os.path.join(RESULTS_DIR, "run_all_overall_leaderboard.csv")
OUT_PER_DATASET_LB = os.path.join(RESULTS_DIR, "run_all_per_dataset_leaderboards.csv")

# ---- GATED_HDIOD params ----
GATED_LAM = 0.6
GATED_GAMMA = 2.0
# ================================================


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def safe_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Return AUC; if y has only one class, return NaN."""
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

    # ---------- KNN ----------
    for k in K_LIST:
        s = knn_distance_score(X, k=k)
        add_row(rows, name, "KNN", f"k={k}", k, safe_auc(y, s))

    # ---------- LOF ----------
    for k in K_LIST:
        if k < 2:
            continue
        s = lof_score(X, k=k)
        add_row(rows, name, "LOF", f"k={k}", k, safe_auc(y, s))

    # ---------- COF ----------
    for k in K_LIST:
        if k < 2:
            continue
        s = cof_score(X, k=k)
        add_row(rows, name, "COF", f"k={k}", k, safe_auc(y, s))

    # ---------- LDOF ----------
    for k in K_LIST:
        if k < 2:
            continue
        s = ldof_score(X, k=k)
        add_row(rows, name, "LDOF", f"k={k}", k, safe_auc(y, s))

    # ---------- HBOS (no k) ----------
    s = hbos_score(X)
    add_row(rows, name, "HBOS", "default", None, safe_auc(y, s))

    # ---------- HDIOD ----------
    for k in K_LIST:
        if k < 2:
            continue
        s = hdiod_score_paper(X, k=k)
        add_row(rows, name, "HDIOD", f"k={k}", k, safe_auc(y, s))

    # ---------- GATED_HDIOD ----------
    for k in K_LIST:
        if k < 2:
            continue
        s = gated_hdiod_score(X, k=k, lam=GATED_LAM, gamma=GATED_GAMMA)
        add_row(
            rows,
            name,
            "GATED_HDIOD",
            f"k={k},lam={GATED_LAM},gamma={GATED_GAMMA}",
            k,
            safe_auc(y, s),
        )

    df = pd.DataFrame(rows)
    df["auc"] = df["auc"].astype(float)
    return df


def print_all_results_table(df_one: pd.DataFrame):
    show = df_one.copy()
    show["auc"] = show["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    show = show.sort_values(["method", "k", "param"], na_position="last")
    print(show[["method", "param", "auc"]].to_string(index=False))


def leaderboard_fixed_k(df_one: pd.DataFrame, k: int) -> pd.DataFrame:
    """Leaderboard for methods that share k."""
    methods_with_k = ["KNN", "LOF", "COF", "LDOF", "HDIOD", "GATED_HDIOD"]

    lb = df_one[(df_one["k"] == k) & (df_one["method"].isin(methods_with_k))].copy()
    lb = lb.sort_values("auc", ascending=False).reset_index(drop=True)
    lb.insert(0, "rank", np.arange(1, len(lb) + 1))
    lb["auc"] = lb["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return lb[["rank", "method", "param", "auc"]]


def extras_no_k(df_one: pd.DataFrame) -> pd.DataFrame:
    """Show HBOS + IForest (trees=...) results."""
    extra = df_one[df_one["k"].isna() & df_one["method"].isin(["HBOS", "IForest"])].copy()
    extra = extra.sort_values(["method", "auc"], ascending=[True, False])
    extra["auc"] = extra["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return extra[["method", "param", "auc"]]


def overall_leaderboard(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall leaderboard:
    - For each fixed k: mean AUC across datasets for k-based methods
    - For HBOS/IForest: mean AUC across datasets (no k)
    """
    rows = []

    # k-based methods
    methods_with_k = ["KNN", "LOF", "COF", "LDOF", "HDIOD", "GATED_HDIOD"]
    for k in K_LIST:
        part = all_df[(all_df["k"] == k) & (all_df["method"].isin(methods_with_k))].copy()
        g = part.groupby("method", as_index=False)["auc"].mean()
        g["k"] = k
        rows.append(g)

    # no-k methods
    part2 = all_df[all_df["k"].isna() & (all_df["method"].isin(["HBOS", "IForest"]))].copy()
    if len(part2) > 0:
        g2 = part2.groupby("method", as_index=False)["auc"].mean()
        g2["k"] = np.nan
        rows.append(g2)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["method", "auc", "k"])
    out["auc"] = out["auc"].astype(float)
    out["auc_round"] = out["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 4))
    return out


def main():
    warnings.filterwarnings("ignore")

    ensure_results_dir()

    all_results = []
    per_dataset_lb_rows = []

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
        print_all_results_table(df)

        # 2) 固定 k 排行榜
        for k in K_LIST:
            print("\n" + "-" * 70)
            print(f"[Leaderboard | fixed k = {k}]")
            print("-" * 70)
            lb = leaderboard_fixed_k(df, k=k)
            print(lb.to_string(index=False))

            tmp = lb.copy()
            tmp.insert(0, "dataset", name)
            tmp.insert(2, "k", k)
            per_dataset_lb_rows.append(tmp)

        # 3) no-k methods results
        extra = extras_no_k(df)
        if len(extra) > 0:
            print("\n" + "-" * 70)
            print("[Extra methods | no k]")
            print("-" * 70)
            print(extra.to_string(index=False))

    all_df = pd.concat(all_results, ignore_index=True)

    # Save full results
    all_df_out = all_df.copy()
    all_df_out["auc"] = all_df_out["auc"].map(lambda x: np.nan if pd.isna(x) else round(float(x), 6))
    all_df_out.to_csv(OUT_ALL, index=False)
    print("\nSaved ->", OUT_ALL)

    # Save per-dataset leaderboards
    if per_dataset_lb_rows:
        lb_df = pd.concat(per_dataset_lb_rows, ignore_index=True)
        lb_df.to_csv(OUT_PER_DATASET_LB, index=False)
        print("Saved ->", OUT_PER_DATASET_LB)

    # Overall leaderboard
    overall = overall_leaderboard(all_df)

    print("\n" + "=" * 70)
    print("OVERALL LEADERBOARD (mean AUC across datasets)")
    print("=" * 70)

    # Print fixed-k overall
    for k in K_LIST:
        sub = overall[overall["k"] == k].copy()
        sub = sub.sort_values("auc", ascending=False).reset_index(drop=True)
        sub.insert(0, "rank", np.arange(1, len(sub) + 1))
        print(f"\n[Overall | fixed k = {k}]")
        print(sub[["rank", "method", "auc_round"]].rename(columns={"auc_round": "auc"}).to_string(index=False))

    # Print no-k overall
    sub2 = overall[overall["k"].isna()].copy()
    if len(sub2) > 0:
        sub2 = sub2.sort_values("auc", ascending=False).reset_index(drop=True)
        sub2.insert(0, "rank", np.arange(1, len(sub2) + 1))
        print("\n[Overall | no k methods]")
        print(sub2[["rank", "method", "auc_round"]].rename(columns={"auc_round": "auc"}).to_string(index=False))

    # Save overall
    overall_out = overall.copy()
    overall_out = overall_out.rename(columns={"auc_round": "auc_rounded"})
    overall_out.to_csv(OUT_OVERALL, index=False)
    print("\nSaved ->", OUT_OVERALL)


if __name__ == "__main__":
    main()