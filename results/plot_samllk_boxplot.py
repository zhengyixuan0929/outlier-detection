from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # ===== paths =====
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    out_dir = script_path.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_csv_candidates = [
        project_root / "scripts" / "results" / "run_new_results.csv",
        project_root / "scripts" / "results" / "run_all_results_k_5_100_step5.csv",
    ]

    in_csv = None
    for p in in_csv_candidates:
        if p.exists():
            in_csv = p
            break

    if in_csv is None:
        raise FileNotFoundError(
            "Cannot find results csv. Tried:\n" + "\n".join(str(x) for x in in_csv_candidates)
        )

    print(f"[OK] Input : {in_csv}")
    print(f"[OK] Output: {out_dir}")

    df = pd.read_csv(in_csv)

    required = {"dataset", "method", "k", "auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}. Found columns: {list(df.columns)}")

    df["dataset"] = df["dataset"].astype(str)
    df["method"] = df["method"].astype(str)

    # ===== keep only synthetic datasets =====
    synth_df = df[df["dataset"].str.lower().str.startswith("syn_")].copy()

    if synth_df.empty:
        raise ValueError("No synthetic datasets found (expect dataset name starts with 'syn_').")

    synth_df = synth_df[synth_df["k"].notna()].copy()
    synth_df["k"] = synth_df["k"].astype(int)
    synth_df["auc"] = synth_df["auc"].astype(float)

    # ===== mean AUC across all synthetic datasets at each k =====
    g = synth_df.groupby(["method", "k"], as_index=False)["auc"].mean()
    g = g.rename(columns={"auc": "mean_auc_over_synth"}).sort_values(["method", "k"])

    out_table = out_dir / "synthetic_mean_auc_by_k.csv"
    g.to_csv(out_table, index=False)
    print(f"[OK] Saved table: {out_table}")

    # ===== method order =====
    methods_order = ["KNN", "GATED_HDIOD", "HDIOD", "LOF", "COF", "LDOF"]
    methods = [m for m in methods_order if m in g["method"].unique()] + \
              [m for m in sorted(g["method"].unique()) if m not in methods_order]

    k_values = sorted(g["k"].unique().tolist())
    print(f"[INFO] k values in file: {k_values}")

    # ===== boxplot data =====
    box_data = []
    for m in methods:
        vals = g[g["method"] == m].sort_values("k")["mean_auc_over_synth"].to_numpy()
        box_data.append(vals)

    plt.figure(figsize=(8, 5))
    plt.boxplot(box_data, labels=methods, showfliers=True)
    plt.ylabel("Mean AUC across synthetic datasets (each point = one k)")
    plt.title("Synthetic datasets: AUC distribution across k")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out_png = out_dir / "synthetic_auc_boxplot_over_k.png"
    plt.savefig(out_png, dpi=220)
    plt.close()
    print(f"[OK] Saved figure: {out_png}")


if __name__ == "__main__":
    main()