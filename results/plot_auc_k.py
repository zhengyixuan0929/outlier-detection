import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.datasets import load_dataset, DATASETS
from src.baselines import knn_distance_score, lof_score, cof_score, ldof_score
from src.hdiod import hdiod_score
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# 你可以改成 range(5, 101, 5) 这种更密一点
K_LIST = list(range(5, 101, 5))

METHODS = {
    "KNN": knn_distance_score,
    "LOF": lof_score,
    "COF": cof_score,
    "LDOF": ldof_score,
    "HDIOD": hdiod_score,
}


def safe_auc(y, scores):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return roc_auc_score(y, scores)


def compute_auc_table(dataset_names):
    rows = []
    for ds in dataset_names:
        X, y = load_dataset(ds)
        for k in K_LIST:
            for mname, fn in METHODS.items():
                # LOF 要求 k>=2（你这里最小是5没问题）
                scores = fn(X, k=k)
                auc = safe_auc(y, scores)
                rows.append({"dataset": ds, "method": mname, "k": k, "auc": auc})
    return pd.DataFrame(rows)


def plot_grid(df, dataset_names, ncols=3, out_png="auc_vs_k_grid.png"):
    n = len(dataset_names)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), dpi=150)
    axes = np.array(axes).reshape(-1)

    for i, ds in enumerate(dataset_names):
        ax = axes[i]
        sub = df[df["dataset"] == ds]

        for mname in METHODS.keys():
            s = sub[sub["method"] == mname].sort_values("k")
            ax.plot(s["k"], s["auc"], marker="o", linewidth=1, markersize=2, label=mname)

        ax.set_title(ds)
        ax.set_xlabel("k")
        ax.set_ylabel("AUC")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8)

    # 多余空子图关掉
    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved figure -> {out_png}")


def main():
    warnings.filterwarnings("ignore")

    dataset_names = list(DATASETS.keys())

    df = compute_auc_table(dataset_names)
    csv_path = RESULTS_DIR / "auc_vs_k_results.csv"
    png_path = RESULTS_DIR / "auc_vs_k_grid.png"
    df.to_csv(csv_path, index=False)
    print(f"Saved raw results -> {csv_path}")

    plot_grid(df, dataset_names, ncols=3, out_png=str(png_path))

if __name__ == "__main__":
    main()