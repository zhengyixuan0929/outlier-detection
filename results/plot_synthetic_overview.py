import os
import numpy as np
import matplotlib.pyplot as plt

from src.synthetic import SYNTHETICS


# 你想展示的 10 个数据集（按你 synthetic.py 的名字）
SYN_NAMES = [
    "syn_gauss_uo",
    "syn_multi_blobs_uo",
    "syn_two_density",
    "syn_blocks",
    "syn_vshape",
    "syn_moons",
    "syn_spiral",
    "syn_double_spiral",
    "syn_sine",
    "syn_two_lines",
]

# 展示名称（图下面标注用，你也可以改成 DS01...）
DISPLAY_NAMES = {
    "syn_gauss_uo": "DS01",
    "syn_multi_blobs_uo": "DS02",
    "syn_two_density": "DS03",
    "syn_blocks": "DS04",
    "syn_vshape": "DS05",
    "syn_moons": "DS06",
    "syn_spiral": "DS07",
    "syn_double_spiral": "DS08",
    "syn_sine": "DS09",
    "syn_two_lines": "DS10",
}


def _ensure_results_dir():
    os.makedirs("results", exist_ok=True)


def _get_dataset_stats(name: str):
    X, y = SYNTHETICS[name]()  # 默认参数生成
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    n = X.shape[0]
    d = X.shape[1]
    n_out = int((y == 1).sum())
    return X, y, n, d, n_out

def plot_synthetic_overview(save_path="results/synthetic_overview.png", dpi=300):
    _ensure_results_dir()

    rows = []
    data_cache = []

    for name in SYN_NAMES:
        X, y, n, d, n_out = _get_dataset_stats(name)
        rows.append([DISPLAY_NAMES.get(name, name), n, d, n_out])
        data_cache.append((name, X, y))

    fig = plt.figure(figsize=(18, 10))

    # -------- 表格区域 --------
    ax_table = fig.add_axes([0.05, 0.72, 0.90, 0.25])  # 手动控制位置
    ax_table.axis("off")

    col_labels = ["Dataset", "Instances", "Dims", "Outliers"]

    table = ax_table.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.3)

    # -------- 子图区域 --------
    gs = fig.add_gridspec(
        2, 5,
        left=0.05,
        right=0.95,
        bottom=0.08,
        top=0.68,
        wspace=0.25,
        hspace=0.35
    )

    for idx, (name, X, y) in enumerate(data_cache):
        r = idx // 5
        c = idx % 5
        ax = fig.add_subplot(gs[r, c])

        normal = (y == 0)
        outlier = (y == 1)

        ax.scatter(X[normal, 0], X[normal, 1], s=6, alpha=0.8)
        ax.scatter(X[outlier, 0], X[outlier, 1], s=12, alpha=0.9)

        ax.set_xticks([])
        ax.set_yticks([])

        # 强制正方形
        ax.set_box_aspect(1)

        ax.set_title(f"({chr(ord('a') + idx)}) {DISPLAY_NAMES.get(name, name)}", fontsize=10)

        ax.text(
            0.02, 0.98,
            f"n={X.shape[0]}\nout={int(outlier.sum())}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )

    # -------- legend --------
    handles = [
        plt.Line2D([], [], marker='o', linestyle='', markersize=6),
        plt.Line2D([], [], marker='o', linestyle='', markersize=6)
    ]
    labels = ["Normal", "Outlier"]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=11
    )

    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {save_path}")


if __name__ == "__main__":
    plot_synthetic_overview()