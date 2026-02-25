from src.datasets import load_dataset, DATASETS
from src.baselines import (
    knn_distance_score,
    lof_score,
    cof_score,
    ldof_score,
    hbos_score,
)
from sklearn.metrics import roc_auc_score
import warnings

# 关闭 sklearn 重复值 warning
warnings.filterwarnings(
    action="ignore",
    message="Duplicate values are leading to incorrect results",
    category=UserWarning,
)

def main():
    ks = [30, 50, 70]

    for name in DATASETS.keys():
        print("=" * 60)
        print(f"Dataset: {name}")
        print("=" * 60)

        X, y = load_dataset(name)

        # ---------------- KNN ----------------
        print("KNN:")
        for k in ks:
            s = knn_distance_score(X, k=k)
            auc = roc_auc_score(y, s)
            print(f"  k={k:3d}  AUC={auc:.4f}")

        # ---------------- LOF ----------------
        print("LOF:")
        for k in ks:
            s = lof_score(X, k=k)
            auc = roc_auc_score(y, s)
            print(f"  k={k:3d}  AUC={auc:.4f}")

        # ---------------- COF ----------------
        print("COF:")
        for k in ks:
            s = cof_score(X, k=k)
            auc = roc_auc_score(y, s)
            print(f"  k={k:3d}  AUC={auc:.4f}")

        # ---------------- LDOF ----------------
        print("LDOF:")
        for k in ks:
            s = ldof_score(X, k=k)
            auc = roc_auc_score(y, s)
            print(f"  k={k:3d}  AUC={auc:.4f}")

        # ---------------- HBOS ----------------
        print("HBOS:")
        s = hbos_score(X)  # HBOS 不依赖 k
        auc = roc_auc_score(y, s)
        print(f"  AUC={auc:.4f}")

        print("=" * 60)


if __name__ == "__main__":
    main()