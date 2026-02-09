from src.datasets import load_dataset, DATASETS
from src.baselines import knn_distance_score, lof_score, iforest_score
from sklearn.metrics import roc_auc_score
import warnings

# 关闭 sklearn 的 UserWarning
warnings.filterwarnings(
    "ignore",
    message="Duplicate values are leading to incorrect results",
    category=UserWarning,
)

def main():
    ks = [4, 10, 20, 50]

    for name in DATASETS.keys():
        print("=" * 60)
        print(f"Dataset: {name}")
        print("-" * 60)

        X, y = load_dataset(name)

        # ---------- KNN ----------
        print("KNN:")
        for k in ks:
            s_knn = knn_distance_score(X, k=k)
            auc_knn = roc_auc_score(y, s_knn)
            print(f"  k={k:3d}  AUC={auc_knn:.4f}")

        # ---------- LOF ----------
        print("LOF:")
        for k in ks:
            if k < 5:   # LOF 跳过小 k
                continue
            s_lof = lof_score(X, k=k)
            auc_lof = roc_auc_score(y, s_lof)
            print(f"  k={k:3d}  AUC={auc_lof:.4f}")

        # ---------- Isolation Forest ----------
        print("IForest:")
        for n_estimators in [100, 200]:   # 先用两档够了（你也可以只用200）
            s = iforest_score(X, n_estimators=n_estimators, random_state=42)
            print(f"  trees={n_estimators:3d}  AUC={roc_auc_score(y, s):.4f}")


    print("=" * 60)

if __name__ == "__main__":
    main()
