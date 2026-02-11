from src.datasets import load_dataset, DATASETS
from src.hdiod import hdiod_score
from sklearn.metrics import roc_auc_score

def main():
    ks = [10, 20, 50]

    for name in DATASETS.keys():
        print("=" * 60)
        print(f"Dataset: {name}")
        print("-" * 60)

        X, y = load_dataset(name)

        for k in ks:
            scores = hdiod_score(X, k=k)
            auc = roc_auc_score(y, scores)
            print(f"  k={k:3d}  AUC={auc:.4f}")

    print("=" * 60)

    auc = roc_auc_score(y, scores)
    auc_flip = roc_auc_score(y, -scores)
    print("AUC:", auc, " AUC(-score):", auc_flip)

if __name__ == "__main__":
    main()
