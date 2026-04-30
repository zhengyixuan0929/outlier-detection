import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from src.synthetic import SYNTHETICS


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

DATASETS = {
    #真实数据集
    "Breast Cancer": {
        "path": DATA_DIR / "breast+cancer+wisconsin+diagnostic" / "wdbc.csv",
        "read_csv": {"header": None,"skiprows": 1},
        "drop_cols": [0],  # Which columns cannot be used as features
        "label_col": 0,  # Those indicate categories
        "anomaly": ["M"], # What is the abnormal value
        "normalize": True, # Whether to perform normalization processing
    },

    "heart_failure": {
        "path": DATA_DIR / "heart+failure+clinical+records" / "heart_failure_clinical_records_dataset.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [1],
        "normalize": True,
    },

    "liver disorders": {
        "path": DATA_DIR / "liver+disorders" / "bupa.csv",
        "read_csv": {"header": None,"skiprows": 1},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [1],
        "normalize": True,
    },

    "Parkinsons": {
        "path": DATA_DIR / "parkinsons" / "parkinsons.csv",
        "read_csv": {"header": 0},
        "drop_cols": [0],
        "label_col": 16,
        "anomaly": [1],
        "normalize": True,
    },

    "Gallstone": {
        "path": DATA_DIR / "Gallstone" / "dataset-uci.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": 0,
        "anomaly": [1],
        "normalize": True,
    },

    "Hepatitis C Virus (HCV) for Egyptian patients": {
        "path": DATA_DIR / "HCV-Egy" / "HCV-Egy-Data.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [4],
        "normalize": True,
    },

    "Diabetic Retinopathy Debrecen": {
        "path": DATA_DIR / "Messidor_features" / "messidor_features.csv",
        "read_csv": {"header": None},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [1],
        "normalize": True,
    },

    "Thoracic Surgery": {
        "path": DATA_DIR / "ThoraricSurgery" / "ThoraricSurgery.csv",
        "read_csv": {"header": 0},
        "drop_cols": [0],
        "label_col": -1,
        "anomaly": ["T"],
        "normalize": True,
    },

    "Cervical Cancer": {
        "path": DATA_DIR / "Cervical Cancer" / "risk_factors_cervical_cancer.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [1],
        "normalize": True,
    },

    "Cardiotocography": {
        "path": DATA_DIR / "CTG" / "CTG.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": -1,
        "anomaly": [3],
        "normalize": True,
    },
}
# Add the synthetic dataset to DATASETS
for syn_name in SYNTHETICS.keys():
    DATASETS[syn_name] = {"type": "synthetic"}

def load_dataset(name: str):
    if name in SYNTHETICS:
        return SYNTHETICS[name]()

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]

    if "generator" in cfg:
        X, y = cfg["generator"]()
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        return X, y

    path = cfg["path"]

    read_kwargs = cfg.get("read_csv", {})
    df = pd.read_csv(path, **read_kwargs)


    drop_cols = cfg.get("drop_cols", [])
    if drop_cols:
        df = df.drop(df.columns[drop_cols], axis=1)

    label_col = cfg["label_col"]
    label = df.iloc[:, label_col]
    anomaly_values = cfg["anomaly"]
    y = label.isin(anomaly_values).astype(int).to_numpy()

    X_df = df.drop(df.columns[label_col], axis=1)
    X_df = pd.get_dummies(X_df, drop_first=True)
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(
            f"{name}: Non-numeric or missing values found after conversion. "
            f"Columns with NaN: {bad_cols} (check drop_cols/label_col)."
        )
    X = X_df.to_numpy(dtype=float)

    # Data Normalization
    if cfg.get("normalize", False):
        X = MinMaxScaler().fit_transform(X)

    return X, y


