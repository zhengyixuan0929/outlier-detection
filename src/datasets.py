import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "Data"

DATASETS = {
    "wdbc": {
        "path": DATA_DIR / "breast+cancer+wisconsin+diagnostic" / "wdbc.csv",
        "read_csv": {"header": None,"skiprows": 1},
        "drop_cols": [0],  # 哪些列不能当作特征
        "label_col": 0,  # 那些列表示类别
        "anomaly": ["M"], #异常的值是什么

    },

    "diabetes_binary": {
        "path": DATA_DIR / "archive" / "diabetes_binary_health_indicators_BRFSS2015.csv",
        "read_csv": {"header": 0},
        "drop_cols": [],
        "label_col": 0,
        "anomaly": [1],
    },


}

def load_dataset(name: str):
    """
    Load dataset by config in DATASETS.
    Returns X (float ndarray) and y (0/1 ndarray).
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    cfg = DATASETS[name]
    path = cfg["path"]

    # 1) read
    read_kwargs = cfg.get("read_csv", {})
    df = pd.read_csv(path, **read_kwargs)

    # 2) drop useless cols (e.g., id / name)
    drop_cols = cfg.get("drop_cols", [])
    if drop_cols:
        df = df.drop(df.columns[drop_cols], axis=1)

    # 3) build y from label column
    label_col = cfg["label_col"]
    label = df.iloc[:, label_col]
    anomaly_values = cfg["anomaly"]
    y = label.isin(anomaly_values).astype(int).to_numpy()

    # 4) build X = all remaining numeric features
    X_df = df.drop(df.columns[label_col], axis=1)

    # 强制数值化：如果还有字符串，会变成 NaN（便于定位）
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    # 如果你想严格：发现 NaN 就报错（推荐，早点发现脏数据）
    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(
            f"{name}: Non-numeric or missing values found after conversion. "
            f"Columns with NaN: {bad_cols} (check drop_cols/label_col)."
        )

    X = X_df.to_numpy(dtype=float)
    return X, y


