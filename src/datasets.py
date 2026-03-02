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
        "drop_cols": [0],  # 哪些列不能当作特征
        "label_col": 0,  # 那些列表示类别
        "anomaly": ["M"], #异常的值是什么
        "normalize": True, #是否进行归一化处理
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
# 将合成数据集添加到DATASETS
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
    # 读取文件
    read_kwargs = cfg.get("read_csv", {})
    df = pd.read_csv(path, **read_kwargs)

    # 删除无用列
    drop_cols = cfg.get("drop_cols", [])
    if drop_cols:
        df = df.drop(df.columns[drop_cols], axis=1)

    # 根据标签列构建 y
    label_col = cfg["label_col"]
    label = df.iloc[:, label_col]
    anomaly_values = cfg["anomaly"]
    y = label.isin(anomaly_values).astype(int).to_numpy()

    # 构建 X = 所有其余的数值特征
    X_df = df.drop(df.columns[label_col], axis=1)
    # 自动处理类别变量
    X_df = pd.get_dummies(X_df, drop_first=True)
    # 强制数值化：如果还有字符串，就会变成 NaN
    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    if X_df.isna().any().any():
        bad_cols = X_df.columns[X_df.isna().any()].tolist()
        raise ValueError(
            f"{name}: Non-numeric or missing values found after conversion. "
            f"Columns with NaN: {bad_cols} (check drop_cols/label_col)."
        )
    #把 pandas DataFrame 转成 numpy 数组
    X = X_df.to_numpy(dtype=float)

    #数据归一化
    if cfg.get("normalize", False):
        X = MinMaxScaler().fit_transform(X)

    return X, y


