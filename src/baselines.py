import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pyod.models.hbos import HBOS
from pyod.models.cof import COF

def knn_distance_score(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    return distances[:, -1]  # 第k个邻居距离（越大越异常）

def lof_score(X, k=20):
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)  # unsupervised fit_predict 模式
    lof.fit_predict(X)  # 必须调用一次才会生成 negative_outlier_factor_
    # LOF anomaly score 返回值越大表示越异，所以 sklearn 的 negative_outlier_factor_ 取反
    scores = -lof.negative_outlier_factor_
    return scores

def cof_score(X, k=10):
    model = COF(n_neighbors=k)
    model.fit(X)
    return model.decision_scores_

def ldof_score(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    # 第 k 近邻距离
    dk = distances[:, -1]
    # 邻居之间平均距离
    mean_dist = distances.mean(axis=1)
    # LDOF
    score = dk / (mean_dist + 1e-12)
    return score

def hbos_score(X):
    model = HBOS()
    model.fit(X)
    return model.decision_scores_