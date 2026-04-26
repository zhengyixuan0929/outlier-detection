import numpy as np
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pyod.models.hbos import HBOS
from pyod.models.cof import COF

def knn_distance_score(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    return distances[:, -1]

def lof_score(X, k=20):
    lof = LocalOutlierFactor(n_neighbors=k, novelty=False)
    lof.fit_predict(X)
    scores = -lof.negative_outlier_factor_
    return scores

def cof_score(X, k=10):
    model = COF(n_neighbors=k)
    model.fit(X)
    return model.decision_scores_

def ldof_score(X, k=10):
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    dk = distances[:, -1]

    mean_dist = distances.mean(axis=1)

    score = dk / (mean_dist + 1e-12)
    return score

def hbos_score(X):
    model = HBOS()
    model.fit(X)
    return model.decision_scores_