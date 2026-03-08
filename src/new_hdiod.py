import numpy as np
from sklearn.neighbors import NearestNeighbors


def _minmax_score(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    smin = np.min(s)
    smax = np.max(s)
    if smax - smin < 1e-12:
        return np.zeros_like(s, dtype=float)
    return (s - smin) / (smax - smin)


def local_kernel_density(distances: np.ndarray) -> np.ndarray:
    """
    rho(x_i) = (1/k) * sum exp( -d(x_i, x_j)^2 / 2 )
    Gaussian constant term omitted since it cancels in ratio-type scores.
    """
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


def knn_distance_score_fixed(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Fair KNN distance score:
    use k+1 neighbors and then remove self, so the last column is the true k-th NN distance.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]  # remove self
    return distances[:, -1]


def hdiod_score_base(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Original HDIOD-style score:
    cof(x) = rho(peak(x)) / rho(x)
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    distances = distances[:, 1:]  # remove self
    indices = indices[:, 1:]

    rho = local_kernel_density(distances)

    best_pos = np.argmax(rho[indices], axis=1)
    best_neighbor = indices[np.arange(n), best_pos]
    best_neighbor_rho = rho[best_neighbor]

    peaks = np.empty(n, dtype=int)

    for i in range(n):
        traj = [i]
        cur = i
        steps = 0

        while best_neighbor_rho[cur] > rho[cur]:
            nxt = best_neighbor[cur]
            if nxt == cur:
                break
            cur = nxt
            traj.append(cur)
            steps += 1
            if steps > n:
                break

        eknn = set()
        for t in traj:
            eknn.update(indices[t].tolist())

        eknn = np.fromiter(eknn, dtype=int)
        if len(eknn) == 0:
            peaks[i] = i
        else:
            peaks[i] = eknn[np.argmax(rho[eknn])]

    cof = rho[peaks] / (rho + 1e-12)
    return cof


def gated_hdiod_score(
    X: np.ndarray,
    k: int = 10,
    lam: float = 0.6,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    Gated HDIOD:
        score = hdiod * (1 + lam * gate)

    gate is constructed from normalized kNN distance:
        gate = (norm_knn_distance) ** gamma

    Parameters
    ----------
    X : ndarray
        Input samples.
    k : int
        Number of neighbors.
    lam : float
        Enhancement strength. Typical values: 0.3 ~ 1.0
    gamma : float
        Gate sharpness. gamma > 1 makes enhancement focus more on large-distance points.

    Returns
    -------
    ndarray
        Outlier scores, larger means more anomalous.
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    hdiod = hdiod_score_base(X, k=k)
    knn_dist = knn_distance_score_fixed(X, k=k)

    gate = _minmax_score(knn_dist) ** gamma
    score = hdiod * (1.0 + lam * gate)
    return score


def adaptive_gated_hdiod_score(
    X: np.ndarray,
    k: int = 10,
    lam_small: float = 0.9,
    lam_large: float = 0.25,
    k_switch: int = 30,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    Adaptive Gated HDIOD:
    stronger distance enhancement for small k, weaker for large k.

    If k <= k_switch -> use lam_small
    else -> use lam_large
    """
    lam = lam_small if k <= k_switch else lam_large
    return gated_hdiod_score(X, k=k, lam=lam, gamma=gamma)