import numpy as np
from sklearn.neighbors import NearestNeighbors


def _minmax_score(s: np.ndarray) -> np.ndarray:
    """
    Min-Max normalize scores to [0, 1].
    """
    s = np.asarray(s, dtype=float)
    smin = np.min(s)
    smax = np.max(s)
    if smax - smin < 1e-12:
        return np.zeros_like(s, dtype=float)
    return (s - smin) / (smax - smin)


def local_kernel_density(distances: np.ndarray) -> np.ndarray:
    """
    Paper-style local kernel density:
    rho(x_i) = (1/k) * sum exp( -d(x_i, x_j)^2 / 2 )
    Constant term is omitted since it cancels in the ratio.
    """
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


def knn_distance_score_fixed(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Fair KNN score:
    use k+1 neighbors, then remove self, finally take the true k-th NN distance.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1]. Got k={k}, n={n}.")

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
        raise ValueError(f"k must be in [1, n-1]. Got k={k}, n={n}.")

    # build kNN graph
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    # remove self
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # local kernel density
    rho = local_kernel_density(distances)

    # best neighbor = the densest one among kNN
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


def new_hdiod_score(
    X: np.ndarray,
    k: int = 10,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    New HDIOD (Scheme B: HDIOD + KNN hybrid)

    final_score = alpha * norm(HDIOD) + (1 - alpha) * norm(KNN)

    Parameters
    ----------
    X : ndarray
        Input samples
    k : int
        Number of neighbors
    alpha : float
        Weight for HDIOD part. Recommend 0.6 ~ 0.8

    Returns
    -------
    ndarray
        Outlier scores, larger means more anomalous
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    hdiod = hdiod_score_base(X, k=k)
    knn = knn_distance_score_fixed(X, k=k)

    hdiod_norm = _minmax_score(hdiod)
    knn_norm = _minmax_score(knn)

    final_score = alpha * hdiod_norm + (1.0 - alpha) * knn_norm
    return final_score