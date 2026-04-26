import numpy as np
from sklearn.neighbors import NearestNeighbors


def _minmax_score(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    smin = np.min(s)
    smax = np.max(s)
    if smax - smin < 1e-12:
        return np.zeros_like(s, dtype=float)
    return (s - smin) / (smax - smin)


def knn_distance_score_fixed(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Fair KNN distance score:
    use k+1 neighbors and remove self, then return the true k-th NN distance.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, _ = nn.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]  # remove self
    return distances[:, -1]


def local_kernel_density(distances: np.ndarray) -> np.ndarray:
    """
    rho(x_i) = (1/k) * sum exp( -d(x_i, x_j)^2 / 2 )
    Constant factor is omitted because it cancels in ratio-type scores.
    """
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


def hdiod_score_with_custom_k(X: np.ndarray, k_density: int = 10) -> np.ndarray:
    """
    HDIOD core score with custom neighborhood size for density estimation
    and high-density iteration.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if k_density <= 0 or k_density >= n:
        raise ValueError(f"k_density must be in [1, n-1], got k_density={k_density}, n={n}")

    nn = NearestNeighbors(n_neighbors=k_density + 1, metric="euclidean")
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
    GATED_HDIOD:
        score = HDIOD_k * (1 + lam * gate_k)
    where gate_k is based on normalized kNN distance under the same k.
    """
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    hdiod = hdiod_score_with_custom_k(X, k_density=k)
    knn_dist = knn_distance_score_fixed(X, k=k)

    gate = _minmax_score(knn_dist) ** gamma
    score = hdiod * (1.0 + lam * gate)
    return score


def eg_hdiod_score(
    X: np.ndarray,
    k: int = 10,
    k_expand_mode: str = "double",
    expand_offset: int = 10,
    lam: float = 0.6,
    gamma: float = 2.0,
) -> np.ndarray:
    """
    EG_HDIOD (Expanded Gated HDIOD):
        score = HDIOD_{k_exp} * (1 + lam * gate_k)
    where:
    - HDIOD uses an expanded neighborhood k_exp
    - gate still uses the original small k
    Parameters
    ----------
    X : ndarray
        Input samples.
    k : int
        Original neighborhood size used for gate / KNN distance.
    k_expand_mode : {"double", "plus"}
        Strategy to generate k_exp:
        - "double": k_exp = 2 * k
        - "plus":   k_exp = k + expand_offset
    expand_offset : int
        Used only when k_expand_mode == "plus".
    lam : float
        Gate enhancement strength.
    gamma : float
        Gate sharpness.
    Returns
    -------
    ndarray
        Outlier scores, larger means more anomalous.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1], got k={k}, n={n}")
    if lam < 0:
        raise ValueError(f"lam must be >= 0, got {lam}")
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    if k_expand_mode == "double":
        k_exp = 2 * k
    elif k_expand_mode == "plus":
        k_exp = k + expand_offset
    else:
        raise ValueError("k_expand_mode must be 'double' or 'plus'")

    # keep k_exp valid
    k_exp = min(max(k_exp, k + 1), n - 1)
    hdiod_expanded = hdiod_score_with_custom_k(X, k_density=k_exp)
    knn_dist_smallk = knn_distance_score_fixed(X, k=k)
    gate = _minmax_score(knn_dist_smallk) ** gamma
    score = hdiod_expanded * (1.0 + lam * gate)
    return score