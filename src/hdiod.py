import numpy as np
from sklearn.neighbors import NearestNeighbors

#计算局部核密度
def local_kernel_density_paper(distances: np.ndarray) -> np.ndarray:
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


def hdiod_score_paper(X: np.ndarray, k: int = 10) -> np.ndarray:

    X = np.asarray(X, dtype=float)

    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1]. Got k={k}, n={n}.")

    # ---- kNN (exclude self) ----
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]  # drop self
    indices = indices[:, 1:]
    # ---- rho ----
    rho = local_kernel_density_paper(distances)
    # ---- high-density iteration ----
    best_pos = np.argmax(rho[indices], axis=1)
    best_neighbor = indices[np.arange(n), best_pos]
    best_neighbor_rho = rho[best_neighbor]

    peaks = np.empty(n, dtype=int)

    for i in range(n):
        cur = i
        steps = 0
        while best_neighbor_rho[cur] > rho[cur]:
            nxt = best_neighbor[cur]
            if nxt == cur:
                break
            cur = nxt
            steps += 1
            if steps > n:
                break

        # collect trajectory
        traj = [i]
        cur2 = i
        steps2 = 0
        while best_neighbor_rho[cur2] > rho[cur2]:
            nxt2 = best_neighbor[cur2]
            if nxt2 == cur2:
                break
            cur2 = nxt2
            traj.append(cur2)
            steps2 += 1
            if steps2 > n:
                break
        eknn = set()
        for t in traj:
            eknn.update(indices[t].tolist())
        eknn = np.fromiter(eknn, dtype=int)
        peaks[i] = eknn[np.argmax(rho[eknn])]

    cof = rho[peaks] / (rho + 1e-12)
    return cof