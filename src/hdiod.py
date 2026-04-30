import numpy as np
from sklearn.neighbors import NearestNeighbors

# Calculate local kernel density
# Corresponding to the paper Eq. (3)
# ρ(x_i) = (1/k) * Σ exp( - d(x_i, x_j)^2 / 2 )
# The term (2π)^d constant is omitted because it will cancel out in the cof ratio
def local_kernel_density_paper(distances: np.ndarray) -> np.ndarray:
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


# HDIOD Main Function
# kNN construction
# Local Kernel Density Calculation
# High-Density Iteration (HDI)
# Extended Neighborhood EkNN
# centripetal outlier factor (cof)
def hdiod_score_paper(X: np.ndarray, k: int = 10) -> np.ndarray:

    X = np.asarray(X, dtype=float)

    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1]. Got k={k}, n={n}.")

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    rho = local_kernel_density_paper(distances)
    best_pos = np.argmax(rho[indices], axis=1)
    best_neighbor = indices[np.arange(n), best_pos]
    best_neighbor_rho = rho[best_neighbor]

    peaks = np.empty(n, dtype=int)

    # If the density of the current point is less than the maximum density among its neighbors,
    # then move to the neighbor with the maximum density and repeat this process.
    # Continue until the local density reaches its peak.
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

        # Along the iterative path, merge the kNN of all points along the path
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
        # KNN of all points on the merged path
        eknn = set()
        for t in traj:
            eknn.update(indices[t].tolist())
        eknn = np.fromiter(eknn, dtype=int)
        peaks[i] = eknn[np.argmax(rho[eknn])]

    cof = rho[peaks] / (rho + 1e-12)
    return cof