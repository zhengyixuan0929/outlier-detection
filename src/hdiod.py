import numpy as np
from sklearn.neighbors import NearestNeighbors

def _local_kernel_density(distances, sigma=None):
    if sigma is None:
        sigma = np.mean(distances)  # 全局带宽
        sigma = max(sigma, 1e-12)
    w = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    return w.mean(axis=1)



def _high_density_peak(indices: np.ndarray, rho: np.ndarray) -> np.ndarray:

    #对每个点做高密度迭代：每次跳到邻居里 rho 最大的点，直到不再上升返回：peak_index[i] = 迭代终点（峰值点）的样本索引
    n, k = indices.shape
    peak = np.zeros(n, dtype=int)

    # 预先算每个点“邻居中 rho 最大对应的邻居索引”
    best_neighbor = indices[np.arange(n), np.argmax(rho[indices], axis=1)]
    best_neighbor_rho = rho[best_neighbor]

    for i in range(n):
        cur = i
        while best_neighbor_rho[cur] > rho[cur]:
            nxt = best_neighbor[cur]
            if nxt == cur:
                break
            cur = nxt
        peak[i] = cur
    return peak

def _peak_via_eknn(i, indices, rho, max_steps=1000):
    # 轨迹点
    traj = [i]
    cur = i
    steps = 0

    while steps < max_steps:
        nbrs = indices[cur]
        nxt = nbrs[np.argmax(rho[nbrs])]
        if rho[nxt] <= rho[cur]:
            break
        cur = nxt
        traj.append(cur)
        steps += 1

    # EkNN = 轨迹点的 kNN 并集
    eknn = set()
    for t in traj:
        eknn.update(indices[t].tolist())

    # 扩展邻域里 rho 最大的点
    eknn = np.fromiter(eknn, dtype=int)
    peak_ext = eknn[np.argmax(rho[eknn])]
    return peak_ext

def hdiod_score(X: np.ndarray, k: int = 10) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)

    rho = _local_kernel_density(distances, sigma=None)
    n = X.shape[0]
    peaks = np.zeros(n, dtype=int)
    for i in range(n):
        peaks[i] = _peak_via_eknn(i, indices, rho)

    cof = rho[peaks] / (rho + 1e-12)
    return cof


