import numpy as np
from sklearn.neighbors import NearestNeighbors

#计算局部核密度
#对应论文 Eq.(3)
# ρ(x_i) = (1/k) * Σ exp( - d(x_i, x_j)^2 / 2 )
# 省略了 (2π)^d 常数项，因为在 cof 比值中会相互抵消
def local_kernel_density_paper(distances: np.ndarray) -> np.ndarray:
    return np.exp(-(distances ** 2) / 2.0).mean(axis=1)


# HDIOD 主函数
# 实现论文中的
# kNN 构建
# 局部核密度计算
# 高密度迭代 (High-Density Iteration)
# 扩展邻域 EkNN
# centripetal outlier factor (cof)
def hdiod_score_paper(X: np.ndarray, k: int = 10) -> np.ndarray:

    X = np.asarray(X, dtype=float)

    n = X.shape[0]
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, n-1]. Got k={k}, n={n}.")

    # 构建 kNN 图
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)
    # 去掉自身 避免自身成为第一个邻居
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    # 局部核密度
    rho = local_kernel_density_paper(distances)
    # 对于每个样本 找出其 KNN 中局部密度最大的邻居
    best_pos = np.argmax(rho[indices], axis=1)
    best_neighbor = indices[np.arange(n), best_pos]
    best_neighbor_rho = rho[best_neighbor]

    peaks = np.empty(n, dtype=int)

    # 如果当前点的密度小于其邻居中最大密度
    # 则跳转到该最大密度邻居 重复该过程
    # 直到达到局部密度最高点
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

        # 沿着迭代路径，将路径上所有点的 kNN 合并
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
        # 合并路径上所有点的kNN
        eknn = set()
        for t in traj:
            eknn.update(indices[t].tolist())
        eknn = np.fromiter(eknn, dtype=int)
        peaks[i] = eknn[np.argmax(rho[eknn])]
    #计算cof 值越大样本就越远离高密度地区
    cof = rho[peaks] / (rho + 1e-12)
    return cof