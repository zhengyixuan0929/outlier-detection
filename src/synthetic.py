import numpy as np
from sklearn.datasets import make_blobs, make_moons

def synth_gaussian_with_uniform_outliers(n_normal=1000, contamination=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n_out = int(n_normal * contamination)

    Xn, _ = make_blobs(
        n_samples=n_normal,
        centers=[(0, 0)],
        cluster_std=1.0,
        random_state=random_state
    )

    # 在一个更大的方框里均匀撒异常
    Xo = rng.uniform(low=-8, high=8, size=(n_out, 2))

    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_out, dtype=int)])
    return X, y

def synth_two_density_clusters_with_outliers(n_normal=1200, contamination=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n_out = int(n_normal * contamination)

    # 两个簇：一个密、一个稀（测试 LOF/HDIOD 很有意义）
    n1 = n_normal // 2
    n2 = n_normal - n1
    X1, _ = make_blobs(n_samples=n1, centers=[(-3, 0)], cluster_std=0.4, random_state=random_state)
    X2, _ = make_blobs(n_samples=n2, centers=[(3, 0)], cluster_std=1.2, random_state=random_state + 1)

    Xn = np.vstack([X1, X2])
    Xo = rng.uniform(low=-10, high=10, size=(n_out, 2))

    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_out, dtype=int)])
    return X, y

def synth_moons_with_outliers(n_normal=1200, contamination=0.05, random_state=42):
    rng = np.random.default_rng(random_state)
    n_out = int(n_normal * contamination)

    Xn, _ = make_moons(n_samples=n_normal, noise=0.08, random_state=random_state)
    Xo = rng.uniform(low=-3, high=3, size=(n_out, 2))

    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(n_normal, dtype=int), np.ones(n_out, dtype=int)])
    return X, y
