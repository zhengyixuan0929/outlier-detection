import numpy as np
from sklearn.datasets import make_blobs, make_moons


def _rng(random_state=42):
    return np.random.default_rng(random_state)


def _inject_uniform_outliers(Xn, n_out, low=-10, high=10, random_state=42):
    rng = _rng(random_state)
    Xo = rng.uniform(low=low, high=high, size=(n_out, Xn.shape[1]))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S01: single Gaussian + uniform outliers (类似 DS01)
# -----------------------------
def syn_gauss_uo(n_normal=1000, contamination=0.05, random_state=42):
    n_out = int(n_normal * contamination)
    Xn, _ = make_blobs(
        n_samples=n_normal,
        centers=[(0, 0)],
        cluster_std=1.0,
        random_state=random_state
    )
    return _inject_uniform_outliers(Xn, n_out, low=-8, high=8, random_state=random_state)


# -----------------------------
# S02: multi Gaussian blobs + uniform outliers (类似 DS02)
# -----------------------------
def syn_multi_blobs_uo(n_normal=1999, contamination=0.08, random_state=42):
    n_out = int(n_normal * contamination)
    Xn, _ = make_blobs(
        n_samples=n_normal,
        centers=[(-3, -2), (3, -1), (0, 3)],
        cluster_std=[0.6, 0.9, 0.7],
        random_state=random_state
    )
    return _inject_uniform_outliers(Xn, n_out, low=-10, high=10, random_state=random_state)


# -----------------------------
# S03: two density clusters + uniform outliers (类似 DS03/DS04 的密度差)
# -----------------------------
def syn_two_density(n_normal=2026, contamination=0.06, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)
    n1 = n_normal // 2
    n2 = n_normal - n1

    X1, _ = make_blobs(n_samples=n1, centers=[(-3, 0)], cluster_std=0.35, random_state=random_state)
    X2, _ = make_blobs(n_samples=n2, centers=[(3, 0)], cluster_std=1.25, random_state=random_state + 1)
    Xn = np.vstack([X1, X2])

    # outliers 更远一些
    Xo = rng.uniform(low=-12, high=12, size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S04: block / checkerboard normal + scattered outliers (类似 DS03/DS04 方块结构)
# -----------------------------
def syn_blocks(n_normal=1689, contamination=0.05, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    # 4 个方块区域作为正常点
    blocks = [(-4, -4), (-4, 4), (4, -4), (4, 4)]
    X_list = []
    per = n_normal // len(blocks)

    for (cx, cy) in blocks:
        # 方块：均匀分布在小正方形内
        Xb = rng.uniform(low=[cx - 1.2, cy - 1.2], high=[cx + 1.2, cy + 1.2], size=(per, 2))
        X_list.append(Xb)

    Xn = np.vstack(X_list)
    # outliers：铺在全局范围
    Xo = rng.uniform(low=-8, high=8, size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S05: V-shape curve + outliers (类似 DS05 折线结构)
# -----------------------------
def syn_vshape(n_normal=3975, contamination=0.08, noise=0.08, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    x = np.linspace(-4, 4, n_normal)
    y1 = np.abs(x)  # V shape
    Xn = np.column_stack([x, y1])
    Xn += rng.normal(0, noise, size=Xn.shape)

    Xo = rng.uniform(low=[-6, -1], high=[6, 6], size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S06: moons + outliers (类似 DS06 半月)
# -----------------------------
def syn_moons(n_normal=873, contamination=0.06, noise=0.08, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    Xn, _ = make_moons(n_samples=n_normal, noise=noise, random_state=random_state)
    # 放大一点让结构更明显
    Xn = Xn * 4.0

    Xo = rng.uniform(low=-6, high=6, size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S07: spiral + outliers (类似 DS07 螺旋)
# -----------------------------
def syn_spiral(n_normal=1458, contamination=0.06, noise=0.15, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    t = np.linspace(0, 4*np.pi, n_normal)
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)
    Xn = np.column_stack([x, y])
    Xn += rng.normal(0, noise, size=Xn.shape)

    Xo = rng.uniform(low=-15, high=15, size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S08: double spiral + outliers (类似 DS08 双螺旋)
# -----------------------------
def syn_double_spiral(n_normal=2894, contamination=0.06, noise=0.18, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    n1 = n_normal // 2
    n2 = n_normal - n1

    t1 = np.linspace(0, 4*np.pi, n1)
    r1 = t1
    X1 = np.column_stack([r1*np.cos(t1), r1*np.sin(t1)])

    t2 = np.linspace(0, 4*np.pi, n2)
    r2 = t2
    # 第二条螺旋相位偏移
    X2 = np.column_stack([r2*np.cos(t2 + np.pi), r2*np.sin(t2 + np.pi)])

    Xn = np.vstack([X1, X2])
    Xn += rng.normal(0, noise, size=Xn.shape)

    Xo = rng.uniform(low=-15, high=15, size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S09: sine curve + outliers (类似 DS09)
# -----------------------------
def syn_sine(n_normal=1298, contamination=0.05, noise=0.10, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    x = np.linspace(-6, 6, n_normal)
    y_curve = np.sin(x)
    Xn = np.column_stack([x, y_curve])
    Xn += rng.normal(0, noise, size=Xn.shape)

    Xo = rng.uniform(low=[-7, -3], high=[7, 3], size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# S10: two parallel lines + outliers (类似 DS11 斜线)
# -----------------------------
def syn_two_lines(n_normal=1234, contamination=0.06, noise=0.08, random_state=42):
    rng = _rng(random_state)
    n_out = int(n_normal * contamination)

    n1 = n_normal // 2
    n2 = n_normal - n1
    x1 = rng.uniform(-6, 6, size=n1)
    y1 = 0.6 * x1 + 1.5 + rng.normal(0, noise, size=n1)

    x2 = rng.uniform(-6, 6, size=n2)
    y2 = 0.6 * x2 - 1.5 + rng.normal(0, noise, size=n2)

    Xn = np.column_stack([np.hstack([x1, x2]), np.hstack([y1, y2])])

    Xo = rng.uniform(low=[-7, -7], high=[7, 7], size=(n_out, 2))
    X = np.vstack([Xn, Xo])
    y = np.hstack([np.zeros(len(Xn), dtype=int), np.ones(n_out, dtype=int)])
    return X, y


# -----------------------------
# registry: 10 synthetic datasets
# -----------------------------
SYNTHETICS = {
    "syn_gauss_uo": syn_gauss_uo,
    "syn_multi_blobs_uo": syn_multi_blobs_uo,
    "syn_two_density": syn_two_density,
    "syn_blocks": syn_blocks,
    "syn_vshape": syn_vshape,
    "syn_moons": syn_moons,
    "syn_spiral": syn_spiral,
    "syn_double_spiral": syn_double_spiral,
    "syn_sine": syn_sine,
    "syn_two_lines": syn_two_lines,
}