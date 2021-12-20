import numpy as np
import random
from sklearn.cluster import KMeans


def l2_dist(X, Y):
    return ((np.sum(X ** 2, axis=1, keepdims=True)) + (np.sum(Y ** 2, axis=1, keepdims=True)).T - 2 * X @ Y.T + 1e-10) ** 0.5


def get_normal(X):
    _, s, vh = np.linalg.svd(X.T @ X)
    min_idx = np.argmin(s)
    eigne_vec = vh[min_idx, :]
    return eigne_vec


def get_curvature(X):
    _, s, _ = np.linalg.svd(X.T @ X)
    least3 = np.sort(s ** 2)[:3]
    return least3[0] / np.sum(least3)


def random_pick(X, r, ratio=0.5):
    size = int(X.shape[0] * ratio)
    idx_all = list(range(X.shape[0]))
    idx = random.sample(idx_all, k=size)
    return np.squeeze(X[idx]).T


def spatial_clustering(X, r, ratio=0.5):
    size = int(X.shape[0] * ratio)
    kmeans = KMeans(n_clusters=size, init='k-means++', random_state=0).fit(X)
    dist = l2_dist(kmeans.cluster_centers_, X)
    idx = np.argmin(dist, axis=1)
    return np.squeeze(X[idx]).T


def curvature_thresholding(X, r, ratio=0.5):
    dist = l2_dist(X, X)
    size = int(X.shape[0] * ratio)
    curvature = [-get_curvature(X[dist[i, :] < r, :])
                 for i in range(X.shape[0])]
    print(np.mean(curvature), np.min(curvature), np.max(curvature))
    idx = np.argsort(curvature)[:size]
    return np.squeeze(X[idx]).T


def patch_stats_clustering(X, r, ratio=0.5):
    N = X.shape[0]
    dist = l2_dist(X, X)
    size = int(X.shape[0] * ratio)
    stats = []
    for i in range(N):
        patch_idx = np.squeeze(np.argwhere(dist[i, :] < r))
        patch = X[patch_idx, :]
        mean = np.mean(patch, axis=0)
        stderr = np.std(patch, axis=0)  # variance of the patch
        # distance from mean to center
        target2mean = np.linalg.norm(X[i, :] - mean)
        stats.append(np.append(stderr, [target2mean]))
    stats = np.stack(stats, axis=0)
    feat = np.concatenate([X, stats], axis=1)
    kmeans = KMeans(n_clusters=size, init='k-means++',
                    random_state=0).fit(feat)
    dist = l2_dist(kmeans.cluster_centers_, feat)
    idx = np.argmin(dist, axis=1)
    return np.squeeze(X[idx]).T


def get_reduce_method(method):
    if method == 'rand':
        return random_pick
    elif method == 'sp_cluster':
        return spatial_clustering
    elif method == 'curvature':
        return curvature_thresholding
    elif method == 'patch_stats':
        return patch_stats_clustering
    else:
        return None


def PFH(X, r=1e-1, bins=4):
    print(X.shape)
    '''X: Nx3'''
    N = X.shape[0]
    dist = l2_dist(X, X)
    normals = np.stack([get_normal(X[dist[i, :] < r, :])
                       for i in range(N)], axis=0)

    signitures = []
    patch_size_stats = []
    # dim 0 to be t, dim 1 to be s
    for i in range(N):
        patch_idx = np.squeeze(np.argwhere(dist[i, :] < r))
        patch = X[patch_idx, :]
        patch_size = patch.shape[0]
        patch_size_stats.append(patch_size)

        tri_idx = np.tri(patch_size) - np.eye(patch_size)
        tri_idx = tri_idx.astype(bool)

        patch_normal = normals[patch_idx, :]
        patch_dist = dist[patch_idx]  # kxN
        patch_dist = patch_dist[:, patch_idx]  # kxk

        u = np.repeat(patch_normal[:, None], patch_size, axis=1)  # kxkx3
        diff = patch[:, None] - patch[None]  # kxkx3
        normalized_diff = diff / (patch_dist[..., None] + 1e-10)  # kxkx
        normalized_diff[np.eye(patch_size, dtype=bool)] = 0
        v = np.cross(u, normalized_diff)  # kxkx3
        w = np.cross(u, v)  # kxkx3

        alpha = np.sum(v * patch_normal[:, None], axis=2)  # kxk
        phi = np.sum(u * normalized_diff, axis=2)  # kxk
        x1 = np.sum(w * patch_normal[:, None], axis=2)
        x2 = np.sum(u * patch_normal[:, None], axis=2)
        theta = np.arctan2(x1, x2)

        alpha = alpha[tri_idx]
        phi = phi[tri_idx]
        theta = theta[tri_idx]
        d = patch_dist[tri_idx]

        feat = np.stack([alpha, phi, theta], axis=1)  # k(k-1)/2 x 4
        hist, _ = np.histogramdd(feat, bins=bins)
        signitures.append(hist.flatten())

    signitures = np.stack(signitures, axis=0)
    return signitures, patch_size_stats
