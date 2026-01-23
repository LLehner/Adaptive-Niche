import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
import squidpy as sq
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, cKDTree
import os
import spatialdata as sd
import heapq
import numpy as np
from sklearn.mixture import GaussianMixture

#TODO adapt to both adata/sdata usage
#TODO add other clustering methods (DBSCAN, kmeans. leiden) for comparison
def fit_gmm(sdata, k_range, random_state=0, covariance_type="full", reg_covar=1e-6):
    """
    Fit 1D GMMs over a range of components.

    Returns
    -------
    results : dict
        Keys are K, values are dicts with fitted model outputs.
    """

    adata = sdata.tables["table"]
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=1)
    x = np.log(adata.obsp["spatial_distances"].data)
    x = np.asarray(x).reshape(-1, 1)
    n = x.shape[0]

    results = {}

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        gmm.fit(x)

        labels = gmm.predict(x)
        resp = gmm.predict_proba(x)

        results[k] = {
            "model": gmm,
            "labels": labels,
            "responsibilities": resp,
            "log_likelihood": gmm.score(x) * n,
            "bic": gmm.bic(x),
            "n_params": gmm._n_parameters(),
        }
        print(f"Fitted GMM with k={k}")

    return results, x

def choose_component(results, delta_bic_threshold=10.0):
    """
    Select optimal GMM using ICL with ΔBIC safeguard.

    Returns
    -------
    selection : dict
        Contains optimal K, labels, and diagnostics.
    """
    ks = sorted(results.keys())

    bic = {}
    icl = {}

    for k in ks:
        r = results[k]["responsibilities"]
        entropy = -np.sum(r * np.log(r + 1e-12))
        bic[k] = results[k]["bic"]
        icl[k] = bic[k] - 2.0 * entropy

    # ICL-optimal K
    k_icl = min(icl, key=icl.get)

    # ΔBIC check against K=1
    delta_bic = bic[1] - bic[k_icl]
    passed_delta_bic = (k_icl == 1) or (delta_bic > delta_bic_threshold)

    # Conservative fallback
    if not passed_delta_bic:
        k_icl = 1

    return {
        "optimal_k": k_icl,
        "labels": results[k_icl]["labels"],
        "bic": bic,
        "icl": icl,
        "passed_delta_bic": passed_delta_bic,
    }

def cluster_stability(x, k, n_repeats=50, subsample_fraction=0.8, random_state=0):
    """
    Perform clustering stability analysis for fixed K.

    Returns
    -------
    stability : dict
        Contains ARI distribution and mean drift statistics.
    """
    if k <= 1:
        return None

    rng = np.random.RandomState(random_state)
    x = np.asarray(x)
    n = len(x)

    base_gmm = GaussianMixture(
        n_components=k, random_state=random_state
    ).fit(x.reshape(-1, 1))
    base_labels = base_gmm.predict(x.reshape(-1, 1))
    base_means = np.sort(base_gmm.means_.ravel())

    ari_scores = []
    mean_drifts = []

    for i in range(n_repeats):
        idx = rng.choice(
            n, size=int(subsample_fraction * n), replace=False
        )
        xs = x[idx].reshape(-1, 1)

        gmm = GaussianMixture(
            n_components=k, random_state=random_state + i + 1
        ).fit(xs)

        labels = gmm.predict(xs)
        means = np.sort(gmm.means_.ravel())

        ari = adjusted_rand_score(base_labels[idx], labels)
        ari_scores.append(ari)

        drift = np.linalg.norm(means - base_means)
        mean_drifts.append(drift)

    return {
        "ari_mean": np.mean(ari_scores),
        "ari_std": np.std(ari_scores),
        "mean_drift_mean": np.mean(mean_drifts),
        "mean_drift_std": np.std(mean_drifts),
        "ari_scores": ari_scores,
    }

def watershed(h, neighbors, labels, n):
    """performs watershed-like 'segmentation' on graph based on scalar distance values using priority queue."""
    pq = []

    for i in range(n):
        if labels[i] > 0:
            heapq.heappush(pq, (h[i], i))

    visited = labels > 0

    while pq:
        _, i = heapq.heappop(pq)

        for j in neighbors[i]:
            if not visited[j]:
                labels[j] = labels[i]
                visited[j] = True
                heapq.heappush(pq, (h[j], j))
    return labels