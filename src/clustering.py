import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
import squidpy as sq
from spatialdata import SpatialData
import heapq
import numpy as np
from sklearn.mixture import GaussianMixture
from anndata import AnnData

#TODO adapt to both adata/sdata usage
#TODO add other clustering methods (DBSCAN, kmeans. leiden) for comparison

def fit_gmm(adata, distance_key, k_range, random_state=0, covariance_type="full", reg_covar=1e-6):
    """
    Fit 1D GMMs over a range of components.

    Returns
    -------
    results : dict
        Keys are K, values are dicts with fitted model outputs.
    """
    distances = adata.obs[distance_key].values.reshape(-1, 1)
    n = distances.shape[0]

    results = {}

    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=reg_covar,
            random_state=random_state,
        )
        gmm.fit(distances)

        labels = gmm.predict(distances)
        resp = gmm.predict_proba(distances)

        results[k] = {
            "model": gmm,
            "labels": labels,
            "responsibilities": resp,
            "log_likelihood": gmm.score(distances) * n,
            "bic": gmm.bic(distances),
            "aic": gmm.aic(distances),
            "n_params": gmm._n_parameters(),
        }
        print(f"Fitted GMM with k={k}")

    return results

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
    aic = {}
    icl = {}

    for k in ks:
        r = results[k]["responsibilities"]
        entropy = -np.sum(r * np.log(r + 1e-12))
        bic[k] = results[k]["bic"]
        aic[k] = results[k]["aic"]
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
        "aic": aic,
        "icl": icl,
        "passed_delta_bic": passed_delta_bic,
    }

def cluster_stability(adata, distance_key, k, n_repeats=50, subsample_fraction=0.8, random_state=0):
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
    x = adata.obs[distance_key].values
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

def watershed(adata, distances_key, spatial_connectivity_key, labels_key):
    """performs watershed-like 'segmentation' on graph based on scalar distance values using priority queue."""
    # Get scalar values
    h = adata.obs[distances_key].values
    labels = adata.obs[labels_key].values.copy()
    n = len(labels)

    # Get sparse adjacency graph in CSR format
    graph = adata.obsp[spatial_connectivity_key].tocsr()

    # Priority queue initialization: seeds
    pq = [(h[i], i) for i in range(n) if labels[i] > 0]
    heapq.heapify(pq)

    # Visited mask
    visited = labels > 0

    while pq:
        _, i = heapq.heappop(pq)

        # Retrieve neighbors from sparse row
        neighbors = graph.indices[graph.indptr[i]:graph.indptr[i + 1]]

        for j in neighbors:
            if not visited[j]:
                labels[j] = labels[i]     # propagate basin label
                visited[j] = True
                heapq.heappush(pq, (h[j], j))

    return labels