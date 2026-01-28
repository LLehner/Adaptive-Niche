import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import heapq
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans,DBSCAN
import igraph as ig
import leidenalg as la

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
        bic = gmm.bic(distances)
        aic = gmm.aic(distances)
        results[k] = {
            "model": gmm,
            "labels": labels,
            "responsibilities": resp,
            "log_likelihood": gmm.score(distances) * n,
            "bic": bic,
            "aic": aic,
            "icl": bic - 2.0 * (-np.sum(resp * np.log(resp + 1e-12))),
            "n_params": gmm._n_parameters(),
        }
        print(f"Fitted GMM with k={k}")

    return results

def choose_component(results, criterion='icl', delta_bic_threshold=10.0):
    """
    Select optimal GMM using a specified criterion (BIC, AIC, or ICL) 
    with a Delta-BIC safeguard.

    Parameters
    ----------
    results : dict
        Dictionary containing GMM results for each K.
    criterion : str, default='icl'
        The metric to minimize ('bic', 'aic', 'icl').
    delta_bic_threshold : float, default=10.0
        Threshold for BIC improvement required to reject K=1.

    Returns
    -------
    selection : dict
        Contains optimal K, labels, and diagnostics.
    """
    valid_criteria = ['bic', 'aic', 'icl']
    if criterion not in valid_criteria:
        raise ValueError(f"Invalid criterion '{criterion}'. Must be one of {valid_criteria}.")

    ks = sorted(results.keys())

    metrics = {
        "bic": {k: results[k]["bic"] for k in ks},
        "aic": {k: results[k]["aic"] for k in ks},
        "icl": {k: results[k]["icl"] for k in ks}
    }

    target_metric = metrics[criterion]
    k_optimal = min(target_metric, key=target_metric.get)

    bic_vals = metrics["bic"]
    delta_bic = bic_vals[1] - bic_vals[k_optimal]
    
    passed_delta_bic = (k_optimal == 1) or (delta_bic > delta_bic_threshold)

    if not passed_delta_bic:
        k_optimal = 1

    return {
        "optimal_k": k_optimal,
        "labels": results[k_optimal]["labels"],
        "criterion_used": criterion,
        "bic": metrics["bic"],
        "aic": metrics["aic"],
        "icl": metrics["icl"],
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

def _watershed(adata, distances_key, spatial_connectivity_key, labels_key):
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

def cluster_domains(
    adata,
    spatial_connectivity_key="spatial_connectivities",
    flavor="leiden",
    distances_key="distances",
    seeds_key="watershed_seeds",
    n_clusters=100,
    resolution=1.0,
    eps=10,
    min_samples=5,
    random_state=0
):
    """
    Cluster a sparse adjacency matrix using either Leiden (graph-based)
    or KMeans (row-wise embedding of adjacency).

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    spatial_connectivity_key : str
        Key in adata.obsp for adjacency matrix
    flavor : {"watershed", "leiden", "kmeans", "dbscan"}
        Clustering method
    distances_key : str
        Key in adata.obs for distances (used only for watershed)
    n_clusters : int
        Number of clusters (used only for kmeans)
    resolution : float
        Leiden resolution parameter
    random_state : int
        Random seed

    Returns
    -------
    labels : np.ndarray (shape: N)
        Updates adata.obs with cluster labels under keys.
    """
    
    if flavor == "watershed":
        labels = _watershed(
            adata,
            distances_key=distances_key,
            spatial_connectivity_key=spatial_connectivity_key,
            labels_key=seeds_key
        )
        adata.obs["watershed_niches"] = pd.Categorical(labels)

    elif flavor == "leiden":
        sources, targets = adata.obsp[spatial_connectivity_key].nonzero()

        g = ig.Graph(
            n=adata.n_obs,
            edges=list(zip(sources, targets)),
            edge_attrs={"weight": None},
            directed=False
        )

        partition = la.find_partition(
            g,
            la.RBConfigurationVertexPartition,
            weights=None,
            resolution_parameter=resolution,
            seed=random_state
        )

        labels = np.array(partition.membership)
        adata.obs["leiden_niches"] = pd.Categorical(labels)

    elif flavor == "kmeans":
        X = adata.obs[distances_key].values.reshape(-1, 1)

        km = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init="auto"
        )

        labels = km.fit_predict(X)
        adata.obs["kmeans_niches"] = pd.Categorical(labels)
    
    elif flavor == "dbscan":
        coords = adata.obsm["spatial"]

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1
        )

        labels = db.fit_predict(coords)
        adata.obs["dbscan_niches"] = pd.Categorical(labels)
    else:
        raise ValueError(f"Unknown flavor '{flavor}'")

    return labels
