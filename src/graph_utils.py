from scipy.spatial import Delaunay, cKDTree
import numpy as np
from spatialdata import SpatialData
from anndata import AnnData
import squidpy as sq
import scipy.sparse as sp

def get_distances(adata: AnnData | SpatialData, k=1, log=True, transform="log"):
    """For each cell get the distance to its k-th nearest neighbor.
    
    Returns
    -------
    distances : array-like
        Array of distances to k-th nearest neighbor for each cell.
    """
    adata = adata.tables["table"] if isinstance(adata, SpatialData) else adata
    sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=k)
    distances = adata.obsp["spatial_distances"].data
    if log and transform == "log":
        distances = np.log(distances)
    elif log and transform == "log1p":
        distances = np.log(1 + distances)
    
    adata.obs[f"{k}_nn_distance"] = distances

# DEPRECATED
# def get_neighbors(adata, type, n=40):
#     if type == "delaunay":
#         sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
#     elif type =="knn":
#         sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n)

def prune_graph(adata, gmm_key, prune_by="label", distance_key=None, n_std=2.0):
    """
    Prunes the spatial graph.
    
    If prune_by="distance":
    1. Calculates thresholds using GMM stats (assumed to be from log-data).
    2. Takes the LOG of the graph's spatial_distances.
    3. Compares: log(graph_distance) > (Mean + n*Std).
    """

    # CRITICAL: Use .copy() to avoid destroying the original spatial_connectivities in-place
    A = adata.obsp["spatial_connectivities"].copy()
    D = adata.obsp["spatial_distances"].copy()
    
    indptr = A.indptr
    indices = A.indices

    if prune_by == "distance":
        if distance_key is None:
            raise ValueError("You must provide `distance_key` when using `prune_by='distance'`.")
        
        # 1. Group by GMM component and calculate stats (on already logged data)
        stats = adata.obs.groupby(gmm_key, observed=False)[distance_key].agg(['mean', 'std'])
        stats['std'] = stats['std'].fillna(0)
        
        # 2. Define threshold in LOG space
        stats['log_threshold'] = stats['mean'] + (n_std * stats['std'])
        
        # 3. Map thresholds to cells (no exp conversion needed here)
        mapping_dict = stats['log_threshold'].to_dict()
        cell_thresholds = adata.obs[gmm_key].map(mapping_dict).astype(float).values

    for i in range(A.shape[0]):
        start, end = indptr[i], indptr[i + 1]

        if start == end:
            continue

        neigh_indices = indices[start:end]

        if prune_by == "label":
            row_label = adata.obs[gmm_key][i]
            neigh_labels = adata.obs[gmm_key][neigh_indices]
            keep = neigh_labels == row_label
            drop = ~keep

        elif prune_by == "distance":
            # Get physical edge lengths
            raw_edge_lengths = D.data[start:end]
            
            # Log-transform the graph edges to match the domain of the GMM stats
            # Adding a tiny epsilon to avoid log(0) if duplicate cells exist
            log_edge_lengths = np.log(raw_edge_lengths + 1e-12)
            
            # Get the max log-threshold for this cell
            max_log_allowed = cell_thresholds[i]
            
            # Prune if log(dist) > log(threshold)
            drop = log_edge_lengths > max_log_allowed

        # Apply pruning
        if np.any(drop):
            A.data[start:end][drop] = 0.0
            D.data[start:end][drop] = 0.0

    # Cleanup
    A.eliminate_zeros()
    D.eliminate_zeros()
    
    adata.obsp["pruned_spatial_connectivities"] = A
    adata.obsp["pruned_spatial_distances"] = D
    
    print(f"Graph pruned by '{prune_by}'. stored in 'pruned_spatial_connectivities'.")

# DEPRECATED
# def set_seeds(
#     adata,
#     distances_key,
#     spatial_connectivity_key="spatial_connectivities",
#     min_neighbors=1,
# ):
#     # build undirected, clean graph
#     graph = adata.obsp[spatial_connectivity_key].tocsr()
#     graph = graph.maximum(graph.T)
#     graph.eliminate_zeros()

#     distances = adata.obs[distances_key].to_numpy()
#     n = graph.shape[0]

#     labels = np.zeros(n, dtype=np.int32)
#     current_label = 1

#     indptr = graph.indptr
#     indices = graph.indices

#     for i in range(n):

#         start, end = indptr[i], indptr[i + 1]
#         if start == end:
#             continue

#         neigh = indices[start:end]

#         # remove self
#         neigh = neigh[neigh != i]

#         if neigh.size < min_neighbors:
#             continue

#         di = distances[i]

#         # strictly lower than all neighbors
#         if np.all(di < distances[neigh]):
#             labels[i] = current_label
#             current_label += 1

#     adata.obs["watershed_seeds"] = labels
    

