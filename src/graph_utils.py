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

def get_neighbors(adata, type, gmm_labels, n=10):
    if type == "delaunay":
        sq.gr.spatial_neighbors(adata, library_key=gmm_labels, coord_type="generic", delaunay=True)
    elif type =="knn":
        sq.gr.spatial_neighbors(adata, library_key=gmm_labels,coord_type="generic", n_neighs=n)

# not yet tested!
def prune_graph(adata, type, threshold=None, percentile=None):
    A = adata.obsp["spatial_connectivities"].tocsr()
    D = adata.obsp["spatial_distances"].tocsr()

    # Convert to COO for masking
    A_coo = A.tocoo()
    D_coo = D.tocoo()

    if type == "threshold":
        mask = D_coo.data <= threshold

    elif type == "percentile":
        cutoff = np.percentile(D_coo.data, percentile)
        mask = D_coo.data <= cutoff

    else:
        raise ValueError("type must be 'threshold' or 'percentile'")

    # Prune adjacency
    A_pruned = sp.coo_matrix(
        (A_coo.data[mask], (A_coo.row[mask], A_coo.col[mask])),
        shape=A.shape
    ).tocsr()

    # Prune distances (same mask)
    D_pruned = sp.coo_matrix(
        (D_coo.data[mask], (D_coo.row[mask], D_coo.col[mask])),
        shape=D.shape
    ).tocsr()

    adata.obsp["spatial_connectivities_pruned"] = A_pruned
    adata.obsp["spatial_distances_pruned"] = D_pruned


def set_seeds(adata, distances_key, spatial_connectivity_key="spatial_connectivities"):
    
    graph = adata.obsp[spatial_connectivity_key].tocsr()
    n = graph.shape[0]
    distances = adata.obs[distances_key].values

    labels = np.zeros(n, dtype=int)
    current_label = 1

    for i in range(n):
        # indices of neighbors of i (excluding self if present)
        neighbors = graph.indices[graph.indptr[i]:graph.indptr[i+1]]
        neighbors = neighbors[neighbors != i]

        if neighbors.size == 0:
            continue

        # local minimum condition
        if np.all(distances[i] < distances[neighbors]):
            labels[i] = current_label
            current_label += 1

    adata.obs["watershed_seeds"] = labels

