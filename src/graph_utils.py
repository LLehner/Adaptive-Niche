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

def get_neighbors(adata, type, n=10):
    if type == "delaunay":
        sq.gr.spatial_neighbors(adata, coord_type="generic", delaunay=True)
    elif type =="knn":
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n)


def prune_graph(adata, gmm_key):

    A = adata.obsp["spatial_connectivities"]
    D = adata.obsp["spatial_distances"]
    indptr = A.indptr
    indices = A.indices

    for i in range(A.shape[0]):
        start, end = indptr[i], indptr[i + 1]

        if start == end:
            continue

        row_label = adata.obs[gmm_key][i]
        neigh = indices[start:end]

        # mask of edges to KEEP
        keep = adata.obs[gmm_key][neigh] == row_label

        # set pruned entries to zero (both matrices)
        drop = ~keep
        if np.any(drop):
            A.data[start:end][drop] = 0.0
            D.data[start:end][drop] = 0.0

    # actually remove the zeros from the sparse structure
    A.eliminate_zeros()
    D.eliminate_zeros()
    
    adata.obsp["pruned_spatial_connectivities"] = A
    adata.obsp["pruned_spatial_distances"] = D


def set_seeds(
    adata,
    distances_key,
    spatial_connectivity_key="spatial_connectivities",
    min_neighbors=1,
):
    # build undirected, clean graph
    graph = adata.obsp[spatial_connectivity_key].tocsr()
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()

    distances = adata.obs[distances_key].to_numpy()
    n = graph.shape[0]

    labels = np.zeros(n, dtype=np.int32)
    current_label = 1

    indptr = graph.indptr
    indices = graph.indices

    for i in range(n):

        start, end = indptr[i], indptr[i + 1]
        if start == end:
            continue

        neigh = indices[start:end]

        # remove self
        neigh = neigh[neigh != i]

        if neigh.size < min_neighbors:
            continue

        di = distances[i]

        # strictly lower than all neighbors
        if np.all(di < distances[neigh]):
            labels[i] = current_label
            current_label += 1

    adata.obs["watershed_seeds"] = labels

