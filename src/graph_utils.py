from scipy.spatial import Delaunay, cKDTree
import numpy as np

#TODO simplify with squidpy
#TODO add graph pruning and node-merging option

def _get_distances(adata, k=1):
    coords = adata.obsm["spatial"]
    k_density = k

    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=k_density + 1)
    h = np.log(dists[:, k_density])  # scalar field
    n = len(h)
    return h,n,tree

def _get_neighbors(coords, tree, type, n):

    if type == "delaunay":
        tri = Delaunay(coords)

        neighbors = {i: set() for i in range(n)}
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    a, b = simplex[i], simplex[j]
                    neighbors[a].add(b)
                    neighbors[b].add(a)
    else:
        k_adj = 5  # typical planar degree

        _, idx = tree.query(coords, k=k_adj + 1)

        neighbors = {i: set() for i in range(n)}
        for i in range(n):
            for j in idx[i, 1:]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    Adj = np.zeros((n, n), dtype=int)

    for i, nbrs in neighbors.items():
        for j in nbrs:
            Adj[i, j] = 1
            Adj[j, i] = 1  # redundant if neighbors is symmetric, but safe
    return neighbors, Adj

def _set_seeds(h, neighbors, n):
    labels = np.zeros(n, dtype=int)
    current_label = 1

    for i in range(n):
        if all(h[i] <= h[j] for j in neighbors[i]):
            labels[i] = current_label
            current_label += 1
    return labels