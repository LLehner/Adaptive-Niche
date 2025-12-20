import squidpy as sq
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gstools as gs

def plot_dist_histogram(adata, n_neighs_range, bins, save=False):
    """Plots histogram of distance distributions for each spatial neighbor graph."""
    
    fig, axes = plt.subplots(1, len(n_neighs_range), figsize=(5*len(n_neighs_range), 5))
    for i, n_neighs in enumerate(n_neighs_range):
        sq.gr.spatial_neighbors(adata, coord_type="generic", n_neighs=n_neighs, key_added=f"{n_neighs}_neighbors")
        histogram = sns.histplot(adata.obsp[f"{n_neighs}_neighbors_distances"].data, bins=bins, ax=axes[i])
        histogram.set_title(f"Distance Distribution for {n_neighs} Neighbors")
        histogram.set_xlabel("Distance")
        fig = histogram.get_figure()
        
        if save:
            fig.savefig(f"{n_neighs}_neighbors_dist_histogram.png")
        #else:
            #fig.show()
        
def plot_dist_by_neigh(adata, distances_key):
    """Plots spatial distances by nth neighbor."""
    
    csr = adata.obsp[distances_key]
    row_to_values = {}
    
    for i in range(csr.shape[0]):
        sl = slice(csr.indptr[i], csr.indptr[i + 1])
        row_to_values[i] = csr.data[sl]
        row_to_values[i].sort()
        
    arrays = list(row_to_values.values())
    max_len = max(len(a) for a in arrays)

    data = np.full((len(arrays), max_len), np.nan)
    for i, a in enumerate(arrays):
        data[i, :len(a)] = a

    x = np.tile(np.arange(max_len), len(arrays))
    y = data.flatten()
    mask = ~np.isnan(y)
    plt.scatter(x[mask], y[mask], color='grey', alpha=0.3, s=10, label='raw distances')

    means = np.nanmean(data, axis=0)
    plt.plot(np.arange(max_len), means, color='black', linestyle='--', marker='o', markersize=6, label='mean distance')

    plt.xlabel("nth neighbor")
    plt.ylabel("spatial distance")
    plt.legend()
    plt.show()
    
def get_neighs_by_radius(adata, radii):
    """Calculate summary statistics of neighbors within radius across range of radii."""
    
    cells_in_radius = {}
    for max_radius in radii:
        sq.gr.spatial_neighbors(adata, coord_type="generic", radius=max_radius)
        row_counts = np.diff(adata.obsp["spatial_connectivities"].indptr)
        cells_in_radius[max_radius] = row_counts
    
    cells_in_radius_df = pd.DataFrame(cells_in_radius)
    cells_in_radius_df.index = adata.obs_names

    mean = cells_in_radius_df.to_numpy().mean(axis=0).round().astype(int)
    min_cells = cells_in_radius_df.to_numpy().min(axis=0).round().astype(int)
    first_min_idx =np.argmax(min_cells>=20)
    max_cells = cells_in_radius_df.to_numpy().max(axis=0).round().astype(int)
    #first_max_idx = np.argmax(max_cells>=50)

    # cells within radius with radius such that min cells >=20
    adata.obs["cells_in_radius"] = cells_in_radius_df[radii[first_min_idx]]
    adata.uns["radius_stats"] = {
        "radii": radii,
        "mean_cells": mean,
        "min_cells": min_cells,
        "max_cells": max_cells,
        "first_min_idx": first_min_idx}
    

def plot_neighs_by_radius(adata):
    """Plots number of neighbors within radius across range of radii."""
    radii = adata.uns["radius_stats"]["radii"]
    mean = adata.uns["radius_stats"]["mean_cells"]
    min_cells = adata.uns["radius_stats"]["min_cells"]
    max_cells = adata.uns["radius_stats"]["max_cells"]
    first_min_idx = adata.uns["radius_stats"]["first_min_idx"]
    
    plt.plot(radii, mean, color='black', marker='o', markersize=6, label="mean n_neighbors within radius")
    plt.plot(radii, max_cells, color='red', marker='o', markersize=6, label="min n_neighbors within radius")
    plt.plot(radii, min_cells, color='blue', marker='o', markersize=6, label="max n_neighbors within radius")
    plt.axvline(x=radii[first_min_idx], color='orange', label='first radius with min 20 neighbors')
    #plt.axvline(x=radii[first_max_idx], color='orange', label='first radius with max 50 neighbors')
    plt.xlabel("radius")
    plt.ylabel("n_neighbors")
    plt.legend()
    plt.show()