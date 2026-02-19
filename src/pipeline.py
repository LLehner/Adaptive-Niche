from src import clustering, graph_utils, plotting, LR
import spatialdata as sd
import os
import squidpy as sq

INPUT_DIR = "zarr/"

def zarr_pipeline(zarr_dir, method="knn", plot=True):
    """
    method: knn, delaunay
    """
    sdata = sd.read_zarr(os.path.join(INPUT_DIR, zarr_dir))
    adata = sdata.tables["table"]
    K = 1
    graph_utils.get_distances(adata, k=K, log=True)
    results = clustering.fit_gmm(adata, distance_key="1_nn_distance", k_range=range(1,10))
    best_model_icl = clustering.choose_component(results, criterion="icl", delta_bic_threshold=10.0)
    adata.obs["gmm_labels"] = best_model_icl["labels"]
    adata.obs["gmm_labels"] = adata.obs["gmm_labels"].astype("category") 
    graph_utils.get_neighbors(adata, type=method)
    graph_utils.prune_graph(adata, gmm_key="gmm_labels", distance_key="1_nn_distance", prune_by="distance")
    clustering.watershed_by_descent(adata, distances_key="1_nn_distance", spatial_connectivity_key="pruned_spatial_connectivities")
    plotting.assign_colors(sdata, "watershed_by_descent")
    sq.pl.spatial_scatter(
            adata,
            color="watershed_by_descent",
            size=1,
            title="watershed niches",
            shape=None,      
            legend_loc=None, 
            dpi=300,
            figsize=(10, 10),
            save=f"{zarr_dir}_plot.png"
        )