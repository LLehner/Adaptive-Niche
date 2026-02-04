import os
import gc  # Garbage collection interface
import matplotlib.pyplot as plt
import seaborn as sns
import spatialdata as sd

# Import your custom modules
from src import clustering, graph_utils, plotting

# Configuration
INPUT_DIR = "../single_cell/data/zarr_outputs_no_images/"
# INPUT_DIR = "zarr/"


# Get a list of all .zarr directories in the input folder
zarr_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.zarr') and os.path.isdir(os.path.join(INPUT_DIR, f))]
zarr_files.sort()  # Optional: sort alphabetically for consistent ordering

print(f"Found {len(zarr_files)} datasets to process: {zarr_files}")

# Iterate through each dataset sequentially
for zarr_name in zarr_files:
    zarr_path = os.path.join(INPUT_DIR, zarr_name)
    dataset_name = zarr_name.replace(".zarr", "") # Clean name for the output file
    
    print(f"--- Processing: {dataset_name} ---")

    try:
        # 1. Load Data
        sdata = sd.read_zarr(zarr_path)
        adata = sdata.tables["table"]

        # 2. Pre-processing & GMM
        print("Running GMM and graph construction...")
        graph_utils.get_distances(adata, k=1, log=True)
        results = clustering.fit_gmm(adata, distance_key="1_nn_distance", k_range=range(1, 10))
        best_model = clustering.choose_component(results, delta_bic_threshold=10.0)
        
        adata.obs["gmm_labels"] = best_model["labels"]
        adata.obs["gmm_labels"] = adata.obs["gmm_labels"].astype("category")
        
        # 3. Graph operations
        graph_utils.get_neighbors(adata, type="delaunay", gmm_labels="gmm_labels")
        graph_utils.prune_graph(adata, type="percentile", percentile=95)
        graph_utils.set_seeds(adata, "1_nn_distance", spatial_connectivity_key="spatial_connectivities_pruned")

        # 4. Clustering Loop (handling flavors)
        clustering_methods = ["watershed"] # Add "leiden", "kmeans" here if needed later
        
        for flavor in clustering_methods:
            print(f"Running clustering: {flavor}")
            
            # Run the specific clustering flavor
            clustering.cluster_domains(
                adata, 
                spatial_connectivity_key="spatial_connectivities_pruned", 
                flavor=flavor, 
                distances_key="1_nn_distance"
            )

            # Define the key specifically for this flavor
            niche_key = f"{flavor}_niches"

            # Check if the key exists before plotting (safety check)
            if niche_key not in adata.obs:
                print(f"Warning: {niche_key} not found in obs. Skipping plot.")
                continue

            # Assign colors
            plotting.assign_colors(sdata, niche_key)

            # Extract plotting data
            coords = adata.obsm["spatial"]
            labels = adata.obs[niche_key]
            
            # Create palette
            if f"{niche_key}_colors" in adata.uns:
                category_colors = adata.uns[f"{niche_key}_colors"]
                categories = labels.cat.categories
                palette = dict(zip(categories, category_colors))
            else:
                # Fallback if custom colors fail
                palette = "tab20"

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 10))

            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=labels,
                palette=palette,
                s=1,
                linewidth=0,
                legend=False,
                ax=ax
            )

            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"Spatial Domains: {niche_key} ({dataset_name})")

            # Construct Output Filename: {input_zarr}_niches_{clustering_method}.png
            out_filename = f"{dataset_name}_niches_{flavor}.png"
            
            plt.savefig(out_filename, dpi=300, bbox_inches='tight')
            print(f"Saved image: {out_filename}")
            
            # Close the specific figure to free memory immediately
            plt.close(fig)

    except Exception as e:
        print(f"ERROR processing {dataset_name}: {e}")
        # Continue to next file even if this one fails

    finally:
        # 5. Memory Cleanup
        # Explicitly delete large objects
        if 'sdata' in locals(): del sdata
        if 'adata' in locals(): del adata
        if 'results' in locals(): del results
        
        # Force garbage collection
        gc.collect()
        
print("Processing complete.")
