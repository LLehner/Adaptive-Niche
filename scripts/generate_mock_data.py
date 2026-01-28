import os
import shutil
import numpy as np
import pandas as pd
import anndata as ad
import spatialdata as sd
from spatialdata.models import TableModel
from sklearn.datasets import make_moons, make_circles, make_blobs

# Configuration
OUTPUT_NAME = "debug_dataset.zarr"
N_POINTS_DENSE = 1000
N_POINTS_SPARSE = 300
N_NOISE = 200

def generate_mock_data():
    print("Generating synthetic coordinates...")
    
    # 1. Dense Cluster (Tight Gaussian blob)
    # Centered at (5, 5)
    dense_blobs, _ = make_blobs(n_samples=N_POINTS_DENSE, centers=[(5, 5)], cluster_std=0.5)
    
    # 2. Sparse Cluster (Loose Gaussian blob)
    # Centered at (15, 5)
    sparse_blobs, _ = make_blobs(n_samples=N_POINTS_SPARSE, centers=[(15, 5)], cluster_std=2.5)
    
    # 3. Intertwined 'C' shapes (Moons)
    # Shifted to (5, 15)
    moons, _ = make_moons(n_samples=600, noise=0.05)
    moons = moons * 3  # Scale up
    moons[:, 0] += 5   # Shift X
    moons[:, 1] += 15  # Shift Y
    
    # 4. 'O' shape (Circles - Outer and Inner ring)
    # Shifted to (15, 15)
    circles, _ = make_circles(n_samples=600, factor=0.5, noise=0.05)
    circles = circles * 3 # Scale up
    circles[:, 0] += 15
    circles[:, 1] += 15
    
    # 5. Random Noise (Uniform distribution over the whole area)
    # Bounding box roughly (0,0) to (20,20)
    noise = np.random.uniform(low=-2, high=22, size=(N_NOISE, 2))

    # Combine all points
    all_coords = np.vstack([dense_blobs, sparse_blobs, moons, circles, noise])
    
    # Create labels for ground truth (optional, helpful for debugging visualization)
    labels = (
        ['dense'] * len(dense_blobs) +
        ['sparse'] * len(sparse_blobs) +
        ['moons'] * len(moons) +
        ['circles'] * len(circles) +
        ['noise'] * len(noise)
    )

    print(f"Total points generated: {len(all_coords)}")
    
    # --- Create AnnData ---
    # Your pipeline expects no gene expression, so we create an empty X matrix
    # But we ensure obs/var are initialized to avoid generic errors
    X = np.zeros((len(all_coords), 1)) # Single dummy variable
    
    obs = pd.DataFrame({
        "ground_truth": labels,
        "cell_id": [f"cell_{i}" for i in range(len(all_coords))]
    })
    obs.set_index("cell_id", inplace=True)
    
    adata = ad.AnnData(X=X, obs=obs)
    
    # KEY STEP: Populate obsm['spatial'] as your script uses coords = adata.obsm["spatial"]
    adata.obsm["spatial"] = all_coords
    
    # --- Wrap in SpatialData ---
    # We parse the AnnData as a TableModel. 
    # Note: Usually SpatialData requires a 'region' link, but for pure coordinate 
    # processing pipelines dependent on obsm['spatial'], an unlinked table often works 
    # or we can link it to a dummy region.
    
    # To be safe for sd.read_zarr, we strictly format it.
    # We will assume a dummy "points" region exists implicitly or just save the table.
    # Your script reads `sdata.tables["table"]`, so we must put it there.
    
    # Parse validates the AnnData for SpatialData storage
    adata = TableModel.parse(adata)
    
    # Create the SpatialData object
    sdata = sd.SpatialData(tables={"table": adata})
    
    return sdata

def main():
    # Clean up previous run
    if os.path.exists(OUTPUT_NAME):
        shutil.rmtree(OUTPUT_NAME)
        
    sdata = generate_mock_data()
    
    print(f"Saving to {OUTPUT_NAME}...")
    sdata.write(OUTPUT_NAME)
    print("Done! You can now point your pipeline INPUT_DIR to this folder's parent.")

if __name__ == "__main__":
    main()