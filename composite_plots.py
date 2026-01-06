import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import os
import gc

# --- Configuration ---
# Define the pairs (Human, Mouse) for each tissue
data_pairs = [
    {
        "tissue": "Bone",
        "human_dir": "H_Bone_Xenium_V1_hBoneMarrow_acute_lymphoid_leukemia_section_outs",
        "mouse_dir": "M_Bone_Xenium_V1_mFemur_formic_acid_24hrdecal_section_outs"
    },
    {
        "tissue": "Brain",
        "human_dir": "H_Brain_Xenium_V1_FFPE_Human_Brain_Healthy_With_Addon_outs (1)",
        "mouse_dir": "M_Brain_Xenium_V1_FF_Mouse_Brain_MultiSection_1_outs"
    },
    {
        "tissue": "Colon",
        "human_dir": "H_Colon_Xenium_V1_hColon_Non_diseased_Base_FFPE_outs",
        "mouse_dir": "M_Colon_Xenium_V1_mouse_Colon_FF_outs"
    }
]

base_path = 'data'
output_filename = "Human_vs_Mouse_Density_Comparison.png"

# Setup the figure grid: 3 Rows (Tissues) x 4 Columns (H_Map, H_Hist, M_Map, M_Hist)
fig, axes = plt.subplots(3, 4, figsize=(24, 18), constrained_layout=True)
fig.suptitle('Comparative Cellular Density: Human vs Mouse (Xenium)', fontsize=20, weight='bold')

def get_knn_data(file_path):
    """
    Loads data, calculates k-NN distances, and returns coordinates and distances.
    Does NOT return the dataframe to save memory.
    """
    full_path = os.path.join(base_path, file_path, 'cells.csv.gz')
    print(f"--> Loading {full_path}...")
    
    try:
        # Load only necessary columns to save memory
        df = pd.read_csv(full_path, usecols=['x_centroid', 'y_centroid'])
    except FileNotFoundError:
        print(f"Error: File not found at {full_path}")
        return None, None

    coords = df[['x_centroid', 'y_centroid']].values
    
    # K-NN Calculation
    print(f"    Computing k-NN for {len(df)} cells...")
    nbrs = NearestNeighbors(n_neighbors=11, algorithm='kd_tree', n_jobs=-1).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    
    # Extract specific k distances
    dist_k1 = distances[:, 1]
    dist_k5 = distances[:, 5]
    dist_k10 = distances[:, 10]
    
    # Clean up massive objects
    del df
    del nbrs
    del distances
    gc.collect()
    
    return coords, (dist_k1, dist_k5, dist_k10)

def plot_density_row(row_idx, tissue_name, h_dir, m_dir):
    """
    Handles the plotting for a single row (Tissue) containing Human and Mouse data.
    """
    
    # --- Process Human ---
    print(f"\nProcessing Human {tissue_name}...")
    h_coords, h_dists = get_knn_data(h_dir)
    
    if h_coords is not None:
        # Ax 0: Human Density Map (using k=10)
        ax_map = axes[row_idx, 0]
        # Calculate upper limit for color scaling (robust max)
        vmax = np.percentile(h_dists[2], 95) 
        
        scatter = ax_map.scatter(
            h_coords[:, 0], h_coords[:, 1],
            c=h_dists[2], s=0.3, cmap='viridis_r', alpha=0.8, rasterized=True, vmax=vmax
        )
        ax_map.set_title(f"Human {tissue_name} Density Map\n(Color: Dist to 10th NN)", fontsize=12)
        ax_map.set_aspect('equal')
        ax_map.invert_yaxis()
        ax_map.axis('off') # Cleaner look
        plt.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04, label='Dist ($\mu m$)')

        # Ax 1: Human Histograms (Overlay k=1, 5, 10)
        ax_hist = axes[row_idx, 1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
        labels = ['k=1', 'k=5', 'k=10']
        
        for i, dist in enumerate(h_dists):
            sns.kdeplot(dist, ax=ax_hist, fill=True, label=labels[i], color=colors[i], alpha=0.3, linewidth=1.5)
        
        ax_hist.set_title(f"Human {tissue_name} Neighbor Distances", fontsize=12)
        ax_hist.set_xlabel('Distance ($\mu m$)')
        ax_hist.set_ylabel('Density')
        ax_hist.set_xlim(0, 100) # Cap x-axis for readability
        ax_hist.legend()
        
        # Cleanup Human Data from Memory
        del h_coords, h_dists
        gc.collect()

    # --- Process Mouse ---
    print(f"Processing Mouse {tissue_name}...")
    m_coords, m_dists = get_knn_data(m_dir)
    
    if m_coords is not None:
        # Ax 2: Mouse Density Map
        ax_map = axes[row_idx, 2]
        vmax = np.percentile(m_dists[2], 95)
        
        scatter = ax_map.scatter(
            m_coords[:, 0], m_coords[:, 1],
            c=m_dists[2], s=0.3, cmap='viridis_r', alpha=0.8, rasterized=True, vmax=vmax
        )
        ax_map.set_title(f"Mouse {tissue_name} Density Map\n(Color: Dist to 10th NN)", fontsize=12)
        ax_map.set_aspect('equal')
        ax_map.invert_yaxis()
        ax_map.axis('off')
        plt.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04, label='Dist ($\mu m$)')

        # Ax 3: Mouse Histograms
        ax_hist = axes[row_idx, 3]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, dist in enumerate(m_dists):
            sns.kdeplot(dist, ax=ax_hist, fill=True, label=labels[i], color=colors[i], alpha=0.3, linewidth=1.5)
        
        ax_hist.set_title(f"Mouse {tissue_name} Neighbor Distances", fontsize=12)
        ax_hist.set_xlabel('Distance ($\mu m$)')
        ax_hist.set_xlim(0, 100)
        ax_hist.legend()

        # Cleanup Mouse Data
        del m_coords, m_dists
        gc.collect()

# --- Execution Loop ---
for i, item in enumerate(data_pairs):
    plot_density_row(i, item['tissue'], item['human_dir'], item['mouse_dir'])

print("Saving high-quality plot...")
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"Done! Saved to {output_filename}")