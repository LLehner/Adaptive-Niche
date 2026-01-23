import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# Xenium's 'cells.csv.gz' file contains centroids
file_path = 'data/Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs/cells.csv.gz'

print(f"Loading data from {file_path}...")
df = pd.read_csv(file_path)

coords = df[['x_centroid', 'y_centroid']].values

print(f"Loaded {len(df)} cells. Computing neighbors...")

nbrs = NearestNeighbors(n_neighbors=11, algorithm='kd_tree').fit(coords)
print("Calculating KNN")
distances, indices = nbrs.kneighbors(coords)
print("finished KNN")

dist_k1  = distances[:, 1]
dist_k5  = distances[:, 5]
dist_k10 = distances[:, 10]

k_values = [1, 5, 10]
data_map = {1: dist_k1, 5: dist_k5, 10: dist_k10}

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
fig.suptitle('Distributions of Distance to K-th Nearest Neighbor', fontsize=16)

for i, k in enumerate(k_values):
    ax = axes[i]
    data = data_map[k]
    
    sns.histplot(data, bins=50, kde=True, ax=ax, color='teal', edgecolor='black', alpha=0.6)
    
    ax.set_title(f'K = {k}')
    ax.set_xlabel(f'Distance to {k}-th NN (microns)')
    ax.set_ylabel('Number of Cells (Frequency)')
    
    # Add mean/median lines for reference
    ax.axvline(np.mean(data), color='r', linestyle='--', label=f'Mean: {np.mean(data):.2f}')
    ax.axvline(np.median(data), color='y', linestyle='--', label=f'Median: {np.median(data):.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig("k1_5_10_Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.png")
# plt.show()


plt.clf()

import matplotlib.pyplot as plt
import numpy as np

df['dist_k10'] = dist_k10
fig, ax = plt.subplots(1, 2, figsize=(20, 8))
vmax_val = np.percentile(df['dist_k10'], 95)

scatter = ax[0].scatter(
    df['x_centroid'], 
    df['y_centroid'], 
    c=df['dist_k10'], 
    s=0.5,                # Keep point size small for Xenium data
    cmap='viridis_r',     # '_r' reverses the map: Yellow=Dense (Low Dist), Purple=Sparse (High Dist)
    alpha=0.8,
    vmax=vmax_val
)

cbar = plt.colorbar(scatter, ax=ax[0])
cbar.set_label('Distance to 10th NN (microns)')

ax[0].set_title('Spatial Density Map (Based on K=10 Distance)')
ax[0].set_aspect('equal')
ax[0].invert_yaxis() 
threshold = np.mean(df['dist_k10']) 
colors = np.where(df['dist_k10'] < threshold, '#d62728', '#d3d3d3')

ax[1].scatter(
    df['x_centroid'], 
    df['y_centroid'], 
    c=colors, 
    s=0.5, 
    alpha=0.6
)

ax[1].set_title(f'Binary Niche Segmentation (Threshold < {threshold:.1f} microns)')
ax[1].set_aspect('equal')
ax[1].invert_yaxis() 

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label='High Density Niche'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d3d3d3', label='Low Density / Background')
]
ax[1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig("density_Xenium_Preview_Human_Non_diseased_Lung_With_Add_on_FFPE_outs.png")
plt.show()