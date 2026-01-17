import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
import seaborn as sns
import os
import gc
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
import networkx as nx

# --- Configuration ---
data_root = "data"
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
    },
]

# --- Initialize Global Figures ---
nrows, ncols = 3, 2
fig_size_base = (16, 20) 

# 1. Histogram Grid
fig_hist, axes_hist = plt.subplots(nrows, ncols, figsize=fig_size_base)
fig_hist.suptitle('KNN Distance Histograms (K=1) Across Datasets', fontsize=20, y=0.98)

# 2. GMM Decomposition Grid
fig_gmm, axes_gmm = plt.subplots(nrows, ncols, figsize=fig_size_base)
fig_gmm.suptitle('GMM Density Decomposition (Log-Normal Fit)', fontsize=20, y=0.98)

# 3. Model Selection Grid (BIC/AIC)
fig_metrics, axes_metrics = plt.subplots(nrows, ncols, figsize=fig_size_base)
fig_metrics.suptitle('Model Selection: AIC vs BIC & Elbow Cutoff', fontsize=20, y=0.98)

# 4. Spatial Map Grid
fig_spatial, axes_spatial = plt.subplots(nrows, ncols, figsize=fig_size_base)
fig_spatial.suptitle('Spatial Density Maps: Automated Segmentation', fontsize=20, y=0.98)


def get_log_normal_pdf(x_vals, log_mean, log_var, weight):
    """
    Helper to calculate PDF of Log-Normal distribution for plotting on linear axis.
    """
    sigma = np.sqrt(log_var)
    mu = log_mean
    # PDF equation for Log-Normal
    pdf = (np.exp(-(np.log(x_vals) - mu)**2 / (2 * sigma**2)) / (x_vals * sigma * np.sqrt(2 * np.pi)))
    return weight * pdf

def run_adaptive_niche_clustering(df, optimal_n, gmm_model, colors, title_prefix, 
                                  heuristic_scaler=1.0):
    """
    Constructs an adaptive spatial graph and clusters it to find niches.
    """
    print(f"--- Running Adaptive Niche Clustering: {title_prefix} ---")
    
    # 1. Assign Regimes and Determine Thresholds
    coords = df[['x_centroid', 'y_centroid']].values
    
    # Recalculate K=1 for the full dataset to ensure alignment
    nbrs_1 = NearestNeighbors(n_neighbors=2).fit(coords)
    dist_k1, _ = nbrs_1.kneighbors(coords)
    dist_k1 = dist_k1[:, 1]
    
    # Predict GMM component
    dist_log = np.log(dist_k1 + 1e-6).reshape(-1, 1)
    labels = gmm_model.predict(dist_log)
    
    # Calculate Adaptive Thresholds per Regime
    regime_thresholds = {}
    print("Adaptive Radius Thresholds:")
    for i in range(optimal_n):
        mean_log = gmm_model.means_[i, 0]
        cov_log = gmm_model.covariances_[i, 0, 0]
        std_log = np.sqrt(cov_log)
        
        log_threshold = mean_log + (1.0 * std_log) 
        linear_threshold = np.exp(log_threshold)
        adaptive_radius = linear_threshold * heuristic_scaler
        regime_thresholds[i] = adaptive_radius
        print(f"  Regime {i}: 1-NN mean ~{np.exp(mean_log):.2f}um -> Graph Radius: {adaptive_radius:.2f}um")

    cell_radii = np.array([regime_thresholds[l] for l in labels])
    
    # 2. Build the Adaptive Graph
    K_candidates = 50 
    nbrs_large = NearestNeighbors(n_neighbors=K_candidates).fit(coords)
    distances_large, indices_large = nbrs_large.kneighbors(coords)
    
    row_ind = []
    col_ind = []
    data = []
    
    n_cells = len(df)
    
    mask = distances_large <= cell_radii[:, None]
    
    for k in range(1, K_candidates): 
        valid_idx = mask[:, k]
        source_nodes = np.where(valid_idx)[0]
        target_nodes = indices_large[valid_idx, k]
        weights = 1.0 / (distances_large[valid_idx, k] + 1e-6) 
        
        row_ind.extend(source_nodes)
        col_ind.extend(target_nodes)
        data.extend(weights)
        
    adj_mat = csr_matrix((data, (row_ind, col_ind)), shape=(n_cells, n_cells))
    
    # 3. Clustering (Leiden)
    adata = ad.AnnData(X=csr_matrix((n_cells, 1))) 
    adata.obsp['spatial_connectivities'] = adj_mat
    adata.obsp['spatial_distances'] = adj_mat 
    adata.uns['spatial_neighbors'] = {'connectivities_key': 'spatial_connectivities', 'distances_key': 'spatial_distances'}
    
    sc.tl.leiden(adata, resolution=0.5, key_added='niche_cluster', neighbors_key='spatial_neighbors')
    
    df['niche_id'] = adata.obs['niche_cluster'].values
    
    # 4. Visualization of Niches
    plot_df = df.sample(min(len(df), 200000))
    
    plt.figure(figsize=(10, 10))
    sns.scatterplot(data=plot_df, x='x_centroid', y='y_centroid', 
                    hue='niche_id', palette='tab20', s=1, linewidth=0, alpha=0.9)
    
    plt.title(f"Adaptive Niches: {title_prefix}\n(Scaled Adaptive Radius)")
    plt.axis('equal')
    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=5)
    plt.tight_layout()
    plt.savefig(f"Niches_{title_prefix.replace(' ', '_')}.png", dpi=300)
    plt.close()
    
    print(f"  > Found {df['niche_id'].nunique()} niches. Saved plot.")
    return df

# --- (1/2) Calculate Optimal GMM Components ---
def find_optimal_gmm_components(dist_log_reshaped, n_range=range(1, 11)):
    """
    Calculates BIC scores for a range of components and uses the Elbow Method
    to find the optimal number of GMM components.
    
    Returns:
        optimal_n (int): The selected number of components.
        bic_scores (list): The list of BIC scores for plotting.
    """
    bic_scores = []
    
    # Subsample for model selection to speed up loop
    if len(dist_log_reshaped) > 100000:
        sample_idx = np.random.choice(len(dist_log_reshaped), 100000, replace=False)
        fit_data = dist_log_reshaped[sample_idx]
    else:
        fit_data = dist_log_reshaped

    for n in n_range:
        gmm_test = GaussianMixture(n_components=n, n_init=5, random_state=42)
        gmm_test.fit(fit_data)
        bic_scores.append(gmm_test.bic(fit_data))

    # Elbow detection logic
    bic_diffs = np.abs(np.diff(bic_scores))
    threshold = 0.02 * np.max(bic_diffs) 
    significant_changes = np.where(bic_diffs < threshold)[0]
    
    optimal_n = n_range[significant_changes[0]] if len(significant_changes) > 0 else n_range[np.argmin(bic_scores)]
    if optimal_n < 2: optimal_n = 2
    
    print(f"   > Optimal GMM Components: {optimal_n}")
    return optimal_n, bic_scores

# --- (2/2) Assign GMM Component IDs ---
def train_and_assign_gmm(dist_log_reshaped, optimal_n):
    """
    Fits the final GMM with optimal_n components, sorts components by mean (density),
    and assigns labels to the data.
    
    Returns:
        gmm (model): The fitted GaussianMixture object.
        final_labels (list): The re-mapped cluster labels for each cell.
        idx_sorted (array): The indices used to sort the components.
        sorted_means (array): The means of the components (in log space), sorted ascending.
    """
    # Final Fit (on Log Data)
    gmm = GaussianMixture(n_components=optimal_n, n_init=10, random_state=42)
    gmm.fit(dist_log_reshaped)

    # Sort components by mean (so Cluster 0 is always the smallest distance/highest density)
    idx_sorted = np.argsort(gmm.means_.flatten())
    
    # Extract the sorted means immediately
    sorted_means = gmm.means_[idx_sorted]
    
    # Predict raw clusters
    raw_labels = gmm.predict(dist_log_reshaped)
    
    # Create map from raw index to sorted index
    label_map = {old_idx: new_idx for new_idx, old_idx in enumerate(idx_sorted)}
    final_labels = [label_map[l] for l in raw_labels]
    
    return gmm, final_labels, idx_sorted, sorted_means

def process_single_dataset(file_path, title_prefix, ax_hist, ax_gmm, ax_metric, ax_spatial):
    """
    Loads data, runs LOG-TRANSFORMED analysis, plots to 4 main axes, 
    AND generates a separate per-component spatial breakdown figure.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Skipping.")
        return

    print(f"--- Processing: {title_prefix} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    count_col = 'total_counts' if 'total_counts' in df.columns else 'transcript_counts'
    if count_col in df.columns:
        df = df[df[count_col] > 0].copy()
    
    coords = df[['x_centroid', 'y_centroid']].values

    # 2. KNN Calculation (K=1)
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    dist_k1 = distances[:, 1]
    
    # --- TRANSFORM: Log-Transform (safe against 0) ---
    dist_log = np.log(dist_k1 + 1e-6)
    dist_log_reshaped = dist_log.reshape(-1, 1)

    # --- PLOT 1: Histogram (Linear View) ---
    sns.histplot(dist_k1, bins=400, kde=True, ax=ax_hist, color='teal', edgecolor=None, alpha=0.6)
    ax_hist.set_title(f"{title_prefix}\nMean Dist: {np.mean(dist_k1):.2f} $\mu m$", fontsize=12)
    ax_hist.set_xlim(0, 50)
    ax_hist.set_xlabel("Distance ($\mu m$)")

    # 3. Automated GMM Optimization (Using new function 1)
    n_range = range(1, 11)
    optimal_n, bic_scores = find_optimal_gmm_components(dist_log_reshaped, n_range)

    # --- PLOT 3: Model Selection ---
    ax_metric.plot(n_range, bic_scores, marker='o', label='BIC', color='blue')
    ax_metric.axvline(x=optimal_n, color='red', linestyle='--', label=f'Selected K={optimal_n}')
    ax_metric.set_title(f"{title_prefix} Model Selection")
    ax_metric.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_metric.legend()

    # 4. Final Fit and Assignment (Using new function 2)
    gmm, final_labels, idx_sorted, means_log = train_and_assign_gmm(dist_log_reshaped, optimal_n)
    print("Calculated optimal means/thresholds (log(x)):", means_log)
    
    # Assign labels to DataFrame
    df['cluster'] = final_labels

    # Extract sorted parameters for Plotting (Using idx_sorted to ensure colors match labels)
    means_log = gmm.means_[idx_sorted]
    covs_log = gmm.covariances_[idx_sorted]
    weights = gmm.weights_[idx_sorted]

    # --- PLOT 2: GMM Density Decomposition (Detailed) ---
    sns.histplot(dist_k1, bins=400, stat='density', ax=ax_gmm, color='lightgray', alpha=0.5, element="step", fill=True)
    
    x_axis = np.linspace(0.1, 55, 1000)
    total_pdf = np.zeros_like(x_axis) 
    
    colors = sns.color_palette("rocket_r", n_colors=optimal_n)

    for i in range(optimal_n):
        mu_log = means_log[i][0]
        var_log = covs_log[i][0][0]
        w = weights[i]
        
        # Compute component PDF
        y_component = get_log_normal_pdf(x_axis, mu_log, var_log, w)
        total_pdf += y_component
        
        geo_mean = np.exp(mu_log)
        label_text = f"C{i+1}: $\mu_{{geo}}$={geo_mean:.1f}"
        
        ax_gmm.plot(x_axis, y_component, color=colors[i], lw=2, label=label_text)
        ax_gmm.fill_between(x_axis, 0, y_component, color=colors[i], alpha=0.1)

    ax_gmm.plot(x_axis, total_pdf, color='black', linestyle='--', lw=1.5, label='Total')
    ax_gmm.set_title(f"{title_prefix} (N={optimal_n})")
    ax_gmm.set_xlabel("Distance ($\mu m$)")
    ax_gmm.set_xlim(0, 50) 
    ax_gmm.legend(fontsize='small', loc='upper right')

    # --- PLOT 4: Spatial Map (Combined) ---
    legend_handles = []
    cluster_centers_um = np.exp(means_log)

    for i in range(optimal_n):
        mask = df['cluster'] == i
        subset = df[mask]
        if len(subset) > 500000: subset = subset.sample(500000)
            
        ax_spatial.scatter(subset['x_centroid'], subset['y_centroid'], s=0.05, c=[colors[i]], alpha=0.8)
        
        geo_mean = cluster_centers_um[i][0]
        label_text = f"C{i+1}: $\mu_{{geo}}$={geo_mean:.1f}"
        
        handle = mlines.Line2D([], [], color='white', marker='o', 
                               markerfacecolor=colors[i], markersize=8, label=label_text)
        legend_handles.append(handle)

    ax_spatial.set_title(f"{title_prefix}")
    ax_spatial.set_aspect('equal')
    ax_spatial.invert_yaxis()
    ax_spatial.axis('off')
    ax_spatial.legend(handles=legend_handles, loc='upper right', fontsize='small')

    # =========================================================================
    # NEW: PLOT 5 - Independent Spatial Breakdown (Grid per Tissue)
    # =========================================================================
    
    n_cols_bd = min(optimal_n, 4)
    n_rows_bd = (optimal_n + n_cols_bd - 1) // n_cols_bd 
    
    fig_bd, axes_bd = plt.subplots(n_rows_bd, n_cols_bd, 
                                   figsize=(5 * n_cols_bd, 5 * n_rows_bd), 
                                   constrained_layout=True)
    
    fig_bd.suptitle(f"Component Breakdown: {title_prefix}", fontsize=20)
    
    if optimal_n > 1:
        axes_bd_flat = axes_bd.flatten()
    else:
        axes_bd_flat = [axes_bd]

    for i in range(len(axes_bd_flat)):
        ax = axes_bd_flat[i]
        
        if i < optimal_n:
            mask = df['cluster'] == i
            subset = df[mask]
            
            if len(subset) > 500000: subset = subset.sample(500000)
            
            ax.scatter(subset['x_centroid'], subset['y_centroid'], s=0.05, c=[colors[i]], alpha=0.8)
            
            geo_mean = cluster_centers_um[i][0]
            ax.set_title(f"Component {i+1}\n($\mu_{{geo}}$={geo_mean:.1f} $\mu m$)", 
                         color=colors[i], fontweight='bold')
            ax.set_aspect('equal')
            ax.invert_yaxis()
            ax.axis('off')
        else:
            ax.axis('off')
    
    safe_name = title_prefix.replace(" ", "_")
    save_name = f"Breakdown_{safe_name}.png"
    fig_bd.savefig(save_name, dpi=200)
    plt.close(fig_bd) 
    print(f"   > Saved breakdown: {save_name}")

    # =========================================================================

    # adaptive clustering
    run_adaptive_niche_clustering(df, optimal_n, gmm, colors, title_prefix)

    del df, coords, nbrs, distances, gmm
    gc.collect()

# --- Main Execution ---
for row_idx, item in enumerate(data_pairs):
    tissue = item['tissue']
    h_path = os.path.join(data_root, item['human_dir'], 'cells.csv.gz')
    m_path = os.path.join(data_root, item['mouse_dir'], 'cells.csv.gz')
    
    process_single_dataset(h_path, f"Human {tissue}", axes_hist[row_idx, 0], axes_gmm[row_idx, 0], axes_metrics[row_idx, 0], axes_spatial[row_idx, 0])
    process_single_dataset(m_path, f"Mouse {tissue}", axes_hist[row_idx, 1], axes_gmm[row_idx, 1], axes_metrics[row_idx, 1], axes_spatial[row_idx, 1])

# Save GMM
fig_gmm.tight_layout(rect=[0, 0, 1, 0.96])
fig_gmm.savefig("Multi_Tissue_GMM_Detailed.png", dpi=300)
print("Saved: Multi_Tissue_GMM_Detailed.png")

# Save Spatial Map (New)
fig_spatial.tight_layout(rect=[0, 0, 1, 0.96])
fig_spatial.savefig("Multi_Tissue_Spatial_Density_Map.png", dpi=300)
print("Saved: Multi_Tissue_Spatial_Density_Map.png")