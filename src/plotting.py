import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squidpy as sq
import math
from spatialdata import SpatialData
from matplotlib.collections import LineCollection

def assign_colors(adata, column_key, seed=42):
    """
    Assign unique colors to each category in a given column.
    Category 0 is always assigned grey.
    """
    if isinstance(adata, SpatialData):
        adata = adata.tables["table"]
    else:
        adata = adata
    
    obs = adata.obs[column_key]

    # get sorted unique categories
    cats = np.array(sorted(obs.unique()))

    rng = np.random.default_rng(seed)

    colors = []
    seen = set()

    for c in cats:
        if c == 0:
            # fixed color for background / unassigned
            colors.append("#333333")  # dark grey
            continue

        # sample until we get a new color
        while True:
            h = rng.random()
            s = rng.uniform(0.6, 1.0)
            v = rng.uniform(0.7, 1.0)

            rgb = mcolors.hsv_to_rgb((h, s, v))
            hex_color = mcolors.to_hex(rgb, keep_alpha=False)

            if hex_color not in seen:
                seen.add(hex_color)
                colors.append(hex_color)
                break

    adata.uns[f"{column_key}_colors"] = colors

    
def plot_scores(data):
    scores = ["bic", "aic", "icl"]
    
    score_colors = {
        "bic": "lightblue",
        "aic": "orange",
        "icl": "black",
    }

    keys = sorted(list(data.keys()))

    fig, ax = plt.subplots()

    for score in scores:
        y = [data[k][score] for k in keys]
        
        ax.plot(
            keys,
            y,
            marker="o",
            color=score_colors[score],
            label=score,
        )

        min_score_val = min(y)              
        min_index = y.index(min_score_val)  
        best_k = keys[min_index]            
        
        # add vertical indicator of lowest
        ax.axvline(
            x=best_k, 
            color=score_colors[score], 
            linestyle=":", 
            linewidth=1.5,
            alpha=0.7 
        )

    ax.set_xlabel("Number of components")
    ax.set_ylabel("Score")
    ax.set_title("Scores by number of components")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_knn_by_regime(adata, labels_key="gmm_labels", k=1, cmap="viridis", size=1, figsize=None):
    """
    Plots the k-th NN distances split by density regimes (labels_key) side-by-side.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing coordinates and results.
    labels_key : str
        The column in adata.obs containing the density regime labels (e.g., 'gmm_labels').
    k : int
        The k-th neighbor index used for coloring (e.g., k=1 looks for '1_nn_distance').
    cmap : str
        Colormap for the distance values.
    size : float
        Point size for the scatter plot.
    figsize : tuple, optional
        Figure size (width, height). If None, it is calculated automatically.
    """
    
    dist_key = f"{k}_nn_distance"
    
    if dist_key not in adata.obs:
        raise ValueError(f"Distance key '{dist_key}' not found. Run get_distances(adata, k={k}) first.")
    
    if labels_key not in adata.obs:
        raise ValueError(f"Labels key '{labels_key}' not found in adata.obs.")

    unique_labels = adata.obs[labels_key].unique()
    try:
        unique_labels = sorted(unique_labels)
    except TypeError:
        pass # Handle cases where labels are mixed types

    n_plots = len(unique_labels)
    
    n_cols = 3 if n_plots >= 3 else n_plots
    n_rows = math.ceil(n_plots / n_cols)
    
    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=figsize, constrained_layout=True)
    
    if n_plots > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    vmin = adata.obs[dist_key].min()
    vmax = adata.obs[dist_key].max()

    for i, label in enumerate(unique_labels):
        ax = axes[i]
        subset = adata[adata.obs[labels_key] == label].copy()
        
        sq.pl.spatial_scatter(
            subset,
            color=dist_key,
            ax=ax,
            size=size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            shape=None,
            title=f"Regime: {label}",
            colorbar=False 
        )
        
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Hide any empty subplots slots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(f"Log Distance (k={k})")

    plt.suptitle(f"Spatial Density Regimes ({labels_key})", fontsize=16)

        
def plot_niche_stats(
    adata,
    col_keys,
    sort_by,
    category_key,
    normalize=True,
):
    category_col = category_key

    # --- split requested columns ---
    raw_cols = [c for c in col_keys if c != "n_obs"]
    use_n_obs = "n_obs" in col_keys

    # -------------------------------------------------
    # exclude category 0 only when n_obs is involved
    # -------------------------------------------------
    if use_n_obs:
        obs_use = adata.obs[adata.obs[category_col] != 0]
    else:
        obs_use = adata.obs

    # --- aggregate means for raw columns ---
    df_agg = (
        obs_use
        .groupby(category_col, as_index=False)[raw_cols]
        .mean()
    )

    # --- optionally compute n_obs ---
    if use_n_obs:
        df_count = (
            obs_use
            .groupby(category_col)
            .size()
            .reset_index(name="n_obs")
        )
        df_agg = df_agg.merge(df_count, on=category_col)

    # --- columns to plot ---
    plot_cols = raw_cols + (["n_obs"] if use_n_obs else [])

    # --- validate sort column ---
    if sort_by not in plot_cols:
        raise ValueError(
            f"sort_by='{sort_by}' not in col_keys. "
            f"Available columns: {plot_cols}"
        )

    # --- normalize (optional) ---
    if normalize:
        df_plot = df_agg.copy()

        denom = df_plot[plot_cols].max() - df_plot[plot_cols].min()
        denom = denom.replace(0, 1)

        df_plot[plot_cols] = (
            df_plot[plot_cols] - df_plot[plot_cols].min()
        ) / denom
        y_label = "Normalized value (0–1)"
    else:
        df_plot = df_agg.copy()
        y_label = "Value"

    # --- sort (always on plotted values) ---
    df_plot = df_plot.sort_values(by=sort_by).reset_index(drop=True)

    # --- plot ---
    plt.figure(figsize=(8, 5))

    for col in plot_cols:
        is_sort = col == sort_by

        plt.plot(
            df_plot.index,
            df_plot[col],
            label=col,
            alpha=1.0 if is_sort else 0.4,
            linewidth=3 if is_sort else 2,
            zorder=10 if is_sort else 1
        )

    plt.xlabel(f"niches sorted by {sort_by}")
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_knn_histogram(adata, k=1, bins=50, title=None, ax=None, color='#4c72b0', kde=True, **kwargs):
    """
    Plots the distribution (histogram) of the k-th nearest neighbor distances.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    k : int
        The k-th neighbor index (looks for column '{k}_nn_distance').
    bins : int
        Number of histogram bins.
    title : str, optional
        Custom title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on.
    color : str
        Bar color.
    kde : bool
        Whether to plot the Kernel Density Estimate line.
    **kwargs
        Additional arguments passed to sns.histplot.
    """
    key = f"{k}_nn_distance"
    
    if key not in adata.obs:
        raise ValueError(f"Key '{key}' not found in adata.obs. Run get_distances(adata, k={k}) first.")
        
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        
    if title is None:
        title = f"Distribution of Log Distances (k={k})"

    sns.histplot(
        data=adata.obs, 
        x=key, 
        bins=bins, 
        kde=kde, 
        color=color, 
        edgecolor=None,
        ax=ax,
        **kwargs
    )
    
    ax.set_title(title)
    ax.set_xlabel("Log Distance")
    ax.set_ylabel("Frequency")
    
    mean_val = adata.obs[key].mean()
    ax.axvline(mean_val, color='k', linestyle='--', alpha=0.5, label=f'Mean: {mean_val:.2f}')
    
    if ax is None:
        plt.show()
        
def plot_pruned_connections(
    adata, 
    gmm_key, 
    basis="spatial", 
    prune_by="label", 
    distance_key=None, 
    n_std=2.0, 
    point_size=5,
    mode="comparison",
    ax=None,
    figsize=(16, 8)
):
    """
    Visualizes graph pruning.
    If prune_by="distance", compares log(spatial_distances) vs log-thresholds.
    """
    
    # --- 1. Data Retrieval ---
    if f"X_{basis}" in adata.obsm:
        coords = adata.obsm[f"X_{basis}"]
    elif basis in adata.obsm:
        coords = adata.obsm[basis]
    else:
        raise KeyError(f"Could not find coordinates in adata.obsm['{basis}']")

    adj_matrix = adata.obsp["spatial_distances"].tocoo()
    sources = adj_matrix.row
    targets = adj_matrix.col
    weights = adj_matrix.data # These are linear distances
    
    # --- 2. Determine Edges to Keep ---
    labels = adata.obs[gmm_key].values

    if prune_by == "label":
        edge_is_kept = labels[sources] == labels[targets]

    elif prune_by == "distance":
        if distance_key is None:
            raise ValueError("Must provide 'distance_key' when prune_by='distance'")
            
        # Stats on LOG values
        stats = adata.obs.groupby(gmm_key, observed=False)[distance_key].agg(['mean', 'std'])
        stats['std'] = stats['std'].fillna(0)
        
        # Calc LOG threshold
        stats['log_threshold'] = stats['mean'] + (n_std * stats['std'])
        
        # Map to cells
        mapping_dict = stats['log_threshold'].to_dict()
        cell_thresholds = adata.obs[gmm_key].map(mapping_dict).astype(float).values
        
        # Log the graph weights
        log_weights = np.log(weights + 1e-12)
        
        # Compare Log Weights vs Log Thresholds
        thresholds_per_edge = cell_thresholds[sources]
        edge_is_kept = log_weights <= thresholds_per_edge
        
    else:
        raise ValueError(f"Unknown pruning strategy: {prune_by}")

    # --- 3. Build Segments ---
    start_points = coords[sources]
    end_points = coords[targets]
    segments = np.stack((start_points, end_points), axis=1)

    kept_segments = segments[edge_is_kept]
    pruned_segments = segments[~edge_is_kept]
    
    n_kept = len(kept_segments)
    n_pruned = len(pruned_segments)

    # --- 4. Plotting ---
    def draw_graph(axis, show_pruned=True, title=""):
        axis.scatter(coords[:, 0], coords[:, 1], c=labels, s=point_size, cmap='tab20', zorder=10)
        
        lc_kept = LineCollection(kept_segments, colors='lightgray', linewidths=0.5, alpha=0.5, zorder=1)
        axis.add_collection(lc_kept)
        
        if show_pruned:
            lc_pruned = LineCollection(pruned_segments, colors='red', linewidths=1.0, alpha=0.8, zorder=5)
            axis.add_collection(lc_pruned)
            
        axis.set_title(title)
        axis.axis('equal')
        axis.axis('off')

    if mode == "comparison":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        draw_graph(ax1, show_pruned=True, title=f"Candidates (Red)\ntotal connections: {n_pruned + n_kept}")
        draw_graph(ax2, show_pruned=False, title=f"Final Graph\nconnections after pruning: {n_kept}")
        return fig, (ax1, ax2)
    else:
        if ax is None: fig, ax = plt.subplots(figsize=(8, 8))
        show_p = (mode == "highlight")
        draw_graph(ax, show_pruned=show_p, title=f"Graph ({mode})")
        return ax