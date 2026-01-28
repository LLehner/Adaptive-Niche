import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squidpy as sq
import math
from spatialdata import SpatialData

def assign_colors(sdata, column_key, seed=42):
    """
    Assign unique colors to each category in a given column. 
    """
    rng = np.random.default_rng(seed)

    colors = []
    seen = set()
    n_colors = sdata.tables["table"].obs[column_key].nunique()

    while len(colors) < n_colors:
        # Sample continuous HSV
        h = rng.random()
        s = rng.uniform(0.6, 1.0)  # avoid washed-out colors
        v = rng.uniform(0.7, 1.0)

        rgb = mcolors.hsv_to_rgb((h, s, v))
        hex_color = mcolors.to_hex(rgb, keep_alpha=False)

        if hex_color not in seen:
            seen.add(hex_color)
            colors.append(hex_color)

    sdata.tables["table"].uns[f"{column_key}_colors"] = colors
    
import matplotlib.pyplot as plt

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


def plot_knn_distance(adata, k=1, **kwargs):
    """
    Plots the spatial scatter colored by the k-th nearest neighbor distance.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix.
    k : int
        The k-th neighbor index used to generate the distance column (e.g., '1_nn_distance').
    **kwargs
        Additional arguments passed to sq.pl.spatial_scatter (e.g., size, cmap, shape).
    """
    key = f"{k}_nn_distance"
    
    # Safety check
    if key not in adata.obs:
        raise ValueError(f"Key '{key}' not found in adata.obs. Please run get_distances(adata, k={k}) first.")

    # Set default title if not provided in kwargs
    title = kwargs.pop("title", f"Log Distance to {k}-th NN")
    
    # Set default cmap if not provided
    if "cmap" not in kwargs:
        kwargs["cmap"] = "viridis"

    sq.pl.spatial_scatter(
        adata,
        color=key,
        title=title,
        shape=None,
        **kwargs
    )

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

def plot_niches(data, niche_key="watershed_niches", size=3, dpi=300, output_path=None, title=None, ax=None):
    """
    Plots spatial niches using Squidpy with consistent coloring.
    
    Parameters
    ----------
    data : SpatialData | AnnData
        The SpatialData object (or AnnData). 
        If SpatialData, it extracts the table automatically.
    niche_key : str
        The key in .obs containing the niche/cluster labels.
    size : float
        Point size for the scatter plot.
    dpi : int
        Resolution for saving/plotting.
    output_path : str, optional
        If provided, saves the figure to this path (e.g., "niches.png").
    title : str, optional
        Custom title. If None, uses "Spatial Domains: {niche_key}".
    ax : matplotlib.axes.Axes, optional
        A specific axes to plot on. If None, creates a new figure.
    """
    
    if isinstance(data, SpatialData):
        adata = data.tables["table"]
        assign_colors(data, niche_key)
    else:
        adata = data
        try:
            assign_colors(adata, niche_key)
        except:
            pass 

    if niche_key in adata.obs:
        adata.obs[niche_key] = adata.obs[niche_key].astype("category")

    if title is None:
        title = f"Spatial Domains: {niche_key}"

    sq.pl.spatial_scatter(
        adata,
        color=niche_key,
        size=size,
        title=title,
        shape=None,      # Faster rendering for pure points
        legend_loc=None, # Hides legend (as requested in your snippet)
        ax=ax,
        dpi=dpi,
        figsize=(10, 10) if ax is None else None
    )

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    if ax is None and not output_path:
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