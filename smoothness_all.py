import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
import spatialdata as sd
import tqdm

from src.pipeline import predict_niches
from src.scores import signal_smoothness


def compare_smoothness_all_zarrs(
    input_dir="../single_cell/data/zarr_outputs_no_images/",
    ks=(2, 5, 10, 20),
    watershed_method="delaunay",
    normalize_features=True,
    save_plots=True,
    show_plots=False,
    plot_dir="smoothness_comparison_plots",
    save_tables=True,
    table_dir="smoothness_comparison_tables",
    niche_size_bins=50,
    min_cells_per_niche=3,
    drop_isolates=True,
    save_niche_level_tables=True,
):
    """
    Run watershed-vs-KNN niche-level smoothness comparison over all .zarr datasets.

    For each dataset:
      1. Load SpatialData
      2. Predict watershed niches
      3. Compute smoothness separately within each watershed niche
      4. Compute smoothness separately within each ego-KNN neighborhood
      5. Summarize each method by mean niche smoothness
      6. Plot:
            - left panel: watershed reference vs KNN niche-level smoothness
            - right panel: watershed niche size histogram
      7. Save one CSV per dataset, optional per-niche CSVs, and one combined summary CSV

    Important:
    ----------
    - Watershed niches are defined by `adata.obs["watershed_by_descent"]`
    - KNN niches are defined as ego neighborhoods:
          niche(center i) = {i} U neighbors(i)
    - Error bars are std across niches, NOT std across genes
    """

    def knn_graph(coords, k):
        """
        Build a symmetric binary KNN adjacency matrix.
        """
        nbrs = NearestNeighbors(n_neighbors=k + 1)
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)

        n = coords.shape[0]
        rows = np.repeat(np.arange(n), k)
        cols = indices[:, 1:].reshape(-1)  # exclude self
        data = np.ones(len(rows), dtype=np.float64)

        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
        A = A.maximum(A.T).tocsr()
        A.setdiag(0)
        A.eliminate_zeros()
        return A

    def safe_name(path_or_name):
        """
        Make a filesystem-safe dataset label.
        """
        base = os.path.basename(path_or_name.rstrip("/"))
        base = re.sub(r"\.zarr$", "", base)
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", base)
        return base

    def prepare_subgraph(A_sub, X_sub, min_cells=3, drop_isolates=True):
        """
        Clean a niche subgraph before scoring.
        """
        if A_sub.shape[0] < min_cells:
            return None, None

        A_sub = A_sub.tocsr()
        A_sub.setdiag(0)
        A_sub.eliminate_zeros()

        if drop_isolates:
            deg = np.asarray(A_sub.sum(axis=1)).ravel()
            keep = deg > 0

            if keep.sum() < min_cells:
                return None, None

            A_sub = A_sub[keep][:, keep].tocsr()
            X_sub = X_sub[keep]

        if A_sub.shape[0] < min_cells or A_sub.nnz == 0:
            return None, None

        return A_sub, X_sub

    def niche_smoothness_from_labels(
        adjacency,
        features,
        niche_labels,
        normalize_features=True,
        min_cells=3,
        drop_isolates=True,
    ):
        """
        Compute one smoothness score per niche defined by label assignments.

        Returns
        -------
        niche_df : pd.DataFrame
            One row per niche.
        summary : dict
            Aggregate summary across niches.
        """
        niche_labels = np.asarray(niche_labels)
        unique_labels = pd.unique(niche_labels)

        rows = []

        for niche_id in tqdm.tqdm(unique_labels):
            idx = np.flatnonzero(niche_labels == niche_id)

            if len(idx) < min_cells:
                continue

            A_sub = adjacency[idx][:, idx].tocsr()
            X_sub = features[idx]

            A_sub, X_sub = prepare_subgraph(
                A_sub,
                X_sub,
                min_cells=min_cells,
                drop_isolates=drop_isolates,
            )

            if A_sub is None:
                continue

            res = signal_smoothness(
                adjacency=A_sub,
                features=X_sub,
                normalize_features=normalize_features
            )

            rows.append({
                "niche_id": niche_id,
                "n_cells": A_sub.shape[0],
                "smoothness": res["mean"],              # scalar score for this niche
                "median_gene_smoothness": res["median"],
                "genewise_std": res["std"],
                "random_baseline": res["random_baseline"],
                "graph_nnz": A_sub.nnz,
            })

        niche_df = pd.DataFrame(rows)

        if niche_df.empty:
            summary = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "weighted_mean": np.nan,
                "n_niches_used": 0,
            }
        else:
            summary = {
                "mean": niche_df["smoothness"].mean(),
                "median": niche_df["smoothness"].median(),
                "std": niche_df["smoothness"].std(ddof=1) if len(niche_df) > 1 else 0.0,
                "weighted_mean": np.average(
                    niche_df["smoothness"],
                    weights=niche_df["n_cells"]
                ),
                "n_niches_used": len(niche_df),
            }

        return niche_df, summary

    def niche_smoothness_from_ego_graph(
        adjacency,
        features,
        normalize_features=True,
        min_cells=3,
        drop_isolates=True,
    ):
        """
        Compute one smoothness score per ego neighborhood.

        For each cell i, define niche i as:
            {i} union 1-hop neighbors(i)
        """
        adjacency = adjacency.tocsr()
        n = adjacency.shape[0]
        rows = []

        for center in tqdm.tqdm(range(n)):
            neigh = adjacency[center].indices
            nodes = np.unique(np.concatenate(([center], neigh)))

            if len(nodes) < min_cells:
                continue

            A_sub = adjacency[nodes][:, nodes].tocsr()
            X_sub = features[nodes]

            A_sub, X_sub = prepare_subgraph(
                A_sub,
                X_sub,
                min_cells=min_cells,
                drop_isolates=drop_isolates,
            )

            if A_sub is None:
                continue

            res = signal_smoothness(
                adjacency=A_sub,
                features=X_sub,
                normalize_features=normalize_features
            )

            rows.append({
                "niche_id": center,
                "n_cells": A_sub.shape[0],
                "smoothness": res["mean"],
                "median_gene_smoothness": res["median"],
                "genewise_std": res["std"],
                "random_baseline": res["random_baseline"],
                "graph_nnz": A_sub.nnz,
            })

        niche_df = pd.DataFrame(rows)

        if niche_df.empty:
            summary = {
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "weighted_mean": np.nan,
                "n_niches_used": 0,
            }
        else:
            summary = {
                "mean": niche_df["smoothness"].mean(),
                "median": niche_df["smoothness"].median(),
                "std": niche_df["smoothness"].std(ddof=1) if len(niche_df) > 1 else 0.0,
                "weighted_mean": np.average(
                    niche_df["smoothness"],
                    weights=niche_df["n_cells"]
                ),
                "n_niches_used": len(niche_df),
            }

        return niche_df, summary

    def plot_dataset_comparison(results_df, adata, dataset_name, out_path=None, bins=50):
        """
        Two-panel figure:
          Left: KNN niche smoothness compared to watershed reference
          Right: Watershed niche size distribution (log-scaled y-axis)
        """
        plot_df = results_df.copy()

        watershed_row = plot_df[plot_df["method"] == "watershed"].iloc[0]
        knn_df = plot_df[plot_df["method"] != "watershed"].copy().sort_values("k")

        ws_mean = watershed_row["mean_smoothness"]
        ws_std = watershed_row["std_smoothness"]

        x = np.arange(len(knn_df))
        y = knn_df["mean_smoothness"].values
        yerr = knn_df["std_smoothness"].values
        labels = [f"k={int(k)}" for k in knn_df["k"].values]

        niche_sizes = adata.obs["watershed_by_descent"].value_counts()
        mean_size = niche_sizes.mean()
        median_size = niche_sizes.median()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left panel
        ax = axes[0]

        ax.axhspan(
            ws_mean - ws_std,
            ws_mean + ws_std,
            alpha=0.15,
            color="gray",
            label="Watershed ± 1 SD across niches"
        )

        ax.axhline(
            ws_mean,
            linestyle="--",
            linewidth=2,
            color="black",
            label=f"Watershed mean = {ws_mean:.3f}"
        )

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o",
            capsize=5,
            linewidth=2,
            color="tab:blue",
            ecolor="tab:blue",
            markersize=7,
            label="KNN mean ± 1 SD across niches"
        )

        ax.plot(x, y, color="tab:blue", alpha=0.8)

        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.003, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean niche smoothness")
        ax.set_xlabel("KNN neighborhood size")
        ax.set_title("Niche-level smoothness comparison")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)

        # Right panel
        ax2 = axes[1]

        ax2.hist(
            niche_sizes,
            bins=bins,
            alpha=0.75,
            color="tab:green",
            edgecolor="black",
            linewidth=0.5
        )

        ax2.axvline(
            mean_size,
            linestyle="--",
            linewidth=2,
            color="red",
            label=f"Mean = {mean_size:.2f}"
        )

        ax2.axvline(
            median_size,
            linestyle="-",
            linewidth=2,
            color="orange",
            label=f"Median = {median_size:.2f}"
        )

        ax2.set_yscale("log")
        ax2.set_xlabel("Watershed niche size (# cells)")
        ax2.set_ylabel("Number of niches (log scale)")
        ax2.set_title("Watershed niche sizes")
        ax2.grid(axis="y", alpha=0.25)
        ax2.legend(frameon=False)

        summary_txt = (
            f"Total niches: {len(niche_sizes)}\n"
            f"Min: {niche_sizes.min()}\n"
            f"Max: {niche_sizes.max()}"
        )
        ax2.text(
            0.98, 0.98, summary_txt,
            transform=ax2.transAxes,
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", alpha=0.1)
        )

        fig.suptitle(dataset_name, fontsize=13)
        plt.tight_layout()

        if out_path is not None:
            plt.savefig(out_path, dpi=300, bbox_inches="tight")

        if show_plots:
            plt.show()

        plt.close()

    # --------------------------------------------------
    # Find datasets
    # --------------------------------------------------
    zarr_paths = sorted(
        [
            os.path.join(input_dir, x)
            for x in os.listdir(input_dir)
            if x.endswith(".zarr")
        ]
    )

    if len(zarr_paths) == 0:
        raise FileNotFoundError(f"No .zarr datasets found in: {input_dir}")

    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)

    if save_tables:
        os.makedirs(table_dir, exist_ok=True)

    all_results = []

    # --------------------------------------------------
    # Process each dataset
    # --------------------------------------------------
    for zarr_path in zarr_paths:
        dataset_name = safe_name(zarr_path)
        print("\n" + "=" * 80)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 80)

        # Step 1: Load data
        sdata = sd.read_zarr(zarr_path)
        adata = sdata.tables["table"]

        print(sdata)
        print(adata)

        # Step 2: Predict watershed niches
        adata = predict_niches(sdata, method=watershed_method, plot=False)

        print("\nWatershed niche counts:")
        print(adata.obs["watershed_by_descent"].value_counts())

        # Step 3: Extract features and coordinates
        if sp.issparse(adata.X):
            features = adata.X.toarray()
        else:
            features = np.asarray(adata.X)

        coords = adata.obsm["spatial"]

        # Step 4: Watershed niche-level smoothness
        watershed_adj = adata.obsp["pruned_spatial_connectivities"].tocsr()
        watershed_labels = adata.obs["watershed_by_descent"].to_numpy()

        watershed_niche_df, watershed_summary = niche_smoothness_from_labels(
            adjacency=watershed_adj,
            features=features,
            niche_labels=watershed_labels,
            normalize_features=normalize_features,
            min_cells=min_cells_per_niche,
            drop_isolates=drop_isolates,
        )

        print("\nWatershed niche-level smoothness summary:")
        print(f"Mean across niches:   {watershed_summary['mean']:.6f}")
        print(f"Median across niches: {watershed_summary['median']:.6f}")
        print(f"Std across niches:    {watershed_summary['std']:.6f}")
        print(f"Weighted mean:        {watershed_summary['weighted_mean']:.6f}")
        print(f"N niches used:        {watershed_summary['n_niches_used']}")

        dataset_results = []
        dataset_results.append({
            "dataset": dataset_name,
            "method": "watershed",
            "k": np.nan,
            "mean_smoothness": watershed_summary["mean"],
            "median_smoothness": watershed_summary["median"],
            "std_smoothness": watershed_summary["std"],   # std across niches
            "weighted_mean_smoothness": watershed_summary["weighted_mean"],
            "random_baseline": np.nan,
            "n_cells": adata.n_obs,
            "n_features": adata.n_vars,
            "n_watershed_niches": adata.obs["watershed_by_descent"].nunique(),
            "n_niches_used": watershed_summary["n_niches_used"],
        })

        if save_tables and save_niche_level_tables:
            watershed_niche_csv = os.path.join(
                table_dir,
                f"{dataset_name}_watershed_niche_scores.csv"
            )
            watershed_niche_df.to_csv(watershed_niche_csv, index=False)

        # Step 5: KNN niche-level smoothness
        for k in ks:
            print(f"\nBuilding KNN graph for k={k} ...")
            A_knn = knn_graph(coords, k)

            print(f"Computing ego-KNN niche smoothness for k={k} ...")
            knn_niche_df, knn_summary = niche_smoothness_from_ego_graph(
                adjacency=A_knn,
                features=features,
                normalize_features=normalize_features,
                min_cells=min_cells_per_niche,
                drop_isolates=drop_isolates,
            )

            dataset_results.append({
                "dataset": dataset_name,
                "method": f"ego_knn_k{k}",
                "k": k,
                "mean_smoothness": knn_summary["mean"],
                "median_smoothness": knn_summary["median"],
                "std_smoothness": knn_summary["std"],   # std across niches
                "weighted_mean_smoothness": knn_summary["weighted_mean"],
                "random_baseline": np.nan,
                "n_cells": adata.n_obs,
                "n_features": adata.n_vars,
                "n_watershed_niches": adata.obs["watershed_by_descent"].nunique(),
                "n_niches_used": knn_summary["n_niches_used"],
            })

            print(f"k={k} mean niche smoothness: {knn_summary['mean']:.6f}")
            print(f"k={k} median niche smoothness: {knn_summary['median']:.6f}")
            print(f"k={k} std across niches: {knn_summary['std']:.6f}")
            print(f"k={k} weighted mean smoothness: {knn_summary['weighted_mean']:.6f}")
            print(f"k={k} niches used: {knn_summary['n_niches_used']}")

            if save_tables and save_niche_level_tables:
                knn_niche_csv = os.path.join(
                    table_dir,
                    f"{dataset_name}_ego_knn_k{k}_niche_scores.csv"
                )
                knn_niche_df.to_csv(knn_niche_csv, index=False)

        dataset_results_df = pd.DataFrame(dataset_results)

        print("\nSummary table:")
        print(dataset_results_df)

        # Save per-dataset table
        if save_tables:
            csv_path = os.path.join(table_dir, f"{dataset_name}_smoothness_summary.csv")
            dataset_results_df.to_csv(csv_path, index=False)

        # Save per-dataset plot
        if save_plots or show_plots:
            plot_path = None
            if save_plots:
                plot_path = os.path.join(
                    plot_dir,
                    f"{dataset_name}_watershed_vs_knn_niche_smoothness.png"
                )

            plot_dataset_comparison(
                dataset_results_df,
                adata,
                dataset_name,
                out_path=plot_path,
                bins=niche_size_bins
            )

        all_results.extend(dataset_results)

    # --------------------------------------------------
    # Combined output
    # --------------------------------------------------
    all_results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 80)
    print("Combined summary across all datasets")
    print("=" * 80)
    print(all_results_df)

    if save_tables:
        combined_csv = os.path.join(table_dir, "all_datasets_smoothness_summary.csv")
        all_results_df.to_csv(combined_csv, index=False)

        pivot_csv = os.path.join(table_dir, "all_datasets_smoothness_pivot.csv")
        pivot_df = all_results_df.pivot_table(
            index="dataset",
            columns="method",
            values="mean_smoothness"
        )
        pivot_df.to_csv(pivot_csv)

    return all_results_df

results = compare_smoothness_all_zarrs(
    input_dir="../single_cell/data/zarr_outputs_no_images/",
    ks=(5, 10, 20),
    watershed_method="delaunay",
    normalize_features=True,
    save_plots=True,
    show_plots=False,
    save_tables=True,
    save_niche_level_tables=True,
    min_cells_per_niche=5,
)