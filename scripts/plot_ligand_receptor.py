import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def plot_ligand_receptor_abundance(data_folder, ligand, receptor, output_dir="plots"):
    """
    Generates a 2x2 figure:
      - Top Row: Spatial scatter plots for Ligand and Receptor.
      - Bottom Row: Histograms of non-zero expression counts for Ligand and Receptor.

    Args:
        data_folder (str): Path to the dataset folder.
        ligand (str): Gene symbol for the ligand.
        receptor (str): Gene symbol for the receptor.
        output_dir (str): Directory where the output plot will be saved.
    """
    
    dataset_name = os.path.basename(os.path.normpath(data_folder))
    print(f"\nProcessing: {dataset_name}...")

    # --- 1. Load Data ---
    matrix_file = os.path.join(data_folder, "cell_feature_matrix.h5")
    if not os.path.exists(matrix_file):
        print(f"Skipping {dataset_name}: 'cell_feature_matrix.h5' not found.")
        return

    try:
        adata = sc.read_10x_h5(matrix_file)
        adata.var_names_make_unique()
    except Exception as e:
        print(f"Skipping {dataset_name}: Error reading H5 file ({e}).")
        return

    # Handle coordinates
    cells_file_parquet = os.path.join(data_folder, "cells.parquet")
    cells_file_csv = os.path.join(data_folder, "cells.csv.gz")
    
    if os.path.exists(cells_file_parquet):
        df_locs = pd.read_parquet(cells_file_parquet)
    elif os.path.exists(cells_file_csv):
        df_locs = pd.read_csv(cells_file_csv)
    else:
        print(f"Skipping {dataset_name}: No coordinates file found.")
        return

    # Map indices
    if 'cell_id' in df_locs.columns:
        df_locs.set_index('cell_id', inplace=True)
    
    if not df_locs.index.empty and isinstance(df_locs.index[0], bytes):
        df_locs.index = df_locs.index.map(lambda x: x.decode('utf-8'))
    else:
        df_locs.index = df_locs.index.astype(str)

    # Intersect valid cells
    common = adata.obs_names.intersection(df_locs.index)
    if len(common) == 0:
        print(f"Skipping {dataset_name}: No common cell IDs.")
        return
        
    adata = adata[common]
    
    # Assign coordinates
    if 'x_centroid' in df_locs.columns:
        adata.obs[['x', 'y']] = df_locs.loc[common, ['x_centroid', 'y_centroid']]
    elif 'x' in df_locs.columns:
        adata.obs[['x', 'y']] = df_locs.loc[common, ['x', 'y']]
    else:
         print(f"Skipping {dataset_name}: Coordinate columns missing.")
         return

    # --- 2. Check Genes ---
    if ligand not in adata.var_names or receptor not in adata.var_names:
        print(f"Skipping {dataset_name}: Genes missing.")
        return

    # --- 3. Get Counts ---
    # Convert sparse matrix to dense array for plotting
    l_counts = adata[:, ligand].X.toarray().flatten()
    r_counts = adata[:, receptor].X.toarray().flatten()
    
    # --- 4. Plotting Setup ---
    # Create 2x2 Grid: Top for Maps, Bottom for Histograms
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='white', 
                             gridspec_kw={'height_ratios': [3, 1]})
    
    # Helper for Spatial Plots (Row 0)
    def plot_spatial(ax, counts, color_map, title):
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=10, fontweight='bold')

        # Background (all cells faint grey)
        ax.scatter(adata.obs['x'], adata.obs['y'], c='#f0f0f0', s=1, rasterized=True)
        
        # Foreground (expressing cells)
        mask = counts > 0
        if mask.sum() > 0:
            sc_plot = ax.scatter(
                adata.obs.loc[mask, 'x'], 
                adata.obs.loc[mask, 'y'], 
                c=counts[mask], 
                cmap=color_map, 
                s=1.5, 
                alpha=1.0, 
                rasterized=True
            )
            plt.colorbar(sc_plot, ax=ax, fraction=0.03, pad=0.01)

    # Helper for Histograms (Row 1)
    def plot_histogram(ax, counts, color, title):
        # Filter for non-zero counts to make the histogram meaningful
        expr_values = counts[counts > 0]
        
        if len(expr_values) > 0:
            sns.histplot(expr_values, ax=ax, color=color, bins=30, kde=False)
            ax.set_title(f"Distribution of Non-Zero Counts ({title})")
            ax.set_xlabel("Transcript Count")
            ax.set_ylabel("Frequency (Cells)")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No Expression Detected", ha='center', va='center')
            ax.axis('off')

    # --- 5. Generate Plots ---
    
    # Top Left: Ligand Map
    plot_spatial(axes[0, 0], l_counts, 'viridis', f"Ligand: {ligand}")
    
    # Top Right: Receptor Map
    plot_spatial(axes[0, 1], r_counts, 'plasma', f"Receptor: {receptor}")
    
    # Bottom Left: Ligand Histogram
    plot_histogram(axes[1, 0], l_counts, 'teal', ligand)
    
    # Bottom Right: Receptor Histogram
    plot_histogram(axes[1, 1], r_counts, 'purple', receptor)

    # Finalize Layout
    plt.suptitle(f"Spatial Abundance & Distribution: {dataset_name}", fontsize=20, y=0.96)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    out_filename = f"{dataset_name}_{ligand}_{receptor}_abundance.png"
    out_path = os.path.join(output_dir, out_filename)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Ligand-Receptor Spatial Abundance + Histograms")
    parser.add_argument("--root", type=str, default="data", help="Root folder containing dataset subfolders")
    parser.add_argument("--ligand", type=str, required=True, help="Ligand gene symbol (e.g., VCAN)")
    parser.add_argument("--receptor", type=str, required=True, help="Receptor gene symbol (e.g., EGFR)")
    parser.add_argument("--output", type=str, default="plots", help="Output folder for images")
    
    args = parser.parse_args()

    if os.path.exists(args.root):
        folders = [f for f in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, f))]
        
        if not folders:
            print(f"No subfolders found in {args.root}")
        
        for folder in folders:
            full_path = os.path.join(args.root, folder)
            plot_ligand_receptor_abundance(full_path, args.ligand, args.receptor, args.output)
    else:
        print(f"Root directory '{args.root}' does not exist.")