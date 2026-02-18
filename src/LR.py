import numpy as np
import pandas as pd
import liana as li
import scipy.sparse as sp


def get_LR_pairs(adata):

    genes = list(adata.var_names)
    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    # -----------------------------
    # load LIANA resources
    # -----------------------------
    human_db = li.resource.select_resource("consensus")

    try:
        mouse_db = li.resource.select_resource("mouseconsensus")
    except Exception:
        # fallback: approximate mouse symbols from human DB
        def to_murine(s):
            return "_".join(p.capitalize() for p in s.split("_"))

        mouse_db = human_db.copy()
        mouse_db["ligand"] = human_db["ligand"].apply(to_murine)
        mouse_db["receptor"] = human_db["receptor"].apply(to_murine)

    # -----------------------------
    # helpers
    # -----------------------------
    def valid_complex(entity):
        parts = entity.split("_")
        return all(p in gene_set for p in parts), parts

    def build_df(lr_db):

        mat = np.zeros((len(genes), len(genes)), dtype=np.uint8)
        n_interactions = 0

        for _, row in lr_db.iterrows():

            lig_ok, lig_parts = valid_complex(row["ligand"])
            rec_ok, rec_parts = valid_complex(row["receptor"])

            if not (lig_ok and rec_ok):
                continue

            n_interactions += 1

            # add edges for all subunit combinations
            for lg in lig_parts:
                i = gene_to_idx[lg]
                for rg in rec_parts:
                    j = gene_to_idx[rg]
                    mat[i, j] = 1

        df = pd.DataFrame(mat, index=genes, columns=genes)
        return df, n_interactions

    # -----------------------------
    # build matrices
    # -----------------------------
    human_df, n_human = build_df(human_db)
    mouse_df, n_mouse = build_df(mouse_db)

    # store in varm
    adata.varm["liana_adj_human"] = human_df
    adata.varm["liana_adj_mouse"] = mouse_df

    print(f"Detectable human interactions: {n_human}")
    print(f"Detectable mouse interactions: {n_mouse}")
    

def compute_lr_niche_scores(
    adata,
    lr_adj_key,
    niche_key,
    out_key="lr_niche_scores",
    global_out_key="lr_global_score",
    cell_frac_out_key="lr_cell_fraction_scores",
    cell_frac_global_out_key="lr_cell_fraction_global"
):
    """
    Compute two types of LR scores per niche and global:
    1. Sum of counts for LR genes divided by total counts per niche (existing)
    2. Fraction of cells in niche expressing at least one LR pair (new)
    
    adata.varm[lr_adj_key] must be a pandas DataFrame with gene names as rows/cols.
    adata.obs[niche_key] is a categorical or string column.
    """

    # --- LR adjacency ---
    lr_adj = adata.varm[lr_adj_key]
    if not isinstance(lr_adj, pd.DataFrame):
        raise ValueError("adata.varm[lr_adj_key] must be a pandas DataFrame")

    # --- genes participating in at least one LR interaction ---
    lr_genes = lr_adj.index[
        (lr_adj.values > 0).any(axis=1) |
        (lr_adj.values > 0).any(axis=0)
    ]
    lr_genes = lr_genes.intersection(adata.var_names)
    lr_mask = adata.var_names.isin(lr_genes)

    X = adata.X
    niches = adata.obs[niche_key]

    # --- per-niche score (sum of LR counts / total counts) ---
    lr_sum_scores = {}

    # --- per-niche fraction of cells expressing an LR pair ---
    cell_frac_scores = {}

    # iterate over niches
    if pd.api.types.is_categorical_dtype(niches):
        niche_iter = niches.cat.categories
    else:
        niche_iter = niches.unique()

    # convert sparse X to csr for efficient row access
    if sp.issparse(X):
        X = X.tocsr()

    for niche in niche_iter:
        cell_mask = (niches == niche).values
        X_niche = X[cell_mask, :]

        # --- LR gene sum / total counts ---
        if sp.issparse(X_niche):
            lr_sum = X_niche[:, lr_mask].sum()
            total_sum = X_niche.sum()
        else:
            lr_sum = X_niche[:, lr_mask].sum()
            total_sum = X_niche.sum()

        lr_sum_scores[niche] = float(lr_sum / total_sum) if total_sum > 0 else np.nan

        # --- cell fraction expressing at least one LR pair ---
        expressed_cells = 0
        num_cells = X_niche.shape[0]

        # find all LR pairs (ligand, receptor)
        ligand_indices, receptor_indices = np.where(lr_adj.values > 0)

        # check per cell if any LR pair is co-expressed
        for i in range(num_cells):
            if sp.issparse(X_niche):
                row_data = X_niche.getrow(i).toarray().flatten()
            else:
                row_data = X_niche[i, :]

            coexpressed = False
            for l_idx, r_idx in zip(ligand_indices, receptor_indices):
                # only consider if genes exist in adata.var_names
                l_gene = lr_adj.index[l_idx]
                r_gene = lr_adj.columns[r_idx]
                if l_gene in adata.var_names and r_gene in adata.var_names:
                    l_pos = adata.var_names.get_loc(l_gene)
                    r_pos = adata.var_names.get_loc(r_gene)
                    if row_data[l_pos] > 0 and row_data[r_pos] > 0:
                        coexpressed = True
                        break
            if coexpressed:
                expressed_cells += 1

        cell_frac_scores[niche] = float(expressed_cells / num_cells) if num_cells > 0 else np.nan

    # store results in uns
    adata.uns[out_key] = lr_sum_scores
    adata.uns[global_out_key] = float(np.nanmean(list(lr_sum_scores.values())))

    adata.uns[cell_frac_out_key] = cell_frac_scores
    adata.uns[cell_frac_global_out_key] = float(np.nanmean(list(cell_frac_scores.values())))



