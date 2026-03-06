import numpy as np
import pandas as pd
import liana as li
import scipy.sparse as sp

def _valid_complex(entity, gene_set):
        parts = entity.split("_")
        return all(p in gene_set for p in parts), parts

def _build_df(lr_db, genes, gene_to_idx, gene_set):

        mat = np.zeros((len(genes), len(genes)), dtype=np.uint8)
        n_interactions = 0

        for _, row in lr_db.iterrows():

            lig_ok, lig_parts = _valid_complex(row["ligand"], gene_set)
            rec_ok, rec_parts = _valid_complex(row["receptor"], gene_set)

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
    
def build_lr_table(lr_db, gene_set):

    rows = []

    for _, row in lr_db.iterrows():

        lig = row["ligand"]
        rec = row["receptor"]

        lig_parts = lig.split("_")
        rec_parts = rec.split("_")

        if not all(p in gene_set for p in lig_parts):
            continue
        if not all(p in gene_set for p in rec_parts):
            continue

        rows.append({
            "ligand": lig,
            "receptor": rec,
            "pathway": row["pathway"] if "pathway" in row else None
        })
        
    return pd.DataFrame(rows)


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

    # build matrices
    human_df, n_human = _build_df(human_db, genes, gene_to_idx, gene_set)
    mouse_df, n_mouse = _build_df(mouse_db, genes, gene_to_idx, gene_set)

    # store in varm
    adata.varm["liana_adj_human"] = human_df
    adata.varm["liana_adj_mouse"] = mouse_df

    print(f"Detectable human interactions: {n_human}")
    print(f"Detectable mouse interactions: {n_mouse}")
    
    adata.uns["liana_lr_table_human"] = build_lr_table(human_db, gene_set)
    adata.uns["liana_lr_table_mouse"] = build_lr_table(mouse_db, gene_set)
    

def compute_lr_niche_scores(
    adata,
    lr_adj_key,
    niche_key,
    spatial_graph_key="pruned_spatial_connectivities",
    lr_table_key="liana_lr_table_mouse",
    n_permutations=5,
    eps=1e-9
):
    """
    Adds 4 niche-wise scores:

    1. permutation-normalized LR enrichment (z-score)
    2. specificity vs background
    3. pathway coherence
    4. spatial compactness

    Requires:
      - adata.obsp[spatial_graph_key] for score 4
      - adata.uns[lr_table_key] for score 3
    """

    lr_adj = adata.varm[lr_adj_key]
    if not isinstance(lr_adj, pd.DataFrame):
        raise ValueError("adata.varm[lr_adj_key] must be a pandas DataFrame")

    lr_genes = lr_adj.index[
        (lr_adj.values > 0).any(axis=1) |
        (lr_adj.values > 0).any(axis=0)
    ]
    lr_genes = lr_genes.intersection(adata.var_names)
    lr_mask = adata.var_names.isin(lr_genes)

    X = adata.X
    niches = adata.obs[niche_key]

    if pd.api.types.is_categorical_dtype(niches):
        niche_iter = list(niches.cat.categories)
    else:
        niche_iter = list(pd.unique(niches))

    if sp.issparse(X):
        X = X.tocsr()

    # precompute per-cell LR sums and total sums
    lr_gene_idx = np.where(lr_mask)[0]

    if sp.issparse(X):
        cell_lr_sum = np.asarray(X[:, lr_gene_idx].sum(axis=1)).ravel()
        cell_total_sum = np.asarray(X.sum(axis=1)).ravel()
    else:
        cell_lr_sum = X[:, lr_gene_idx].sum(axis=1)
        cell_total_sum = X.sum(axis=1)

    # ------------------------------
    # existing scores
    # ------------------------------
    lr_sum_scores = {}
    cell_frac_scores = {}

    # ------------------------------
    # new scores
    # ------------------------------
    perm_z_scores = {}
    spec_scores = {}
    coh_scores = {}
    spatial_scores = {}

    # precompute pair indices once (for cell-fraction score)
    ligand_indices, receptor_indices = np.where(lr_adj.values > 0)

    # background mask helper
    niche_array = np.asarray(niches)

    # pathway table
    lr_table = None
    if lr_table_key is not None and lr_table_key in adata.uns:
        lr_table = adata.uns[lr_table_key]

    # spatial graph
    G = None
    if spatial_graph_key is not None and spatial_graph_key in adata.obsp:
        G = adata.obsp[spatial_graph_key].tocsr()

    for niche in niche_iter:

        cell_mask = niche_array == niche
        idx = np.where(cell_mask)[0]

        # score 1
        lr_sum = cell_lr_sum[idx].sum()
        total_sum = cell_total_sum[idx].sum()
        lr_sum_scores[niche] = float(lr_sum / total_sum) if total_sum > 0 else np.nan

        # score 2
        X_niche = X[idx, :]
        expressed_cells = 0
        num_cells = X_niche.shape[0]

        for i in range(num_cells):
            if sp.issparse(X_niche):
                row_data = X_niche.getrow(i).toarray().ravel()
            else:
                row_data = X_niche[i, :]

            coexpressed = False
            for l_idx, r_idx in zip(ligand_indices, receptor_indices):
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

        cell_frac_scores[niche] = (
            float(expressed_cells / num_cells) if num_cells > 0 else np.nan
        )

        # permutation-normalized enrichment (z)
        obs = lr_sum_scores[niche]

        #null = []

        # if np.sum(cell_mask) > 0:

        #     for _ in range(n_permutations):
        #         perm = np.random.permutation(niche_array)
        #         pmask = perm == niche

        #         pidx = np.where(pmask)[0]

        #         if pidx.size == 0:
        #             continue

        #         p_lr = cell_lr_sum[pidx].sum()
        #         p_tot = cell_total_sum[pidx].sum()

        #         null.append(p_lr / p_tot if p_tot > 0 else 0.0)

        # if len(null) >= 2 and not np.isnan(obs):
        #     perm_z_scores[niche] = (obs - np.mean(null)) / (np.std(null) + eps)
        # else:
        #     perm_z_scores[niche] = np.nan

        # specificity vs rest
        rest_mask = ~cell_mask

        rest_lr = cell_lr_sum[rest_mask].sum()
        rest_tot = cell_total_sum[rest_mask].sum()

        rest_score = rest_lr / rest_tot if rest_tot > 0 else np.nan

        if not np.isnan(obs) and not np.isnan(rest_score):
            spec_scores[niche] = np.log((obs + eps) / (rest_score + eps))
        else:
            spec_scores[niche] = np.nan

        # pathway coherence
        if lr_table is None or "pathway" not in lr_table.columns:
            coh_scores[niche] = np.nan
        else:

            # very simple proxy:
            # fraction of LR interactions in this niche
            # that belong to the dominant pathway

            # reuse per-niche cell mask and compute expressed LR pairs
            pathway_counts = {}

            for _, row in lr_table.iterrows():
                pw = row["pathway"]
                if pd.isna(pw):
                    continue

                lig_parts = row["ligand"].split("_")
                rec_parts = row["receptor"].split("_")

                valid = (
                    all(g in adata.var_names for g in lig_parts) and
                    all(g in adata.var_names for g in rec_parts)
                )
                if not valid:
                    continue

                lig_idx = [adata.var_names.get_loc(g) for g in lig_parts]
                rec_idx = [adata.var_names.get_loc(g) for g in rec_parts]

                hit = False
                for ci in idx:
                    if sp.issparse(X):
                        rowx = X.getrow(ci)
                        if rowx[:, lig_idx].sum() > 0 and rowx[:, rec_idx].sum() > 0:
                            hit = True
                            break
                    else:
                        if X[ci, lig_idx].sum() > 0 and X[ci, rec_idx].sum() > 0:
                            hit = True
                            break

                if hit:
                    pathway_counts[pw] = pathway_counts.get(pw, 0) + 1

            if len(pathway_counts) == 0:
                coh_scores[niche] = np.nan
            else:
                total = sum(pathway_counts.values())
                coh_scores[niche] = max(pathway_counts.values()) / total

        # spatial compactness
        if G is None or idx.size == 0:
            print("why does this happen")
            spatial_scores[niche] = np.nan
        else:

            inside = 0
            boundary = 0

            for i in idx:
                start = G.indptr[i]
                end = G.indptr[i + 1]
                neigh = G.indices[start:end]

                for j in neigh:
                    if cell_mask[j]:
                        inside += 1
                    else:
                        boundary += 1

            denom = inside + boundary
            spatial_scores[niche] = inside / denom if denom > 0 else np.nan

    # store all results
    scores_df = pd.DataFrame({
        "lr_sum_score": pd.Series(lr_sum_scores),
        "lr_cell_fraction_score": pd.Series(cell_frac_scores),
        #"lr_perm_z_score": pd.Series(perm_z_scores),
        "lr_specificity_score": pd.Series(spec_scores),
        "lr_pathway_coherence_score": pd.Series(coh_scores),
        "lr_spatial_compactness_score": pd.Series(spatial_scores),
    })

    # make sure niches are a column, not only index
    scores_df.index.name = "niche"
    scores_df = scores_df.reset_index()

    adata.uns[f"lr_niche_score_table_{niche_key}"] = scores_df

    global_scores = {
        "lr_sum_score": float(np.nanmean(list(lr_sum_scores.values()))),
        "lr_cell_fraction_score": float(np.nanmean(list(cell_frac_scores.values()))),
        #"lr_perm_z_score": float(np.nanmean(list(perm_z_scores.values()))),
        "lr_specificity_score": float(np.nanmean(list(spec_scores.values()))),
        "lr_pathway_coherence_score": float(np.nanmean(list(coh_scores.values()))),
        "lr_spatial_compactness_score": float(np.nanmean(list(spatial_scores.values()))),
    }

    adata.uns[f"lr_global_scores_{niche_key}"] = global_scores




