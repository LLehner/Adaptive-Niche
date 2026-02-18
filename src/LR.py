import numpy as np
import pandas as pd
import liana as li


def add_liana_adjacency_matrices(adata):

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
