import pandas as pd
import liana as li

# ---------------------------------------------------------
# 1. LOAD YOUR DATA
# ---------------------------------------------------------
# Replace this with your actual file loading
df_meta = pd.read_csv("data/XeniumPrimeHuman5Kpan_tissue_pathways_metadata.csv")
user_genes = set(df_meta['gene_name'].unique())
print(f"Loaded {len(user_genes)} unique genes from your dataset.")

# ---------------------------------------------------------
# 2. LOAD LIANA DATABASE (CONSENSUS)
# ---------------------------------------------------------
# 'consensus' is the best default: it combines CellChat, CellPhoneDB, etc.
# This downloads the database automatically.
lr_db = li.resource.select_resource('consensus')

print(f"\nLIANA Consensus Database loaded: {len(lr_db)} total interactions.")
# Columns usually include: 'ligand', 'receptor', 'source', 'reference'

# ---------------------------------------------------------
# 3. FILTER: FIND PAIRS EXISTING IN YOUR DATASET
# ---------------------------------------------------------

def is_valid_entity(entity_name, available_genes):
    """
    Checks if a gene (or complex like 'ITGA4_ITGB1') is in the user's dataset.
    Returns True only if ALL subunits of the complex are present.
    """
    # LIANA usually separates complexes with underscores '_'
    subunits = entity_name.split('_')
    
    # Check if every single subunit is in your gene list
    return all(subunit in available_genes for subunit in subunits)

# Apply the filter
# We want rows where BOTH the ligand AND the receptor exist in your dataset
valid_mask = lr_db.apply(
    lambda row: is_valid_entity(row['ligand'], user_genes) and 
                is_valid_entity(row['receptor'], user_genes), 
    axis=1
)

my_valid_pairs = lr_db[valid_mask].copy()

# ---------------------------------------------------------
# 4. VIEW & SAVE RESULTS
# ---------------------------------------------------------
print(f"\nMatch Results:")
print(f"Out of {len(lr_db)} known pairs, your dataset supports: {len(my_valid_pairs)} pairs.")

if len(my_valid_pairs) > 0:
    print("\nTop 5 valid pairs found in your data:")
    print(my_valid_pairs[['ligand', 'receptor']].head())
    
    # Save this list - this is your "Ground Truth" for benchmarking
    my_valid_pairs.to_csv("human_xenium_5K_ground_truth.csv", index=False)
else:
    print("\nWARNING: No valid pairs found. Your gene panel might be too small or missing key interactors.")
