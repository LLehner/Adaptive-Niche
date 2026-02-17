import os
import pandas as pd
import liana as li
import scanpy as sc

# ---------------------------------------------------------
# SETUP & HELPER FUNCTIONS
# ---------------------------------------------------------

def is_valid_entity(entity_name, available_genes):
    """
    Checks if a gene (or complex like 'ITGA4_ITGB1') is in the user's dataset.
    Returns True only if ALL subunits of the complex are present.
    """
    subunits = entity_name.split('_')
    return all(subunit in available_genes for subunit in subunits)

def get_genes_from_h5(folder_path):
    """
    Reads metadata from 10x H5 files to get the gene list.
    """
    candidates = [
        os.path.join(folder_path, "cell_feature_matrix.h5"),
        os.path.join(folder_path, "cell_feature_matrix", "cell_feature_matrix.h5")
    ]
    
    for h5_path in candidates:
        if os.path.exists(h5_path):
            try:
                # Read 10x H5 (only metadata, fast)
                adata = sc.read_10x_h5(h5_path)
                return set(adata.var_names)
            except Exception as e:
                print(f"Warning: Found H5 at {h5_path} but failed to read: {e}")
                
    return None

def convert_db_to_mouse_format(human_df):
    """
    Fallback: Converts a Human DB (All Caps) to Mouse format (Title Case).
    """
    print("  -> Generating Mouse DB from Human DB (Homology approximation)...")
    mouse_df = human_df.copy()
    
    # Helper to title case a single gene or complex string
    def to_murine(s):
        return "_".join([part.capitalize() for part in s.split("_")])

    mouse_df['ligand'] = mouse_df['ligand'].apply(to_murine)
    mouse_df['receptor'] = mouse_df['receptor'].apply(to_murine)
    return mouse_df

# ---------------------------------------------------------
# 1. LOAD LIANA DATABASES (Human & Mouse)
# ---------------------------------------------------------
print("Loading LIANA databases...")

# Load Human
lr_db_human = li.resource.select_resource('consensus')
print(f"Human DB loaded: {len(lr_db_human)} interactions (Format: {lr_db_human['ligand'].iloc[0]})")

# Load Mouse (Try 'mouseconsensus', else auto-convert Human DB)
try:
    lr_db_mouse = li.resource.select_resource('mouseconsensus')
    print(f"Mouse DB loaded: {len(lr_db_mouse)} interactions (Format: {lr_db_mouse['ligand'].iloc[0]})")
except Exception as e:
    print("Warning: 'mouseconsensus' resource not found in this LIANA version.")
    lr_db_mouse = convert_db_to_mouse_format(lr_db_human)
    print(f"Mouse DB Generated via conversion: {len(lr_db_mouse)} interactions.")

# ---------------------------------------------------------
# 2. PROCESS FOLDERS
# ---------------------------------------------------------
data_root = "../single_cell/data"
results_summary = []
dataset_pairs = {} 
dataset_species_map = {}

if os.path.exists(data_root):
    folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
else:
    print(f"Directory '{data_root}' not found.")
    folders = []

print(f"\nScanning {len(folders)} folders in '{data_root}'...\n")

for folder_name in folders:
    full_path = os.path.join(data_root, folder_name)
    
    # --- Step A: Get Genes ---
    user_genes = get_genes_from_h5(full_path)
    if user_genes is None or len(user_genes) == 0:
        print(f"Skipping {folder_name}: No gene panel found.")
        continue

    # --- Step B: Determine Species & Select DB ---
    # Heuristic: Folders starting with "M_" are Mouse
    if folder_name.startswith("M_"):
        species = "Mouse"
        current_db = lr_db_mouse
    else:
        species = "Human"
        current_db = lr_db_human

    # --- Step C: Filter Pairs ---
    # Find rows where Ligand AND Receptor are in the dataset
    valid_mask = current_db.apply(
        lambda row: is_valid_entity(row['ligand'], user_genes) and 
                    is_valid_entity(row['receptor'], user_genes), 
        axis=1
    )
    
    my_valid_pairs = current_db[valid_mask].copy()
    
    # --- Step D: Save Individual Dataset Results ---
    output_csv = os.path.join(full_path, "valid_lr_pairs.csv")
    my_valid_pairs.to_csv(output_csv, index=False)
    # Optional: Print less to keep console clean, or print confirmation:
    # print(f"  -> Saved {len(my_valid_pairs)} pairs to {output_csv}")

    # Store for overlap analysis
    pair_set = set(zip(my_valid_pairs['ligand'], my_valid_pairs['receptor']))
    
    dataset_pairs[folder_name] = pair_set
    dataset_species_map[folder_name] = species
    
    results_summary.append({
        'Dataset': folder_name,
        'Species': species,
        'Genes_Detected': len(user_genes),
        'Viable_LR_Pairs': len(pair_set),
        'Output_File': "valid_lr_pairs.csv"
    })

# ---------------------------------------------------------
# 3. DISPLAY SUMMARY
# ---------------------------------------------------------
if len(results_summary) > 0:
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values(by='Viable_LR_Pairs', ascending=False)
    
    print("-" * 80)
    print("SUMMARY OF VIABLE INTERACTIONS")
    print("-" * 80)
    print(results_df[['Dataset', 'Species', 'Viable_LR_Pairs']].to_string(index=False))
    print("-" * 80)
else:
    print("No valid datasets processed.")
    exit()

# ---------------------------------------------------------
# 4. OVERLAP ANALYSIS (Separated by Species)
# ---------------------------------------------------------
print("\n" + "="*80)
print("OVERLAPPING PAIRS ANALYSIS")
print("="*80)

def save_and_print_overlaps(datasets, species_name):
    print(f"\n--- {species_name} Datasets ({len(datasets)}) ---")
    if not datasets:
        print("No datasets found.")
        return

    # Start intersection with the first dataset
    common_pairs = dataset_pairs[datasets[0]].copy()
    
    for ds in datasets[1:]:
        common_pairs = common_pairs.intersection(dataset_pairs[ds])
        
    if common_pairs:
        # Create DataFrame from the set
        df_common = pd.DataFrame(list(common_pairs), columns=['Ligand', 'Receptor'])
        
        # Save to CSV
        output_filename = f"common_pairs_{species_name.lower()}.csv"
        df_common.to_csv(output_filename, index=False)
        
        print(f"Found {len(common_pairs)} pairs common to ALL {species_name} datasets.")
        print(f"Saved common pairs to '{output_filename}'")
        
        # Sort for display
        print(df_common.sort_values('Ligand').head(10).to_string(index=False))
        if len(df_common) > 10:
            print(f"... and {len(df_common) - 10} more.")
    else:
        print("No exact overlaps found across all datasets.")

human_datasets = [d for d, s in dataset_species_map.items() if s == 'Human']
mouse_datasets = [d for d, s in dataset_species_map.items() if s == 'Mouse']

save_and_print_overlaps(human_datasets, "Human")
save_and_print_overlaps(mouse_datasets, "Mouse")