import pandas as pd
import os

# ================================
# File paths
# ================================
influence_file = "influence_scores_merged_top_1percent_abs.csv"
original_file = "binary_imdb_data_with_ids.csv"
output_file = "binary_imdb_data_with_ids_poisoned.csv"

# Ensure files exist
if not os.path.exists(influence_file):
    raise FileNotFoundError(f"Influence file not found: {influence_file}")

if not os.path.exists(original_file):
    raise FileNotFoundError(f"Original dataset file not found: {original_file}")

# ================================
# Step 1: Load the top-k influential datapoints
# ================================
df_influence = pd.read_csv(influence_file)

# Check required columns
if "unique_id" not in df_influence.columns:
    raise ValueError("The influence DataFrame must contain a 'unique_id' column.")

# ================================
# Step 2: Extract unique_ids of top-k influential datapoints
# ================================
topk_unique_ids = df_influence["unique_id"].unique()
print(f"Number of unique training points to flip: {len(topk_unique_ids)}")

# ================================
# Step 3: Load the original dataset
# ================================
df_original = pd.read_csv(original_file)

# Check for columns
if "unique_id" not in df_original.columns:
    raise ValueError("The original dataset must contain a 'unique_id' column.")
if "sentiment" not in df_original.columns:
    raise ValueError("The original dataset must contain a 'sentiment' column.")

# ================================
# Step 4: Flip labels for the selected unique_ids
# ================================
# Create a boolean mask for rows with unique_ids in topk_unique_ids
mask = df_original["unique_id"].isin(topk_unique_ids)

# Flip the labels: if label=0 → 1, if label=1 → 0
df_original.loc[mask, "sentiment"] = df_original.loc[mask, "sentiment"].apply(lambda x: 1 - x)

# ================================
# Step 5: Save the poisoned dataset
# ================================
df_original.to_csv(output_file, index=False)
print(f"Poisoned dataset saved to {output_file}")
