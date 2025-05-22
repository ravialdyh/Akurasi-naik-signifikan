# Akurasi-naik-signifikan

import pandas as pd

# Assume 'train_df' is your existing DataFrame (training data, aitempchanged == 0)
# Assume 'refined_outliers_final_df' is the output from your
# refine_outliers_by_local_distinctness_two_sided_percentage function

# Step 1: Identify the indices of the true distinct outliers to eliminate
# These are outliers that are flagged as either high or low type AND are locally distinct.

# Initialize an empty list to store indices of outliers to remove
indices_to_remove = []

if not refined_outliers_final_df.empty:
    # Identify true distinct high outliers
    true_distinct_high_outliers = refined_outliers_final_df[
        (refined_outliers_final_df['outlier_type'].isin(['high', 'high_and_low_rare'])) &
        (refined_outliers_final_df['is_locally_distinct_high'] == True)
    ]
    indices_to_remove.extend(true_distinct_high_outliers.index.tolist())

    # Identify true distinct low outliers
    true_distinct_low_outliers = refined_outliers_final_df[
        (refined_outliers_final_df['outlier_type'].isin(['low', 'high_and_low_rare'])) &
        (refined_outliers_final_df['is_locally_distinct_low'] == True)
    ]
    indices_to_remove.extend(true_distinct_low_outliers.index.tolist())

    # Ensure unique indices, as 'high_and_low_rare' could be in both lists above
    indices_to_remove = sorted(list(set(indices_to_remove)))

    print(f"Identified {len(indices_to_remove)} true distinct outlier instances to remove.")
else:
    print("No refined outliers provided, so no rows will be removed from train_df.")

# Step 2: Eliminate these outliers from train_df
if indices_to_remove:
    # Ensure train_df's index is what we expect (it should match the indices in refined_outliers_final_df)
    if not train_df.index.equals(refined_outliers_final_df.loc[indices_to_remove].index.drop_duplicates()):
        # This check might be too strict if refined_outliers_final_df is a subset.
        # The important part is that indices_to_remove are valid indices in train_df.
        pass # Proceeding with caution

    print(f"Original train_df shape: {train_df.shape}")
    train_df_cleaned = train_df.drop(index=indices_to_remove, errors='ignore')
    # errors='ignore' will not raise an error if some indices are not found, though they should be.
    # Use errors='raise' if you want to be certain all identified outlier indices exist in train_df.

    print(f"Cleaned train_df shape: {train_df_cleaned.shape}")
    print(f"Number of rows removed: {train_df.shape[0] - train_df_cleaned.shape[0]}")
else:
    print("No outliers identified to remove. train_df remains unchanged.")
    train_df_cleaned = train_df.copy() # Keep a consistent variable name

# Now 'train_df_cleaned' is your training DataFrame with the true distinct outliers removed.
# You can proceed with using train_df_cleaned for model training.
