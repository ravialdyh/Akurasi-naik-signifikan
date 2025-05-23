# Akurasi-naik-signifikan


import pandas as pd

# Assume the following DataFrames are already created and populated:
# refined_outliers_final_df: Output of refine_outliers_by_local_distinctness_two_sided_percentage
# uuid_analysis_stats_df: Output of identify_label_interval_outliers_two_tailed
#
# Also assume you know the percentile values used, e.g.:
# upper_percentile_val_for_naming = 0.95 # Used to construct column names like 'p95_value'
# lower_percentile_val_for_naming = 0.05 # Used to construct column names like 'p05_value'

def generate_true_distinct_outliers_file(
    refined_outliers_final_df,
    uuid_analysis_stats_df,
    upper_percentile_value_used, # e.g., 0.95
    lower_percentile_value_used, # e.g., 0.05
    output_filename="true_distinct_outliers.csv"
):
    """
    Filters the refined outliers to get only true distinct outliers and merges
    relevant UUID-level statistics for a comprehensive report.

    Args:
        refined_outliers_final_df (pd.DataFrame): DataFrame with distinctness flags.
        uuid_analysis_stats_df (pd.DataFrame): DataFrame with UUID-level statistics.
        upper_percentile_value_used (float): The upper percentile (e.g., 0.95).
        lower_percentile_value_used (float): The lower percentile (e.g., 0.05).
        output_filename (str): Name of the CSV file to save.

    Returns:
        pd.DataFrame: DataFrame containing the true distinct outliers with context.
    """
    if refined_outliers_final_df.empty:
        print("No refined outliers provided to process. Empty DataFrame will be returned.")
        return pd.DataFrame()

    # --- Step 1: Identify True Distinct Outliers ---
    # Condition for being a true distinct high outlier
    cond_true_distinct_high = (
        refined_outliers_final_df['outlier_type'].isin(['high', 'high_and_low_rare']) &
        (refined_outliers_final_df['is_locally_distinct_high'] == True)
    )
    # Condition for being a true distinct low outlier
    cond_true_distinct_low = (
        refined_outliers_final_df['outlier_type'].isin(['low', 'high_and_low_rare']) &
        (refined_outliers_final_df['is_locally_distinct_low'] == True)
    )

    true_distinct_outliers = refined_outliers_final_df[cond_true_distinct_high | cond_true_distinct_low].copy()

    if true_distinct_outliers.empty:
        print("No true distinct outliers found after applying distinctness criteria.")
        return pd.DataFrame()
        
    print(f"Identified {len(true_distinct_outliers)} true distinct outlier instances.")

    # --- Step 2: Prepare uuid_analysis_stats_df for merging ---
    # Select and rename columns from uuid_analysis_stats_df for clarity
    # Construct the expected percentile column names
    p_upper_col_name = f'p{int(upper_percentile_value_used*100)}_value'
    p_lower_col_name = f'p{int(lower_percentile_value_used*100)}_value'
    
    # Check if these columns exist and select them, along with other relevant stats
    cols_to_select_from_stats = ['uuid', 'model_driven_interval_count']
    rename_map_stats = {
        'model_driven_interval_count': 'uuid_total_model_driven_intervals'
    }

    if p_upper_col_name in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(p_upper_col_name)
        rename_map_stats[p_upper_col_name] = 'uuid_p_upper_threshold'
    else:
        print(f"Warning: Column '{p_upper_col_name}' not found in uuid_analysis_stats_df.")

    if p_lower_col_name in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(p_lower_col_name)
        rename_map_stats[p_lower_col_name] = 'uuid_p_lower_threshold'
    else:
        print(f"Warning: Column '{p_lower_col_name}' not found in uuid_analysis_stats_df.")
        
    # Also include overall proportion stats for the UUID
    actual_prop_ge_p_upper_col = 'actual_prop_ge_p_upper'
    actual_prop_le_p_lower_filtered_col = 'actual_prop_le_p_lower_filtered'

    if actual_prop_ge_p_upper_col in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(actual_prop_ge_p_upper_col)
        rename_map_stats[actual_prop_ge_p_upper_col] = 'uuid_overall_prop_ge_p_upper'
    if actual_prop_le_p_lower_filtered_col in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(actual_prop_le_p_lower_filtered_col)
        rename_map_stats[actual_prop_le_p_lower_filtered_col] = 'uuid_overall_prop_le_p_lower'

    # Ensure 'uuid' is present for merging
    if 'uuid' not in uuid_analysis_stats_df.columns:
        raise ValueError("'uuid' column is missing from uuid_analysis_stats_df, cannot merge.")

    uuid_context_df = uuid_analysis_stats_df[cols_to_select_from_stats].rename(columns=rename_map_stats)

    # --- Step 3: Merge true distinct outliers with UUID context ---
    # true_distinct_outliers already has 'uuid'
    # Ensure 'uuid' in true_distinct_outliers is of the same type as in uuid_context_df if issues arise
    final_outliers_report_df = pd.merge(true_distinct_outliers, uuid_context_df, on='uuid', how='left')

    # --- Step 4: Select and order columns for the final report ---
    # Start with key identifiers and outlier information
    ordered_cols = [
        'uuid', 'eventtime', 'label_interval', # Key original fields
        'outlier_type',                         # 'high', 'low', 'high_and_low_rare'
        'percentile_value_used',                # The specific Pxx value this outlier crossed for its type
        'actual_outlier_proportion_for_uuid',   # Rarity of this type of statistical outlier for this UUID
        'is_locally_distinct_low',
        'is_locally_distinct_high',
        'uuid_p_lower_threshold',               # UUID's general P-lower threshold
        'uuid_p_upper_threshold',               # UUID's general P-upper threshold
        'uuid_total_model_driven_intervals',
        'uuid_overall_prop_le_p_lower',         # Overall proportion of P-lower type events for this UUID
        'uuid_overall_prop_ge_p_upper'          # Overall proportion of P-upper type events for this UUID
    ]
    
    # Add remaining original columns from true_distinct_outliers, excluding those already listed or added by merge
    original_cols_to_add = [
        col for col in true_distinct_outliers.columns 
        if col not in ordered_cols and col not in uuid_context_df.columns # Avoid duplicating uuid
    ]
    
    final_column_order = ordered_cols + sorted(list(set(original_cols_to_add) - {'uuid'})) # Ensure uuid is not duplicated

    # Make sure all columns in final_column_order actually exist in final_outliers_report_df
    final_column_order = [col for col in final_column_order if col in final_outliers_report_df.columns]


    final_outliers_report_df = final_outliers_report_df[final_column_order]

    # --- Step 5: Save to CSV ---
    try:
        final_outliers_report_df.to_csv(output_filename, index=True) # Keep index if it's meaningful (e.g., from original log)
        print(f"Successfully saved {len(final_outliers_report_df)} true distinct outliers to {output_filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        
    return final_outliers_report_df

# --- Example Call (Ensure DataFrames are populated correctly first) ---
# Assume:
# refined_outliers_final_df is the output of refine_outliers_by_local_distinctness_two_sided_percentage
# uuid_analysis_stats_df is the output of identify_label_interval_outliers_two_tailed
# train_model_driven_df was used to generate both.
#
# upper_perc = 0.95 # The actual upper percentile used
# lower_perc = 0.05 # The actual lower percentile used
#
# if 'refined_outliers_final_df' in locals() and 'uuid_analysis_stats_df' in locals() \
#    and not refined_outliers_final_df.empty:
#
#     true_distinct_outliers_report = generate_true_distinct_outliers_file(
#         refined_outliers_final_df,
#         uuid_analysis_stats_df,
#         upper_percentile_value_used=upper_perc,
#         lower_percentile_value_used=lower_perc,
#         output_filename="final_true_distinct_outliers_report.csv"
#     )
#     print("\nHead of the final true distinct outliers report:")
#     print(true_distinct_outliers_report.head())
# else:
#     print("Could not generate report: refined_outliers_final_df or uuid_analysis_stats_df is missing or empty.")









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
