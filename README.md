# Akurasi-naik-signifikan

import pandas as pd
import numpy as np

def refine_outliers_by_local_distinctness_adaptive(
    all_suspected_outliers_df,
    train_model_driven_df, 
    uuid_median_intervals, # Series or Dict: uuid -> median_label_interval
    # Parameters for adaptive proximity delta
    proximity_delta_low_ratio=0.20,  # e.g., 20% of UUID's median interval
    proximity_delta_high_ratio=0.10, # e.g., 10% of UUID's median interval
    min_absolute_proximity_delta=2.0, # Min window size in minutes
    max_absolute_proximity_delta=30.0, # Max window size in minutes
    # Parameters for density override (percentage based)
    min_density_percentage_low_override=0.05, 
    min_density_percentage_high_override=0.05,
    min_total_normal_intervals_for_percentage_check=10
):
    """
    Refines 'low' and 'high' outliers by checking if they are too close to a
    dense cluster of normal interval values for the same UUID, using adaptive proximity deltas.

    Args:
        all_suspected_outliers_df (pd.DataFrame): From identify_label_interval_outliers_two_tailed.
        train_model_driven_df (pd.DataFrame): All model-driven intervals for training.
        uuid_median_intervals (pd.Series or dict): Maps UUID to its median label_interval.
                                                   Used for calculating adaptive proximity_delta.
        proximity_delta_low_ratio (float): Ratio of median to set proximity window above low outlier.
        proximity_delta_high_ratio (float): Ratio of median to set proximity window below high outlier.
        min_absolute_proximity_delta (float): Minimum size of the proximity window (minutes).
        max_absolute_proximity_delta (float): Maximum size of the proximity window (minutes).
        min_density_percentage_low_override (float): Min proportion of UUID's normal intervals
                                                     in the low window to override.
        min_density_percentage_high_override (float): Min proportion of UUID's normal intervals
                                                      in the high window to override.
        min_total_normal_intervals_for_percentage_check (int): Min normal intervals for % check.


    Returns:
        pd.DataFrame: Input all_suspected_outliers_df with 'is_locally_distinct_low' and
                      'is_locally_distinct_high' columns, and effective deltas used.
    """
    if all_suspected_outliers_df.empty:
        print("No suspected outliers to refine.")
        # Ensure columns exist even if empty
        for col in ['is_locally_distinct_low', 'is_locally_distinct_high', 
                    'effective_proximity_delta_low', 'effective_proximity_delta_high']:
            if col not in all_suspected_outliers_df.columns:
                 all_suspected_outliers_df[col] = pd.NA
        return all_suspected_outliers_df

    required_cols_outliers = {'uuid', 'label_interval', 'outlier_type'}
    if not required_cols_outliers.issubset(all_suspected_outliers_df.columns):
        missing = required_cols_outliers - set(all_suspected_outliers_df.columns)
        raise ValueError(f"all_suspected_outliers_df is missing required columns: {missing}")

    required_cols_train = {'uuid', 'label_interval'} # train_model_driven_df should have index
    if not required_cols_train.issubset(train_model_driven_df.columns):
        missing_train = required_cols_train - set(train_model_driven_df.columns)
        raise ValueError(f"train_model_driven_df is missing required columns: {missing_train}")
    if not isinstance(uuid_median_intervals, (pd.Series, dict)):
        raise ValueError("uuid_median_intervals must be a Pandas Series or dictionary.")


    refined_outliers_df = all_suspected_outliers_df.copy()
    refined_outliers_df['is_locally_distinct_low'] = pd.NA
    refined_outliers_df['is_locally_distinct_high'] = pd.NA
    refined_outliers_df['effective_proximity_delta_low'] = pd.NA
    refined_outliers_df['effective_proximity_delta_high'] = pd.NA


    all_low_type_flagged_indices = set(
        refined_outliers_df[refined_outliers_df['outlier_type'].isin(['low', 'high_and_low_rare'])].index
    )
    all_high_type_flagged_indices = set(
        refined_outliers_df[refined_outliers_df['outlier_type'].isin(['high', 'high_and_low_rare'])].index
    )

    for index, outlier_row in refined_outliers_df.iterrows():
        current_uuid = outlier_row['uuid']
        current_outlier_value = outlier_row['label_interval']
        current_outlier_type = outlier_row['outlier_type']

        uuid_all_intervals_df = train_model_driven_df[
            train_model_driven_df['uuid'] == current_uuid
        ]
        if uuid_all_intervals_df.empty:
            continue

        current_uuid_median = None
        if isinstance(uuid_median_intervals, pd.Series):
            current_uuid_median = uuid_median_intervals.get(current_uuid)
        elif isinstance(uuid_median_intervals, dict):
            current_uuid_median = uuid_median_intervals.get(current_uuid)

        if current_uuid_median is None or pd.isna(current_uuid_median):
            # Fallback to a default fixed delta if median is not available for this UUID
            # Or, mark distinctness as True by default if median is missing
            # For now, let's use a mid-range default fixed delta as fallback
            effective_delta_low = (min_absolute_proximity_delta + max_absolute_proximity_delta) / 2 
            effective_delta_high = (min_absolute_proximity_delta + max_absolute_proximity_delta) / 2
            if current_outlier_type in ['low', 'high_and_low_rare']:
                 refined_outliers_df.loc[index, 'is_locally_distinct_low'] = True # Default to distinct if no median
            if current_outlier_type in ['high', 'high_and_low_rare']:
                 refined_outliers_df.loc[index, 'is_locally_distinct_high'] = True # Default to distinct
            refined_outliers_df.loc[index, 'effective_proximity_delta_low'] = effective_delta_low
            refined_outliers_df.loc[index, 'effective_proximity_delta_high'] = effective_delta_high
            # print(f"Warning: Median not found for UUID {current_uuid}. Using default deltas for distinctness check, defaulting to distinct.")
            # continue # Or proceed with default deltas if that's preferred over auto-True

        else: # Calculate adaptive deltas
            delta_l = current_uuid_median * proximity_delta_low_ratio
            effective_delta_low = np.clip(delta_l, min_absolute_proximity_delta, max_absolute_proximity_delta)
            
            delta_h = current_uuid_median * proximity_delta_high_ratio
            effective_delta_high = np.clip(delta_h, min_absolute_proximity_delta, max_absolute_proximity_delta)
            
            refined_outliers_df.loc[index, 'effective_proximity_delta_low'] = effective_delta_low
            refined_outliers_df.loc[index, 'effective_proximity_delta_high'] = effective_delta_high


        # --- Low Outlier Distinctness Check ---
        if current_outlier_type in ['low', 'high_and_low_rare']:
            # Use effective_delta_low calculated above if median was available, or the default if not.
            # The `else` block for missing median already set distinctness to True and default deltas.
            # So this part only runs if median was available for adaptive delta calculation.
            if current_uuid_median is not None and not pd.isna(current_uuid_median):
                uuid_normal_for_low_check_df = uuid_all_intervals_df[
                    ~uuid_all_intervals_df.index.isin(all_low_type_flagged_indices - {index})
                ]
                total_normal_intervals_for_low_check = len(uuid_normal_for_low_check_df)

                if total_normal_intervals_for_low_check < min_total_normal_intervals_for_percentage_check:
                    refined_outliers_df.loc[index, 'is_locally_distinct_low'] = True
                else:
                    window_start_low = current_outlier_value
                    window_end_low = current_outlier_value + effective_delta_low # Use adaptive delta
                    
                    nearby_normal_points_low_df = uuid_normal_for_low_check_df[
                        (uuid_normal_for_low_check_df['label_interval'] > window_start_low) &
                        (uuid_normal_for_low_check_df['label_interval'] <= window_end_low)
                    ]
                    count_nearby_normal_low = len(nearby_normal_points_low_df)

                    if total_normal_intervals_for_low_check > 0:
                        proportion_nearby_normal_low = count_nearby_normal_low / total_normal_intervals_for_low_check
                        if proportion_nearby_normal_low >= min_density_percentage_low_override:
                            refined_outliers_df.loc[index, 'is_locally_distinct_low'] = False
                        else:
                            refined_outliers_df.loc[index, 'is_locally_distinct_low'] = True
                    else:
                        refined_outliers_df.loc[index, 'is_locally_distinct_low'] = True

        # --- High Outlier Distinctness Check ---
        if current_outlier_type in ['high', 'high_and_low_rare']:
            if current_uuid_median is not None and not pd.isna(current_uuid_median):
                uuid_normal_for_high_check_df = uuid_all_intervals_df[
                    ~uuid_all_intervals_df.index.isin(all_high_type_flagged_indices - {index})
                ]
                total_normal_intervals_for_high_check = len(uuid_normal_for_high_check_df)

                if total_normal_intervals_for_high_check < min_total_normal_intervals_for_percentage_check:
                    refined_outliers_df.loc[index, 'is_locally_distinct_high'] = True
                else:
                    window_start_high = current_outlier_value - effective_delta_high # Use adaptive delta
                    window_end_high = current_outlier_value
                    
                    nearby_normal_points_high_df = uuid_normal_for_high_check_df[
                        (uuid_normal_for_high_check_df['label_interval'] >= window_start_high) &
                        (uuid_normal_for_high_check_df['label_interval'] < window_end_high)
                    ]
                    count_nearby_normal_high = len(nearby_normal_points_high_df)

                    if total_normal_intervals_for_high_check > 0:
                        proportion_nearby_normal_high = count_nearby_normal_high / total_normal_intervals_for_high_check
                        if proportion_nearby_normal_high >= min_density_percentage_high_override:
                            refined_outliers_df.loc[index, 'is_locally_distinct_high'] = False
                        else:
                            refined_outliers_df.loc[index, 'is_locally_distinct_high'] = True
                    else:
                        refined_outliers_df.loc[index, 'is_locally_distinct_high'] = True
                
    return refined_outliers_df

# --- How to prepare uuid_median_intervals (Example) ---
# This should be done once before calling the refinement function.
# Needs train_model_driven_df (same one passed to the refinement function).
# min_samples_for_median_calc_param = 10 # Consistent with visualization's Part 2

# if 'train_model_driven_df' in locals() and not train_model_driven_df.empty:
#     uuid_median_stats = train_model_driven_df.groupby('uuid')['label_interval'].agg(
#         median_label_interval='median',
#         count='count'
#     ).reset_index()
#     # Filter for UUIDs that have enough data points to have a somewhat reliable median
#     uuid_median_stats_filtered = uuid_median_stats[uuid_median_stats['count'] >= min_samples_for_median_calc_param]
#     # Create the Series: uuid -> median_label_interval
#     uuid_median_intervals_map = pd.Series(
#         uuid_median_stats_filtered.median_label_interval.values,
#         index=uuid_median_stats_filtered.uuid
#     )
# else:
#     uuid_median_intervals_map = pd.Series(dtype=float) # Empty series if no data


# --- Example Usage (after running identify_label_interval_outliers_two_tailed) ---
# if 'all_suspected_outliers_df' in locals() and 'train_model_driven_df' in locals() \
#    and 'uuid_median_intervals_map' in locals() \
#    and not all_suspected_outliers_df.empty and not train_model_driven_df.empty:
#
#     refined_outliers_adaptive_distinctness = refine_outliers_by_local_distinctness_adaptive(
#         all_suspected_outliers_df,
#         train_model_driven_df, 
#         uuid_median_intervals_map,
#         proximity_delta_low_ratio=0.20, # Check window = 20% of UUID's median interval size, above low outlier
#         proximity_delta_high_ratio=0.10,# Check window = 10% of UUID's median interval size, below high outlier
#         min_absolute_proximity_delta=2.0, # But window is at least 2 minutes
#         max_absolute_proximity_delta=30.0, # And at most 30 minutes
#         min_density_percentage_low_override=0.05, 
#         min_density_percentage_high_override=0.05,
#         min_total_normal_intervals_for_percentage_check=10
#     )
#
#     print("\nRefined outliers with adaptive two-sided local distinctness check:")
#     display_cols = ['uuid', 'label_interval', 'outlier_type', 
#                     'is_locally_distinct_low', 'effective_proximity_delta_low',
#                     'is_locally_distinct_high', 'effective_proximity_delta_high']
#     
#     # Filter to show rows where at least one distinctness check was made and value set
#     checked_rows_adaptive = refined_outliers_adaptive_distinctness[
#         refined_outliers_adaptive_distinctness['is_locally_distinct_low'].notna() |
#         refined_outliers_adaptive_distinctness['is_locally_distinct_high'].notna()
#     ]
#     print(checked_rows_adaptive[display_cols].head())
#
#     if 'is_locally_distinct_low' in refined_outliers_adaptive_distinctness.columns:
#         overridden_low_adaptive = refined_outliers_adaptive_distinctness[
#             (refined_outliers_adaptive_distinctness['outlier_type'].isin(['low', 'high_and_low_rare'])) &
#             (refined_outliers_adaptive_distinctness['is_locally_distinct_low'] == False)
#         ]
#         print(f"\nNumber of 'low' type outliers considered NOT locally distinct (adaptive): {len(overridden_low_adaptive)}")
#
#     if 'is_locally_distinct_high' in refined_outliers_adaptive_distinctness.columns:
#         overridden_high_adaptive = refined_outliers_adaptive_distinctness[
#             (refined_outliers_adaptive_distinctness['outlier_type'].isin(['high', 'high_and_low_rare'])) &
#             (refined_outliers_adaptive_distinctness['is_locally_distinct_high'] == False)
#         ]
#         print(f"Number of 'high' type outliers considered NOT locally distinct (adaptive): {len(overridden_high_adaptive)}")
#
# else:
#     print("Required DataFrames (all_suspected_outliers_df, train_model_driven_df, uuid_median_intervals_map) not found or empty for refinement.")





# Previous imports: import pandas as pd, import numpy as np

def generate_true_distinct_outliers_file_updated(
    refined_outliers_adaptive_df, # Output of refine_outliers_by_local_distinctness_adaptive
    uuid_analysis_stats_df,
    upper_percentile_value_used, 
    lower_percentile_value_used, 
    output_filename="final_true_distinct_outliers_report_adaptive.csv"
):
    """
    Filters refined outliers for true distinct ones and merges UUID-level statistics,
    including effective proximity deltas used.
    """
    if refined_outliers_adaptive_df.empty:
        print("No refined outliers provided to process. Empty DataFrame will be returned.")
        return pd.DataFrame()

    # --- Step 1: Identify True Distinct Outliers (logic remains the same) ---
    cond_true_distinct_high = (
        refined_outliers_adaptive_df['outlier_type'].isin(['high', 'high_and_low_rare']) &
        (refined_outliers_adaptive_df['is_locally_distinct_high'] == True)
    )
    cond_true_distinct_low = (
        refined_outliers_adaptive_df['outlier_type'].isin(['low', 'high_and_low_rare']) &
        (refined_outliers_adaptive_df['is_locally_distinct_low'] == True)
    )
    true_distinct_outliers = refined_outliers_adaptive_df[cond_true_distinct_high | cond_true_distinct_low].copy()

    if true_distinct_outliers.empty:
        print("No true distinct outliers found after applying distinctness criteria.")
        return pd.DataFrame()
        
    print(f"Identified {len(true_distinct_outliers)} true distinct outlier instances for the report.")

    # --- Step 2: Prepare uuid_analysis_stats_df (logic remains the same) ---
    p_upper_col_name = f'p{int(upper_percentile_value_used*100)}_value'
    p_lower_col_name = f'p{int(lower_percentile_value_used*100)}_value'
    cols_to_select_from_stats = ['uuid', 'model_driven_interval_count']
    rename_map_stats = {'model_driven_interval_count': 'uuid_total_model_driven_intervals'}
    if p_upper_col_name in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(p_upper_col_name)
        rename_map_stats[p_upper_col_name] = 'uuid_p_upper_threshold'
    if p_lower_col_name in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(p_lower_col_name)
        rename_map_stats[p_lower_col_name] = 'uuid_p_lower_threshold'
    # ... (add other stats columns like overall proportions as before) ...
    actual_prop_ge_p_upper_col = 'actual_prop_ge_p_upper'
    actual_prop_le_p_lower_filtered_col = 'actual_prop_le_p_lower_filtered'
    if actual_prop_ge_p_upper_col in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(actual_prop_ge_p_upper_col)
        rename_map_stats[actual_prop_ge_p_upper_col] = 'uuid_overall_prop_ge_p_upper'
    if actual_prop_le_p_lower_filtered_col in uuid_analysis_stats_df.columns:
        cols_to_select_from_stats.append(actual_prop_le_p_lower_filtered_col)
        rename_map_stats[actual_prop_le_p_lower_filtered_col] = 'uuid_overall_prop_le_p_lower'

    if 'uuid' not in uuid_analysis_stats_df.columns:
        raise ValueError("'uuid' column is missing from uuid_analysis_stats_df.")
    uuid_context_df = uuid_analysis_stats_df[cols_to_select_from_stats].rename(columns=rename_map_stats)

    # --- Step 3: Merge (logic remains the same) ---
    final_outliers_report_df = pd.merge(true_distinct_outliers, uuid_context_df, on='uuid', how='left')

    # --- Step 4: Select and order columns (ADD effective_proximity_deltas) ---
    ordered_cols = [
        'uuid', 'eventtime', 'label_interval', 
        'outlier_type',                         
        'percentile_value_used',                
        'actual_outlier_proportion_for_uuid',   
        'is_locally_distinct_low',
        'is_locally_distinct_high',
        'effective_proximity_delta_low',   # New
        'effective_proximity_delta_high',  # New
        'uuid_p_lower_threshold',               
        'uuid_p_upper_threshold',               
        'uuid_total_model_driven_intervals',
        'uuid_overall_prop_le_p_lower',        
        'uuid_overall_prop_ge_p_upper'          
    ]
    
    original_cols_to_add = [
        col for col in true_distinct_outliers.columns 
        if col not in ordered_cols and col not in uuid_context_df.columns 
    ]
    final_column_order = ordered_cols + sorted(list(set(original_cols_to_add) - {'uuid'}))
    final_column_order = [col for col in final_column_order if col in final_outliers_report_df.columns]
    final_outliers_report_df = final_outliers_report_df[final_column_order]

    # --- Step 5: Save to CSV (logic remains the same) ---
    try:
        final_outliers_report_df.to_csv(output_filename, index=True) 
        print(f"Successfully saved {len(final_outliers_report_df)} true distinct outliers to {output_filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")
        
    return final_outliers_report_df

# --- Example Call ---
# Make sure train_model_driven_df and uuid_median_intervals_map are correctly prepared
# and then refine_outliers_by_local_distinctness_adaptive is called.
#
# if 'refined_outliers_adaptive_df' in locals() and 'uuid_analysis_stats_df' in locals():
#     upper_perc_val = 0.95 # Actual value used in identify_...
#     lower_perc_val = 0.05 # Actual value used in identify_...
#
#     true_distinct_outliers_report_updated = generate_true_distinct_outliers_file_updated(
#         refined_outliers_adaptive_df, # This is the output of the adaptive refinement
#         uuid_analysis_stats_df,
#         upper_percentile_value_used=upper_perc_val,
#         lower_percentile_value_used=lower_perc_val,
#         output_filename="final_true_distinct_outliers_report_adaptive.csv"
#     )
#     print("\nHead of the updated final true distinct outliers report:")
#     print(true_distinct_outliers_report_updated.head())





















































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
