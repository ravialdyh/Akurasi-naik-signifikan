# Akurasi-naik-signifikan

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# UUID SIMILARITY INVESTIGATION FUNCTIONS
# =============================================================================

def extract_uuid_behavioral_features(df):
    """
    Extract behavioral features for each UUID to detect potential duplicates.
    
    Args:
        df: DataFrame with columns including uuid, currentt, desiredt, eventtime, etc.
    
    Returns:
        DataFrame with behavioral features per UUID
    """
    print("Extracting behavioral features for UUID similarity analysis...")
    
    # Ensure eventtime is datetime
    df['eventtime'] = pd.to_datetime(df['eventtime'])
    df['hour'] = df['eventtime'].dt.hour
    df['dayofweek'] = df['eventtime'].dt.dayofweek
    
    # Group by UUID and calculate behavioral features
    uuid_features = []
    
    for uuid in df['uuid'].unique():
        uuid_data = df[df['uuid'] == uuid].copy()
        
        if len(uuid_data) < 10:  # Skip UUIDs with very little data
            continue
            
        features = {
            'uuid': uuid,
            'total_records': len(uuid_data),
            'unique_days': uuid_data['eventtime'].dt.date.nunique(),
            'date_range_days': (uuid_data['eventtime'].max() - uuid_data['eventtime'].min()).days + 1,
            
            # Temperature preferences
            'avg_desired_temp': uuid_data['desiredt'].mean(),
            'std_desired_temp': uuid_data['desiredt'].std(),
            'min_desired_temp': uuid_data['desiredt'].min(),
            'max_desired_temp': uuid_data['desiredt'].max(),
            'median_desired_temp': uuid_data['desiredt'].median(),
            
            # Current temperature patterns
            'avg_current_temp': uuid_data['currentt'].mean(),
            'std_current_temp': uuid_data['currentt'].std(),
            'temp_variance': uuid_data['currentt'].var(),
            
            # Temperature difference patterns
            'avg_temp_diff': (uuid_data['currentt'] - uuid_data['desiredt']).mean(),
            'std_temp_diff': (uuid_data['currentt'] - uuid_data['desiredt']).std(),
            'temp_diff_range': (uuid_data['currentt'] - uuid_data['desiredt']).max() - (uuid_data['currentt'] - uuid_data['desiredt']).min(),
            
            # Usage patterns
            'most_common_hour': uuid_data['hour'].mode()[0] if not uuid_data['hour'].mode().empty else -1,
            'usage_hour_spread': uuid_data['hour'].std(),
            'weekend_usage_ratio': len(uuid_data[uuid_data['dayofweek'].isin([5,6])]) / len(uuid_data),
            
            # AI intervention patterns
            'ai_temp_changed_ratio': uuid_data['aitempchanged'].mean() if 'aitempchanged' in uuid_data.columns else 0,
            
            # Environmental conditions (if available)
            'avg_indoor_humidity': uuid_data['indoor_humidity'].mean() if 'indoor_humidity' in uuid_data.columns else None,
            'avg_outdoor_humidity': uuid_data['outdoor_humidity'].mean() if 'outdoor_humidity' in uuid_data.columns else None,
        }
        
        # Mode usage patterns (if available)
        if 'mode' in uuid_data.columns:
            mode_counts = uuid_data['mode'].value_counts(normalize=True)
            features['primary_mode'] = mode_counts.index[0] if len(mode_counts) > 0 else 'unknown'
            features['mode_diversity'] = len(mode_counts)
            features['cool_mode_ratio'] = mode_counts.get('cool', 0)
            features['auto_mode_ratio'] = mode_counts.get('auto', 0)
        
        # Temporal patterns
        features['first_record'] = uuid_data['eventtime'].min()
        features['last_record'] = uuid_data['eventtime'].max()
        features['records_per_day'] = features['total_records'] / max(features['unique_days'], 1)
        
        uuid_features.append(features)
    
    features_df = pd.DataFrame(uuid_features)
    print(f"Extracted features for {len(features_df)} UUIDs")
    
    return features_df

def find_similar_uuids(features_df, similarity_threshold=0.95, min_similarity_features=5):
    """
    Find potentially duplicate UUIDs based on behavioral similarity.
    
    Args:
        features_df: DataFrame from extract_uuid_behavioral_features()
        similarity_threshold: Threshold for considering UUIDs similar
        min_similarity_features: Minimum number of similar features required
    
    Returns:
        List of potential duplicate UUID groups
    """
    print(f"Searching for similar UUIDs with threshold {similarity_threshold}...")
    
    # Select numeric features for similarity calculation
    numeric_features = features_df.select_dtypes(include=[np.number]).columns
    numeric_features = [col for col in numeric_features if col not in ['total_records', 'unique_days', 'date_range_days']]
    
    if len(numeric_features) < min_similarity_features:
        print(f"Warning: Only {len(numeric_features)} numeric features available")
        return []
    
    # Normalize features for comparison
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features_df[numeric_features].fillna(0))
    
    similar_groups = []
    processed_uuids = set()
    
    for i, uuid1 in enumerate(features_df['uuid']):
        if uuid1 in processed_uuids:
            continue
            
        similar_uuids = [uuid1]
        
        for j, uuid2 in enumerate(features_df['uuid']):
            if i >= j or uuid2 in processed_uuids:
                continue
            
            # Calculate similarity
            vec1 = features_normalized[i]
            vec2 = features_normalized[j]
            
            # Use multiple similarity metrics
            correlation = np.corrcoef(vec1, vec2)[0, 1] if not np.isnan(np.corrcoef(vec1, vec2)[0, 1]) else 0
            cosine_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0 else 0
            
            # Feature-by-feature similarity
            feature_similarities = []
            for k in range(len(vec1)):
                if not np.isnan(vec1[k]) and not np.isnan(vec2[k]):
                    if abs(vec1[k]) < 1e-10 and abs(vec2[k]) < 1e-10:  # Both near zero
                        feature_similarities.append(1.0)
                    else:
                        diff = abs(vec1[k] - vec2[k])
                        avg = (abs(vec1[k]) + abs(vec2[k])) / 2
                        feature_similarities.append(1 - min(diff / (avg + 1e-10), 1.0))
            
            avg_feature_similarity = np.mean(feature_similarities)
            
            # Combined similarity score
            combined_similarity = (correlation * 0.3 + cosine_sim * 0.3 + avg_feature_similarity * 0.4)
            
            if combined_similarity > similarity_threshold:
                similar_uuids.append(uuid2)
        
        if len(similar_uuids) > 1:
            similar_groups.append({
                'uuids': similar_uuids,
                'group_size': len(similar_uuids),
                'similarity_score': combined_similarity if len(similar_uuids) == 2 else 'multiple'
            })
            processed_uuids.update(similar_uuids)
    
    print(f"Found {len(similar_groups)} groups of similar UUIDs")
    
    return similar_groups

def analyze_uuid_temporal_overlap(df, uuid_groups):
    """
    Analyze temporal overlap between potentially duplicate UUIDs.
    UUIDs representing the same device should never be active simultaneously.
    
    Args:
        df: Original DataFrame
        uuid_groups: List of UUID groups from find_similar_uuids()
    
    Returns:
        Analysis results for each group
    """
    print("Analyzing temporal overlap between similar UUIDs...")
    
    overlap_analysis = []
    
    for group in uuid_groups:
        uuids = group['uuids']
        group_analysis = {
            'uuids': uuids,
            'group_size': len(uuids),
            'temporal_analysis': {}
        }
        
        # Get data for all UUIDs in group
        group_data = df[df['uuid'].isin(uuids)].copy()
        group_data['eventtime'] = pd.to_datetime(group_data['eventtime'])
        
        # Analyze pairwise temporal overlaps
        overlaps = []
        for i, uuid1 in enumerate(uuids):
            for j, uuid2 in enumerate(uuids):
                if i >= j:
                    continue
                
                data1 = group_data[group_data['uuid'] == uuid1]
                data2 = group_data[group_data['uuid'] == uuid2]
                
                if len(data1) == 0 or len(data2) == 0:
                    continue
                
                # Find time ranges for each UUID
                range1 = (data1['eventtime'].min(), data1['eventtime'].max())
                range2 = (data2['eventtime'].min(), data2['eventtime'].max())
                
                # Check for temporal overlap
                overlap_start = max(range1[0], range2[0])
                overlap_end = min(range1[1], range2[1])
                
                has_overlap = overlap_start < overlap_end
                
                if has_overlap:
                    overlap_duration = (overlap_end - overlap_start).total_seconds() / 3600  # hours
                    
                    # Check for simultaneous activity (within 1 hour)
                    simultaneous_count = 0
                    for _, row1 in data1.iterrows():
                        time_diffs = abs((data2['eventtime'] - row1['eventtime']).dt.total_seconds() / 3600)
                        simultaneous_count += (time_diffs < 1).sum()
                    
                    overlaps.append({
                        'uuid1': uuid1,
                        'uuid2': uuid2,
                        'has_temporal_overlap': has_overlap,
                        'overlap_duration_hours': overlap_duration,
                        'simultaneous_activities': simultaneous_count,
                        'range1': range1,
                        'range2': range2
                    })
        
        group_analysis['temporal_analysis']['overlaps'] = overlaps
        group_analysis['max_simultaneous'] = max([o['simultaneous_activities'] for o in overlaps]) if overlaps else 0
        group_analysis['has_any_overlap'] = any([o['has_temporal_overlap'] for o in overlaps])
        
        overlap_analysis.append(group_analysis)
    
    return overlap_analysis

def investigate_uuid_duplicates(df):
    """
    Complete workflow to investigate potential UUID duplicates.
    
    Args:
        df: DataFrame with AC device data
    
    Returns:
        Dictionary with investigation results
    """
    print("="*60)
    print("INVESTIGATING POTENTIAL UUID DUPLICATES")
    print("="*60)
    
    # Step 1: Extract behavioral features
    features_df = extract_uuid_behavioral_features(df)
    
    # Step 2: Find similar UUIDs
    similar_groups = find_similar_uuids(features_df, similarity_threshold=0.90)
    
    if not similar_groups:
        print("No similar UUID groups found.")
        return {'features_df': features_df, 'similar_groups': [], 'temporal_analysis': []}
    
    # Step 3: Analyze temporal overlaps
    temporal_analysis = analyze_uuid_temporal_overlap(df, similar_groups)
    
    # Step 4: Report findings
    print(f"\nSUMMARY OF UUID INVESTIGATION:")
    print(f"Total UUIDs analyzed: {len(features_df)}")
    print(f"Similar UUID groups found: {len(similar_groups)}")
    
    suspicious_groups = 0
    for i, analysis in enumerate(temporal_analysis):
        print(f"\nGroup {i+1}: {analysis['uuids']}")
        print(f"  Group size: {analysis['group_size']}")
        print(f"  Has temporal overlap: {analysis['has_any_overlap']}")
        print(f"  Max simultaneous activities: {analysis['max_simultaneous']}")
        
        # Flag as suspicious if no temporal overlap (could be same device)
        if not analysis['has_any_overlap'] and analysis['group_size'] > 1:
            print(f"  ⚠️  SUSPICIOUS: No temporal overlap - could be same device!")
            suspicious_groups += 1
    
    print(f"\nSuspicious groups (potential duplicates): {suspicious_groups}")
    
    return {
        'features_df': features_df,
        'similar_groups': similar_groups,
        'temporal_analysis': temporal_analysis,
        'suspicious_groups': suspicious_groups
    }

# =============================================================================
# PROBLEMATIC DEVICE DETECTION FUNCTIONS
# =============================================================================

def detect_sensor_anomalies(df):
    """
    Detect devices with sensor reading anomalies.
    
    Args:
        df: DataFrame with AC device data
    
    Returns:
        DataFrame with anomaly flags per UUID
    """
    print("Detecting sensor anomalies...")
    
    anomaly_results = []
    
    for uuid in df['uuid'].unique():
        uuid_data = df[df['uuid'] == uuid].copy()
        
        if len(uuid_data) < 20:  # Skip devices with too little data
            continue
        
        anomalies = {
            'uuid': uuid,
            'total_records': len(uuid_data),
            
            # Temperature anomalies
            'impossible_temps': 0,
            'extreme_temp_jumps': 0,
            'temp_reading_errors': 0,
            'constant_temp_periods': 0,
            
            # Humidity anomalies (if available)
            'impossible_humidity': 0,
            'humidity_errors': 0,
            
            # Behavioral anomalies
            'impossible_desired_temps': 0,
            'extreme_temp_preferences': 0,
            'inconsistent_behavior': 0,
        }
        
        # Check for impossible temperature readings
        impossible_current = (uuid_data['currentt'] < 5) | (uuid_data['currentt'] > 50)
        impossible_desired = (uuid_data['desiredt'] < 10) | (uuid_data['desiredt'] > 35)
        
        anomalies['impossible_temps'] = impossible_current.sum()
        anomalies['impossible_desired_temps'] = impossible_desired.sum()
        
        # Check for extreme temperature jumps (sensor errors)
        if len(uuid_data) > 1:
            uuid_data_sorted = uuid_data.sort_values('eventtime')
            temp_diffs = uuid_data_sorted['currentt'].diff().abs()
            extreme_jumps = (temp_diffs > 10).sum()  # >10°C jump between readings
            anomalies['extreme_temp_jumps'] = extreme_jumps
            
            # Check for constant temperature periods (sensor stuck)
            constant_periods = 0
            for i in range(len(uuid_data_sorted) - 5):
                temp_window = uuid_data_sorted['currentt'].iloc[i:i+6]
                if temp_window.std() < 0.1:  # Constant temperature for 6+ readings
                    constant_periods += 1
            anomalies['constant_temp_periods'] = constant_periods
        
        # Check humidity anomalies if available
        if 'indoor_humidity' in uuid_data.columns:
            invalid_humidity = (uuid_data['indoor_humidity'] < 0) | (uuid_data['indoor_humidity'] > 100)
            anomalies['impossible_humidity'] = invalid_humidity.sum()
        
        # Check for extreme temperature preferences
        temp_range = uuid_data['desiredt'].max() - uuid_data['desiredt'].min()
        if temp_range > 15:  # >15°C range in preferences
            anomalies['extreme_temp_preferences'] = 1
        
        # Check for inconsistent behavior patterns
        if 'aitempchanged' in uuid_data.columns:
            # High frequency of AI temp changes might indicate erratic behavior
            change_rate = uuid_data['aitempchanged'].mean()
            if change_rate > 0.8:  # >80% of readings have temp changes
                anomalies['inconsistent_behavior'] = 1
        
        # Calculate anomaly score
        total_anomalies = sum([v for k, v in anomalies.items() if k not in ['uuid', 'total_records']])
        anomalies['total_anomaly_score'] = total_anomalies
        anomalies['anomaly_rate'] = total_anomalies / max(anomalies['total_records'], 1)
        
        anomaly_results.append(anomalies)
    
    return pd.DataFrame(anomaly_results)

def detect_irregular_usage_patterns(df):
    """
    Detect devices with irregular usage patterns that might indicate problems.
    
    Args:
        df: DataFrame with AC device data
    
    Returns:
        DataFrame with usage pattern analysis per UUID
    """
    print("Detecting irregular usage patterns...")
    
    pattern_results = []
    
    for uuid in df['uuid'].unique():
        uuid_data = df[df['uuid'] == uuid].copy()
        
        if len(uuid_data) < 50:  # Skip devices with too little data
            continue
        
        uuid_data['eventtime'] = pd.to_datetime(uuid_data['eventtime'])
        uuid_data = uuid_data.sort_values('eventtime')
        
        patterns = {
            'uuid': uuid,
            'total_records': len(uuid_data),
            'analysis_period_days': (uuid_data['eventtime'].max() - uuid_data['eventtime'].min()).days + 1,
            
            # Temporal irregularities
            'irregular_logging': 0,
            'large_gaps': 0,
            'burst_logging': 0,
            
            # Usage pattern irregularities
            'erratic_temperature_control': 0,
            'unusual_usage_times': 0,
            'inconsistent_sessions': 0,
        }
        
        # Check logging intervals
        time_diffs = uuid_data['eventtime'].diff().dt.total_seconds() / 60  # minutes
        time_diffs = time_diffs.dropna()
        
        if len(time_diffs) > 0:
            # Detect irregular logging (should be ~5 minutes)
            expected_interval = 5  # minutes
            irregular_intervals = abs(time_diffs - expected_interval) > 10  # >10 min deviation
            patterns['irregular_logging'] = irregular_intervals.mean()
            
            # Detect large gaps (>6 hours between readings)
            large_gaps = (time_diffs > 360).sum()
            patterns['large_gaps'] = large_gaps
            
            # Detect burst logging (multiple readings within 1 minute)
            burst_logging = (time_diffs < 1).sum()
            patterns['burst_logging'] = burst_logging
        
        # Check for erratic temperature control
        if 'desiredt' in uuid_data.columns:
            temp_changes = uuid_data['desiredt'].diff().abs()
            temp_changes = temp_changes.dropna()
            
            if len(temp_changes) > 0:
                # High frequency of large temperature changes
                large_changes = (temp_changes > 3).sum()  # >3°C changes
                erratic_rate = large_changes / len(temp_changes)
                patterns['erratic_temperature_control'] = erratic_rate
        
        # Check usage times (unusual if mostly outside typical hours)
        uuid_data['hour'] = uuid_data['eventtime'].dt.hour
        unusual_hours = ((uuid_data['hour'] < 6) | (uuid_data['hour'] > 23)).mean()
        patterns['unusual_usage_times'] = unusual_hours
        
        # Check session consistency using 'class' if available
        if 'class' in uuid_data.columns:
            # Detect sessions where class doesn't reset properly
            class_resets = (uuid_data['class'].diff() < 0).sum()
            expected_resets = uuid_data.groupby(uuid_data['eventtime'].dt.date).size().sum() - 1
            if expected_resets > 0:
                reset_ratio = class_resets / expected_resets
                if reset_ratio < 0.5:  # Less than 50% of expected resets
                    patterns['inconsistent_sessions'] = 1
        
        # Calculate overall irregularity score
        irregularity_indicators = [
            patterns['irregular_logging'],
            patterns['erratic_temperature_control'],
            patterns['unusual_usage_times'],
            patterns['inconsistent_sessions']
        ]
        patterns['irregularity_score'] = np.mean(irregularity_indicators)
        
        pattern_results.append(patterns)
    
    return pd.DataFrame(pattern_results)

def detect_outlier_devices(df):
    """
    Detect devices that are statistical outliers compared to the population.
    
    Args:
        df: DataFrame with AC device data
    
    Returns:
        DataFrame with outlier analysis per UUID
    """
    print("Detecting statistical outlier devices...")
    
    # First, get device-level statistics
    device_stats = []
    
    for uuid in df['uuid'].unique():
        uuid_data = df[df['uuid'] == uuid].copy()
        
        if len(uuid_data) < 30:  # Skip devices with too little data
            continue
        
        stats_dict = {
            'uuid': uuid,
            'total_records': len(uuid_data),
            'avg_current_temp': uuid_data['currentt'].mean(),
            'std_current_temp': uuid_data['currentt'].std(),
            'avg_desired_temp': uuid_data['desiredt'].mean(),
            'std_desired_temp': uuid_data['desiredt'].std(),
            'avg_temp_diff': (uuid_data['currentt'] - uuid_data['desiredt']).mean(),
            'temp_diff_volatility': (uuid_data['currentt'] - uuid_data['desiredt']).std(),
        }
        
        # Add interval statistics if available
        if 'label_interval' in uuid_data.columns:
            stats_dict.update({
                'avg_interval': uuid_data['label_interval'].mean(),
                'std_interval': uuid_data['label_interval'].std(),
                'median_interval': uuid_data['label_interval'].median(),
            })
        
        # Add humidity stats if available
        if 'indoor_humidity' in uuid_data.columns:
            stats_dict.update({
                'avg_indoor_humidity': uuid_data['indoor_humidity'].mean(),
                'std_indoor_humidity': uuid_data['indoor_humidity'].std(),
            })
        
        device_stats.append(stats_dict)
    
    device_stats_df = pd.DataFrame(device_stats)
    
    # Detect outliers using multiple methods
    outlier_results = []
    
    numeric_cols = device_stats_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'total_records']
    
    for _, row in device_stats_df.iterrows():
        outlier_flags = {
            'uuid': row['uuid'],
            'total_records': row['total_records'],
        }
        
        outlier_count = 0
        
        for col in numeric_cols:
            if pd.isna(row[col]):
                continue
                
            # Z-score method
            col_values = device_stats_df[col].dropna()
            z_score = abs((row[col] - col_values.mean()) / col_values.std()) if col_values.std() > 0 else 0
            
            # IQR method  
            q1, q3 = col_values.quantile([0.25, 0.75])
            iqr = q3 - q1
            is_outlier_iqr = (row[col] < q1 - 1.5 * iqr) or (row[col] > q3 + 1.5 * iqr)
            
            # Flag as outlier if extreme in either method
            is_outlier = (z_score > 3) or is_outlier_iqr
            
            outlier_flags[f'{col}_zscore'] = z_score
            outlier_flags[f'{col}_outlier'] = is_outlier
            
            if is_outlier:
                outlier_count += 1
        
        outlier_flags['total_outlier_features'] = outlier_count
        outlier_flags['outlier_ratio'] = outlier_count / len(numeric_cols)
        
        outlier_results.append(outlier_flags)
    
    return pd.DataFrame(outlier_results)

def identify_problematic_devices(df, anomaly_threshold=0.1, irregularity_threshold=0.3, outlier_threshold=0.3):
    """
    Complete workflow to identify problematic devices.
    
    Args:
        df: DataFrame with AC device data
        anomaly_threshold: Threshold for anomaly rate
        irregularity_threshold: Threshold for irregularity score
        outlier_threshold: Threshold for outlier ratio
    
    Returns:
        Dictionary with analysis results and list of problematic devices
    """
    print("="*60)
    print("IDENTIFYING PROBLEMATIC DEVICES")
    print("="*60)
    
    # Step 1: Detect sensor anomalies
    sensor_anomalies = detect_sensor_anomalies(df)
    
    # Step 2: Detect irregular usage patterns
    usage_patterns = detect_irregular_usage_patterns(df)
    
    # Step 3: Detect statistical outliers
    outlier_analysis = detect_outlier_devices(df)
    
    # Step 4: Combine results
    all_uuids = set(sensor_anomalies['uuid'].tolist() + 
                   usage_patterns['uuid'].tolist() + 
                   outlier_analysis['uuid'].tolist())
    
    problematic_devices = []
    
    for uuid in all_uuids:
        device_issues = {'uuid': uuid, 'issues': [], 'severity_score': 0}
        
        # Check sensor anomalies
        sensor_data = sensor_anomalies[sensor_anomalies['uuid'] == uuid]
        if not sensor_data.empty:
            anomaly_rate = sensor_data.iloc[0]['anomaly_rate']
            if anomaly_rate > anomaly_threshold:
                device_issues['issues'].append(f"High sensor anomaly rate: {anomaly_rate:.3f}")
                device_issues['severity_score'] += anomaly_rate * 2
        
        # Check usage patterns
        pattern_data = usage_patterns[usage_patterns['uuid'] == uuid]
        if not pattern_data.empty:
            irregularity_score = pattern_data.iloc[0]['irregularity_score']
            if irregularity_score > irregularity_threshold:
                device_issues['issues'].append(f"Irregular usage patterns: {irregularity_score:.3f}")
                device_issues['severity_score'] += irregularity_score
        
        # Check outlier status
        outlier_data = outlier_analysis[outlier_analysis['uuid'] == uuid]
        if not outlier_data.empty:
            outlier_ratio = outlier_data.iloc[0]['outlier_ratio']
            if outlier_ratio > outlier_threshold:
                device_issues['issues'].append(f"Statistical outlier: {outlier_ratio:.3f}")
                device_issues['severity_score'] += outlier_ratio
        
        if device_issues['issues']:
            problematic_devices.append(device_issues)
    
    # Sort by severity
    problematic_devices.sort(key=lambda x: x['severity_score'], reverse=True)
    
    # Print summary
    print(f"\nSUMMARY OF PROBLEMATIC DEVICE DETECTION:")
    print(f"Total devices analyzed: {len(all_uuids)}")
    print(f"Devices with sensor anomalies: {(sensor_anomalies['anomaly_rate'] > anomaly_threshold).sum()}")
    print(f"Devices with irregular patterns: {(usage_patterns['irregularity_score'] > irregularity_threshold).sum()}")
    print(f"Devices that are statistical outliers: {(outlier_analysis['outlier_ratio'] > outlier_threshold).sum()}")
    print(f"Total problematic devices: {len(problematic_devices)}")
    
    if problematic_devices:
        print(f"\nTOP 10 MOST PROBLEMATIC DEVICES:")
        for i, device in enumerate(problematic_devices[:10]):
            print(f"{i+1}. UUID: {device['uuid']}")
            print(f"   Severity Score: {device['severity_score']:.3f}")
            for issue in device['issues']:
                print(f"   - {issue}")
    
    return {
        'sensor_anomalies': sensor_anomalies,
        'usage_patterns': usage_patterns,
        'outlier_analysis': outlier_analysis,
        'problematic_devices': problematic_devices,
        'problematic_uuids': [d['uuid'] for d in problematic_devices]
    }

# =============================================================================
# ANALYSIS AND VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_uuid_investigation_results(uuid_investigation):
    """
    Create visualizations for UUID investigation results.
    """
    features_df = uuid_investigation['features_df']
    similar_groups = uuid_investigation['similar_groups']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Temperature preference distribution
    axes[0,0].hist(features_df['avg_desired_temp'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution of Average Desired Temperature')
    axes[0,0].set_xlabel('Average Desired Temperature (°C)')
    axes[0,0].set_ylabel('Number of Devices')
    
    # 2. Usage patterns
    axes[0,1].scatter(features_df['usage_hour_spread'], features_df['weekend_usage_ratio'], alpha=0.6)
    axes[0,1].set_title('Usage Pattern Analysis')
    axes[0,1].set_xlabel('Usage Hour Spread (std)')
    axes[0,1].set_ylabel('Weekend Usage Ratio')
    
    # 3. Temperature variance vs preference stability
    axes[1,0].scatter(features_df['std_current_temp'], features_df['std_desired_temp'], alpha=0.6)
    axes[1,0].set_title('Temperature Variance vs Preference Stability')
    axes[1,0].set_xlabel('Current Temperature Std Dev')
    axes[1,0].set_ylabel('Desired Temperature Std Dev')
    
    # 4. Records per day distribution
    axes[1,1].hist(features_df['records_per_day'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Records per Day')
    axes[1,1].set_xlabel('Records per Day')
    axes[1,1].set_ylabel('Number of Devices')
    
    plt.tight_layout()
    plt.show()
    
    # Print similarity analysis
    if similar_groups:
        print(f"\nDETAILED SIMILARITY ANALYSIS:")
        for i, group in enumerate(similar_groups):
            print(f"\nGroup {i+1}: {len(group['uuids'])} similar devices")
            for uuid in group['uuids']:
                device_info = features_df[features_df['uuid'] == uuid].iloc[0]
                print(f"  {uuid}: {device_info['total_records']} records, "
                      f"avg_desired: {device_info['avg_desired_temp']:.1f}°C, "
                      f"first: {device_info['first_record']}, "
                      f"last: {device_info['last_record']}")

def visualize_problematic_devices(problematic_analysis):
    """
    Create visualizations for problematic device analysis.
    """
    sensor_anomalies = problematic_analysis['sensor_anomalies']
    usage_patterns = problematic_analysis['usage_patterns']
    outlier_analysis = problematic_analysis['outlier_analysis']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sensor anomalies
    axes[0,0].hist(sensor_anomalies['anomaly_rate'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Distribution of Anomaly Rates')
    axes[0,0].set_xlabel('Anomaly Rate')
    axes[0,0].set_ylabel('Number of Devices')
    
    axes[0,1].scatter(sensor_anomalies['total_records'], sensor_anomalies['total_anomaly_score'], alpha=0.6)
    axes[0,1].set_title('Anomaly Score vs Data Volume')
    axes[0,1].set_xlabel('Total Records')
    axes[0,1].set_ylabel('Total Anomaly Score')
    
    # Usage patterns
    axes[0,2].hist(usage_patterns['irregularity_score'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,2].set_title('Distribution of Irregularity Scores')
    axes[0,2].set_xlabel('Irregularity Score')
    axes[0,2].set_ylabel('Number of Devices')
    
    axes[1,0].scatter(usage_patterns['irregular_logging'], usage_patterns['erratic_temperature_control'], alpha=0.6)
    axes[1,0].set_title('Logging vs Temperature Control Issues')
    axes[1,0].set_xlabel('Irregular Logging Rate')
    axes[1,0].set_ylabel('Erratic Temperature Control Rate')
    
    # Outlier analysis
    axes[1,1].hist(outlier_analysis['outlier_ratio'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,1].set_title('Distribution of Outlier Ratios')
    axes[1,1].set_xlabel('Outlier Ratio')
    axes[1,1].set_ylabel('Number of Devices')
    
    axes[1,2].scatter(outlier_analysis['total_outlier_features'], outlier_analysis['total_records'], alpha=0.6)
    axes[1,2].set_title('Outlier Features vs Data Volume')
    axes[1,2].set_xlabel('Number of Outlier Features')
    axes[1,2].set_ylabel('Total Records')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# EXAMPLE USAGE FUNCTIONS
# =============================================================================

def example_complete_investigation(df):
    """
    Complete example of running both UUID and problematic device investigations.
    
    Args:
        df: Your AC device DataFrame
    
    Returns:
        Combined results from both investigations
    """
    print("STARTING COMPLETE AC DEVICE INVESTIGATION")
    print("="*80)
    
    # 1. UUID Duplicate Investigation
    uuid_results = investigate_uuid_duplicates(df)
    
    print("\n" + "="*80)
    
    # 2. Problematic Device Detection
    problematic_results = identify_problematic_devices(df)
    
    # 3. Create visualizations
    print("\nCreating visualizations...")
    visualize_uuid_investigation_results(uuid_results)
    visualize_problematic_devices(problematic_results)
    
    # 4. Generate recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR MODEL IMPROVEMENT")
    print("="*60)
    
    # UUID recommendations
    if uuid_results['suspicious_groups'] > 0:
        print(f"\n🔍 UUID ISSUES FOUND:")
        print(f"- {uuid_results['suspicious_groups']} suspicious UUID groups detected")
        print("- Recommendation: Investigate these groups manually")
        print("- Consider merging data from suspected duplicate UUIDs")
    else:
        print("✅ No suspicious UUID duplicates found")
    
    # Problematic device recommendations  
    problematic_count = len(problematic_results['problematic_devices'])
    total_devices = len(df['uuid'].unique())
    
    print(f"\n🚨 PROBLEMATIC DEVICES:")
    print(f"- {problematic_count} out of {total_devices} devices flagged as problematic")
    print(f"- Problematic device ratio: {problematic_count/total_devices:.1%}")
    
    if problematic_count > 0:
        print("- Recommendation: Remove these devices from training data")
        print("- Expected improvement: Better model performance on 30-120 minute intervals")
        print("- These devices likely contribute to poor tail prediction performance")
        
        # Calculate impact
        problematic_uuids = problematic_results['problematic_uuids']
        original_size = len(df)
        clean_size = len(df[~df['uuid'].isin(problematic_uuids)])
        reduction_pct = (1 - clean_size/original_size) * 100
        
        print(f"- Data reduction: {reduction_pct:.1f}% of records will be removed")
        
        # Check interval distribution impact
        if 'label_interval' in df.columns:
            problematic_data = df[df['uuid'].isin(problematic_uuids)]
            
            intervals_30_60 = ((problematic_data['label_interval'] >= 30) & 
                              (problematic_data['label_interval'] < 60)).sum()
            intervals_60_90 = ((problematic_data['label_interval'] >= 60) & 
                              (problematic_data['label_interval'] < 90)).sum()
            intervals_90_120 = ((problematic_data['label_interval'] >= 90) & 
                               (problematic_data['label_interval'] <= 120)).sum()
            
            print(f"- Problematic devices contribute:")
            print(f"  - {intervals_30_60} records in 30-60 min range")
            print(f"  - {intervals_60_90} records in 60-90 min range") 
            print(f"  - {intervals_90_120} records in 90-120 min range")
    
    return {
        'uuid_investigation': uuid_results,
        'problematic_devices': problematic_results,
        'recommendations': {
            'remove_uuids': problematic_results['problematic_uuids'],
            'investigate_uuid_groups': [g['uuids'] for g in uuid_results['similar_groups']],
            'data_reduction_pct': reduction_pct if problematic_count > 0 else 0
        }
    }

def create_clean_dataset(df, investigation_results):
    """
    Create a cleaned dataset by removing problematic devices.
    
    Args:
        df: Original DataFrame
        investigation_results: Results from example_complete_investigation()
    
    Returns:
        Cleaned DataFrame
    """
    uuids_to_remove = investigation_results['recommendations']['remove_uuids']
    
    print(f"Creating clean dataset...")
    print(f"Original dataset size: {len(df):,} records")
    print(f"Removing {len(uuids_to_remove)} problematic UUIDs")
    
    clean_df = df[~df['uuid'].isin(uuids_to_remove)].copy()
    
    print(f"Clean dataset size: {len(clean_df):,} records")
    print(f"Data reduction: {(1 - len(clean_df)/len(df))*100:.1f}%")
    
    # Check interval distribution changes
    if 'label_interval' in df.columns:
        print(f"\nInterval distribution comparison:")
        
        ranges = [(0, 30), (30, 60), (60, 90), (90, 120)]
        for min_val, max_val in ranges:
            orig_count = ((df['label_interval'] >= min_val) & 
                         (df['label_interval'] < max_val)).sum()
            clean_count = ((clean_df['label_interval'] >= min_val) & 
                          (clean_df['label_interval'] < max_val)).sum()
            
            reduction = (1 - clean_count/orig_count)*100 if orig_count > 0 else 0
            print(f"  {min_val}-{max_val} min: {orig_count:,} → {clean_count:,} ({reduction:.1f}% reduction)")
    
    return clean_df
