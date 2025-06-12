# Akurasi-naik-signifikan

import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

# --- 1. Method 1: Feature Distribution Analysis ---

def investigate_feature_distributions(df: pd.DataFrame, feature: str):
    """
    Compares the distribution of a single feature across two target interval segments
    and performs a Kolmogorov-Smirnov (K-S) test.

    Args:
        df (pd.DataFrame): The DataFrame containing features and the 'y_true' column.
        feature (str): The name of the feature column to investigate.
    """
    print(f"\n--- 1. Distribution Analysis for Feature: '{feature}' ---")
    
    # Split the data into the two segments of interest
    segment1 = df[(df['y_true'] > 0) & (df['y_true'] <= 30)][feature]
    segment2 = df[(df['y_true'] > 30) & (df['y_true'] <= 60)][feature]
    
    # --- Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))
    sns.kdeplot(segment1, label='0-30 min Interval', fill=True, color='#2a9d8f', lw=2)
    sns.kdeplot(segment2, label='30-60 min Interval', fill=True, color='#e76f51', lw=2)
    plt.title(f"Distribution Comparison for '{feature}'", fontsize=16, weight='bold')
    plt.legend()
    plt.show()
    
    # --- Statistical Test ---
    # The K-S test determines if two samples are drawn from the same distribution.
    # A low p-value (e.g., < 0.05) suggests the distributions are significantly different.
    ks_stat, p_value = ks_2samp(segment1.dropna(), segment2.dropna())
    
    print(f"Kolmogorov-Smirnov Test Results for '{feature}':")
    print(f"  KS Statistic: {ks_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  Conclusion: The distributions are statistically different (p < 0.05).")
    else:
        print("  Conclusion: There is no statistical evidence that the distributions are different (p >= 0.05).")


# --- 2. Method 2: Specialized Feature Importance ---

def compare_feature_importance(df: pd.DataFrame, features: list):
    """
    Trains specialist LightGBM models on each segment and compares their
    top feature importances.
    """
    print("\n--- 2. Specialized Feature Importance Analysis ---")
    
    # Prepare data for each segment
    df_segment1 = df[(df['y_true'] > 0) & (df['y_true'] <= 30)].copy()
    df_segment2 = df[(df['y_true'] > 30) & (df['y_true'] <= 60)].copy()
    
    if df_segment1.empty or df_segment2.empty:
        print("One of the segments is empty. Cannot perform feature importance comparison.")
        return

    # Train a model for each segment
    lgb_params = {'objective': 'regression_l1', 'metric': 'mae', 'n_estimators': 200, 'learning_rate': 0.05, 'verbose': -1, 'n_jobs': -1, 'seed': 42}
    
    model1 = lgb.LGBMRegressor(**lgb_params).fit(df_segment1[features], df_segment1['y_true'])
    model2 = lgb.LGBMRegressor(**lgb_params).fit(df_segment2[features], df_segment2['y_true'])
    
    # Get and compare feature importances
    f_imp1 = pd.DataFrame({'feature': features, 'importance': model1.feature_importances_}).sort_values('importance', ascending=False).head(10)
    f_imp2 = pd.DataFrame({'feature': features, 'importance': model2.feature_importances_}).sort_values('importance', ascending=False).head(10)

    print("Top 10 Most Important Features:")
    print("-" * 60)
    print("For 0-30 min Intervals\t\tFor 30-60 min Intervals")
    print("-" * 60)
    for (idx1, row1), (idx2, row2) in zip(f_imp1.iterrows(), f_imp2.iterrows()):
        print(f"{row1['feature']:<30}\t{row2['feature']}")
    print("-" * 60)
    
    # Evidence is strong if these two lists are substantially different.
    common_features = len(set(f_imp1['feature']) & set(f_imp2['feature']))
    print(f"\nConclusion: {common_features} out of the top 10 features are shared between the two segments.")


# --- 3. Method 3: SHAP Interaction Deep Dive ---

def compare_shap_summary_plots(df: pd.DataFrame, features: list):
    """
    Uses SHAP to visualize and compare feature impacts for the two specialist models.
    """
    print("\n--- 3. SHAP Interaction Analysis (Deep Dive) ---")
    
    # Re-using the specialist models from the previous step
    df_segment1 = df[(df['y_true'] > 0) & (df['y_true'] <= 30)].copy()
    df_segment2 = df[(df['y_true'] > 30) & (df['y_true'] <= 60)].copy()
    
    if df_segment1.empty or df_segment2.empty:
        print("One of the segments is empty. Cannot perform SHAP analysis.")
        return
        
    lgb_params = {'objective': 'regression_l1', 'verbose': -1, 'seed': 42}
    model1 = lgb.LGBMRegressor(**lgb_params).fit(df_segment1[features], df_segment1['y_true'])
    model2 = lgb.LGBMRegressor(**lgb_params).fit(df_segment2[features], df_segment2['y_true'])
    
    # Explain and compute SHAP values for both models
    explainer1 = shap.TreeExplainer(model1)
    shap_values1 = explainer1.shap_values(df_segment1[features])
    
    explainer2 = shap.TreeExplainer(model2)
    shap_values2 = explainer2.shap_values(df_segment2[features])

    # Plotting SHAP summary plots side-by-side
    print("\nSHAP Summary Plots: Comparing Feature Impact")
    print("Reading the plots: Each dot is a sample. Red = high feature value, Blue = low feature value.")
    print("Impact on x-axis: Positive SHAP value means the feature pushes the prediction higher.")
    
    shap.summary_plot(shap_values1, df_segment1[features], show=False)
    plt.title("SHAP Feature Impact (0-30 min Intervals)", fontsize=14)
    plt.show()
    
    shap.summary_plot(shap_values2, df_segment2[features], show=False)
    plt.title("SHAP Feature Impact (30-60 min Intervals)", fontsize=14)
    plt.show()
    
    print("\nConclusion: Look for features that have different color patterns or opposing effects.")
    print("For example, if a feature is red on the right for one plot (high value -> higher prediction) but red on the left for the other (high value -> lower prediction), you have found a 'conflicting pattern'.")


# --- Main Execution Block ---
if __name__ == '__main__':
    # For demonstration, create a synthetic DataFrame with characteristics we want to find.
    # We will engineer 'temp_diff' to have conflicting patterns.
    size = 4000
    df = pd.DataFrame({
        'temp_diff': np.random.randn(size) * 5,
        'humidity': np.random.rand(size) * 100,
        'pressure': np.random.rand(size) * 10 + 1000,
        'historical_avg': np.random.rand(size) * 50 + 15,
        'y_true': np.random.uniform(1, 70, size=size) # This is our target
    })
    
    # Engineer conflicting patterns for 'temp_diff'
    # For short intervals, high temp_diff leads to lower y_true
    short_mask = df['y_true'] <= 30
    df.loc[short_mask, 'y_true'] -= df.loc[short_mask, 'temp_diff'] * 0.5
    
    # For medium intervals, high temp_diff leads to higher y_true
    mid_mask = (df['y_true'] > 30) & (df['y_true'] <= 60)
    df.loc[mid_mask, 'y_true'] += df.loc[mid_mask, 'temp_diff'] * 0.5
    
    df['y_true'] = df['y_true'].clip(1, 120)

    features_to_test = ['temp_diff', 'humidity', 'pressure', 'historical_avg']
    
    # --- Run the Investigation ---
    # 1. Compare feature distributions
    investigate_feature_distributions(df, feature='temp_diff')
    investigate_feature_distributions(df, feature='historical_avg')
    
    # 2. Compare feature importances from specialist models
    compare_feature_importance(df, features=features_to_test)
    
    # 3. Perform a deep dive with SHAP
    compare_shap_summary_plots(df, features=features_to_test)

    



Of course. It is essential to provide a clear and well-reasoned report for your boss that not only explains the final outcome but also details the logical progression of our strategy. This analysis will first cover the background—why these advanced methods were chosen and expected to work—and then provide a deep diagnosis of why the experimental results indicate they failed for your specific problem.

---

### **Post-Mortem Analysis of Advanced Modeling Strategies**

**Objective:** To provide a detailed explanation for why two advanced modeling strategies—(1) a Mixture Density Network (MDN) with a custom loss, and (2) the MDN with a Two-Stage Funnel Calibration—did not succeed in simultaneously reducing Mean Absolute Error (MAE) for both the 0-30 minute and 30-60 minute call interval ranges.

---

### **1. Method: Mixture Density Network (MDN) with Custom Weighted Loss**

#### **A. Background: The Initial Hypothesis and Rationale**

Before this experiment, we faced a core challenge: standard models that predict a single average value are ill-suited for our `label_interval` target variable. [cite_start]The data showed that the target is heavily right-skewed, with a low median (around 20-23 minutes) and a much higher mean (around 72-75 minutes)[cite: 66, 79]. This indicates a complex, non-symmetrical probability distribution.

Our hypothesis was twofold:

1.  [cite_start]**Why MDN:** A Mixture Density Network was chosen because it does not predict a single value; it predicts a full probability distribution, typically as a mixture of several Gaussian (bell curve) distributions[cite: 108]. This architecture is theoretically ideal for capturing the complex, multi-peaked nature of our data, offering a much richer and more accurate representation of the likely outcomes.
2.  **Why Custom Weighted Loss:** We recognized that our business goal was not to be accurate across all possible intervals. The priority was to drastically reduce errors for short intervals (0-60 minutes). We designed a custom loss function to force the model to focus on this goal. By applying a higher penalty weight to errors on samples with a true interval under 60 minutes, we were directly translating our business priority into a mathematical objective for the model to optimize.

**Expected Outcome:** We anticipated that this strategy would force the model to become a specialist on short intervals, leading to a significant decrease in MAE for the 0-30 and 30-60 minute ranges, while accepting that errors on longer intervals might increase as a trade-off.

#### **B. Analysis: Why It Failed Based on Experimental Results**

**Observed Experimental Result:** The strategy was partially successful, achieving a remarkable MAE reduction in the 0-30 minute range (from 12 to 5). However, it simultaneously caused an *increase* in MAE for the 30-60 minute range.

**Primary Reason for Failure: The "Conflicting Objectives" Problem**

The experimental result provides strong evidence that a single model cannot be optimized for both target zones simultaneously.

* **Deep Reasoning:** A neural network, even a complex one like an MDN, is ultimately a single function defined by a single set of learned weights. Our weighted loss function effectively told the model: "The most important thing in the world is to be accurate on 0-30 minute intervals." The model complied, adjusting its internal weights to find patterns specific to that extremely short-term user behavior. However, the experimental result shows that the parameters required to master this 0-30 minute task are fundamentally different from, and likely in conflict with, the parameters needed to master the 30-60 minute task.
* **Conclusion:** By forcing the model to hyper-specialize on the first segment, we inadvertently made it *worse* at the second. The model couldn't find a single set of parameters that was optimal for both ranges, so it found a state that was excellent for the most heavily-weighted range at the expense of all others, including our other priority segment.

---

### **2. Method: MDN with Two-Stage Funnel Calibration**

#### **A. Background: The Initial Hypothesis and Rationale**

This method was designed specifically to solve the "Conflicting Objectives" problem identified above. The strategy was to "divide and conquer."

1.  **Hypothesis for Stage 1 (The Funnel):** We believed we could train an accurate binary classifier (`LGBMClassifier`) to act as a "gatekeeper." Its sole purpose was to look at the input features and predict whether the resulting interval would be "short" (<= 60 minutes) or "long."
2.  **Hypothesis for Stage 2 (The Specialist):** For all the data points the classifier identified as "short," we would then apply a specialist calibration model. This calibrator would be trained *only* on the errors from short intervals, allowing it to learn the precise correction patterns for our target zones without being confused by data from long intervals.

**Expected Outcome:** We hypothesized this would be a definitive solution. By isolating the problem, we expected to apply a highly accurate, specialized fix only where needed, preventing the "seesaw effect" and reducing MAE across both the 0-30 and 30-60 minute ranges.

#### **B. Analysis: Why It Failed Based on Experimental Results**

**Observed Experimental Result:** This complex, multi-stage pipeline failed to deliver the expected improvements and did not solve the trade-off between the two target interval ranges.

**Primary Reason for Failure: The "Weak Funnel Foundation" Problem (Error Propagation)**

The failure of this strategy points to a deeper, more fundamental issue with the data itself, which invalidates the core assumption of the funnel approach.

* **Deep Reasoning:** The entire pipeline's success is critically dependent on the accuracy of the Stage 1 classifier. The experimental result strongly implies that this classifier failed at its job. It was unable to reliably distinguish an upcoming short interval from a long one based on the available features. [cite_start]This is not surprising given the nature of the data; the document notes that the initial state of the AC unit can be unstable and produce erratic logs[cite: 38, 54], which would severely degrade the classifier's "signal."
* **Conclusion:** The specialist calibrator, though powerful in theory, was being aimed at the wrong targets. The classifier was effectively "poisoning the well" by feeding the specialist a mix of true positives, false positives, and missing false negatives. The errors made by this weak foundational stage propagated and were likely amplified by the second stage, leading to no net improvement. The experiment demonstrates that the input features likely do not contain enough clear, upfront signal to reliably predict which *category* of interval will occur, even if they contain enough signal to predict the interval value itself.
