"""
features.py — Feature Engineering for Turbofan RUL Prediction
==============================================================

Creates temporal features from sensor time-series data:
- Rolling statistics (mean, std) at multiple window sizes
- Exponentially weighted moving averages (EWMA)
- Rate of change (first derivative)
- Cumulative degradation score
- Anomaly zone flag (RUL < 30)
- LSTM sequence windows (30 cycles × n_features)

All features are computed per-engine to prevent cross-engine data leakage.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Feature engineering parameters
ROLLING_WINDOWS = [5, 10, 30]
EWMA_ALPHAS = [0.1, 0.3]
SEQUENCE_LENGTH = 30       # LSTM window length
ANOMALY_THRESHOLD = 30     # RUL < 30 = anomaly zone


# ──────────────────────────────────────────────────────────────
# Rolling Statistics
# ──────────────────────────────────────────────────────────────

def add_rolling_features(group, sensor_cols, windows=ROLLING_WINDOWS):
    """
    Compute rolling mean and std for each sensor at multiple window sizes.

    Uses min_periods=1 to avoid NaN at the start of each engine's lifecycle.

    Parameters
    ----------
    group : pd.DataFrame
        Single engine's time-series data (sorted by cycle).
    sensor_cols : list of str
        Sensor column names.
    windows : list of int
        Rolling window sizes.

    Returns
    -------
    pd.DataFrame
        Input with new rolling feature columns appended.
    """
    for w in windows:
        for col in sensor_cols:
            group[f"{col}_rmean_{w}"] = (
                group[col].rolling(window=w, min_periods=1).mean()
            )
            group[f"{col}_rstd_{w}"] = (
                group[col].rolling(window=w, min_periods=1).std().fillna(0)
            )
    return group


# ──────────────────────────────────────────────────────────────
# Exponentially Weighted Moving Average
# ──────────────────────────────────────────────────────────────

def add_ewma_features(group, sensor_cols, alphas=EWMA_ALPHAS):
    """
    Compute EWMA for each sensor at multiple smoothing factors.

    EWMA gives more weight to recent readings, making it sensitive
    to emerging degradation patterns.

    Parameters
    ----------
    group : pd.DataFrame
        Single engine's time-series data.
    sensor_cols : list of str
        Sensor column names.
    alphas : list of float
        EWMA smoothing factors (higher = more weight on recent).

    Returns
    -------
    pd.DataFrame
        Input with EWMA columns appended.
    """
    for alpha in alphas:
        alpha_str = str(alpha).replace(".", "")
        for col in sensor_cols:
            group[f"{col}_ewma_{alpha_str}"] = (
                group[col].ewm(alpha=alpha, adjust=False).mean()
            )
    return group


# ──────────────────────────────────────────────────────────────
# Rate of Change (First Derivative)
# ──────────────────────────────────────────────────────────────

def add_rate_of_change(group, sensor_cols):
    """
    Compute first derivative (rate of change) for each sensor.

    delta = (current_value - previous_value) / 1 cycle.
    First cycle defaults to 0.

    Parameters
    ----------
    group : pd.DataFrame
        Single engine's time-series data.
    sensor_cols : list of str
        Sensor column names.

    Returns
    -------
    pd.DataFrame
        Input with rate-of-change columns appended.
    """
    for col in sensor_cols:
        group[f"{col}_roc"] = group[col].diff().fillna(0)
    return group


# ──────────────────────────────────────────────────────────────
# Cumulative Degradation Score
# ──────────────────────────────────────────────────────────────

def add_cumulative_degradation(group, sensor_cols):
    """
    Compute cumulative degradation score per sensor from cycle 1.

    Cumulative sum of absolute changes — captures total accumulated
    wear regardless of direction.

    Parameters
    ----------
    group : pd.DataFrame
        Single engine's time-series data.
    sensor_cols : list of str
        Sensor column names.

    Returns
    -------
    pd.DataFrame
        Input with cumulative degradation columns appended.
    """
    for col in sensor_cols:
        group[f"{col}_cumdeg"] = group[col].diff().abs().cumsum().fillna(0)
    return group


# ──────────────────────────────────────────────────────────────
# Anomaly Zone Flag
# ──────────────────────────────────────────────────────────────

def add_anomaly_flag(df, threshold=ANOMALY_THRESHOLD):
    """
    Flag cycles in the anomaly zone (RUL < threshold).

    This binary label is the classification target for anomaly detection.
    Engines in the danger zone (close to failure) get flagged.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'RUL' column.
    threshold : int
        RUL threshold below which a cycle is flagged (default 30).

    Returns
    -------
    pd.DataFrame
        Input with 'anomaly' column (0 or 1).
    """
    df = df.copy()
    df["anomaly"] = (df["RUL"] < threshold).astype(int)
    return df


# ──────────────────────────────────────────────────────────────
# Full Feature Engineering Pipeline
# ──────────────────────────────────────────────────────────────

def engineer_features(df, sensor_cols, verbose=True):
    """
    Apply all feature engineering steps to a dataset.

    Processes each engine independently (groupby unit_id) to
    prevent cross-engine data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared data with sensor columns, RUL, health_index.
    sensor_cols : list of str
        Sensor column names.
    verbose : bool
        Show progress bar.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataframe with ~158 total columns.
    """
    df = df.copy().sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    # Add anomaly flag first (doesn't need groupby)
    df = add_anomaly_flag(df)

    # Process per-engine features
    groups = []
    engine_ids = df["unit_id"].unique()
    iterator = tqdm(engine_ids, desc="  Engineering features") if verbose else engine_ids

    for uid in iterator:
        group = df[df["unit_id"] == uid].copy()
        group = group.sort_values("cycle")
        group = add_rolling_features(group, sensor_cols)
        group = add_ewma_features(group, sensor_cols)
        group = add_rate_of_change(group, sensor_cols)
        group = add_cumulative_degradation(group, sensor_cols)
        groups.append(group)

    result = pd.concat(groups, ignore_index=True)

    # Convert to float32 to save memory
    float_cols = result.select_dtypes(include=[np.float64]).columns
    result[float_cols] = result[float_cols].astype(np.float32)

    return result


# ──────────────────────────────────────────────────────────────
# LSTM Sequence Windows
# ──────────────────────────────────────────────────────────────

def get_feature_columns(df, sensor_cols):
    """
    Get the list of feature columns to use for modeling.

    Excludes metadata columns (unit_id, cycle, op_settings, op_cluster, RUL, anomaly).

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe.
    sensor_cols : list of str
        Base sensor column names.

    Returns
    -------
    list of str
        Feature column names for model input.
    """
    exclude = {"unit_id", "cycle", "op_setting_1", "op_setting_2",
               "op_setting_3", "op_cluster", "RUL", "anomaly"}
    return [c for c in df.columns if c not in exclude]


def create_sequences(df, feature_cols, seq_length=SEQUENCE_LENGTH):
    """
    Create sliding-window sequences for LSTM input.

    For each engine, slides a window of `seq_length` cycles across
    the feature matrix. Engines with fewer cycles than seq_length
    are zero-padded from the left.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered data with all feature columns.
    feature_cols : list of str
        Column names to include in sequences.
    seq_length : int
        Number of timesteps per sequence (default 30).

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        (X_seq, y_rul, y_anomaly, seq_info)
        X_seq: shape (n_samples, seq_length, n_features)
        y_rul: shape (n_samples,) — RUL at last timestep
        y_anomaly: shape (n_samples,) — anomaly flag at last timestep
        seq_info: shape (n_samples, 2) — [unit_id, cycle] for tracking
    """
    X_list, y_rul_list, y_anomaly_list, info_list = [], [], [], []

    for uid in df["unit_id"].unique():
        engine = df[df["unit_id"] == uid].sort_values("cycle")
        features = engine[feature_cols].values.astype(np.float32)
        rul_vals = engine["RUL"].values
        anomaly_vals = engine["anomaly"].values
        cycle_vals = engine["cycle"].values

        n_cycles = len(features)

        if n_cycles < seq_length:
            # Zero-pad from the left
            padding = np.zeros(
                (seq_length - n_cycles, features.shape[1]),
                dtype=np.float32,
            )
            padded = np.vstack([padding, features])
            X_list.append(padded)
            y_rul_list.append(rul_vals[-1])
            y_anomaly_list.append(anomaly_vals[-1])
            info_list.append([uid, cycle_vals[-1]])
        else:
            # Sliding window
            for i in range(seq_length, n_cycles + 1):
                X_list.append(features[i - seq_length: i])
                y_rul_list.append(rul_vals[i - 1])
                y_anomaly_list.append(anomaly_vals[i - 1])
                info_list.append([uid, cycle_vals[i - 1]])

    X_seq = np.array(X_list, dtype=np.float32)
    y_rul = np.array(y_rul_list, dtype=np.float32)
    y_anomaly = np.array(y_anomaly_list, dtype=np.float32)
    seq_info = np.array(info_list, dtype=np.int32)

    return X_seq, y_rul, y_anomaly, seq_info


# ──────────────────────────────────────────────────────────────
# Master Pipeline
# ──────────────────────────────────────────────────────────────

def process_subset_features(result_dict, subset_id, save=True, verbose=True):
    """
    Run feature engineering for one CMAPSS subset.

    Parameters
    ----------
    result_dict : dict
        Output from data_prep.prepare_subset() containing
        'train', 'val', 'test', 'sensor_cols'.
    subset_id : str
        Subset identifier (e.g., 'FD001').
    save : bool
        If True, save feature-engineered CSVs and LSTM sequences.
    verbose : bool
        Print progress info.

    Returns
    -------
    dict
        Keys: 'train_feat', 'val_feat', 'test_feat',
              'X_train', 'y_train_rul', 'y_train_anomaly',
              'X_val', 'y_val_rul', 'y_val_anomaly',
              'X_test', 'y_test_rul', 'y_test_anomaly',
              'feature_cols', 'test_info'
    """
    if verbose:
        print(f"\n  Feature engineering: {subset_id}")

    sensor_cols = result_dict["sensor_cols"]

    # Engineer features for train, val, test
    splits = {}
    for split_name in ["train", "val", "test"]:
        if verbose:
            print(f"  Processing {split_name} split...")
        splits[split_name] = engineer_features(
            result_dict[split_name], sensor_cols, verbose=verbose
        )

    # Get feature columns (same across all splits)
    feature_cols = get_feature_columns(splits["train"], sensor_cols)
    if verbose:
        print(f"  Total features: {len(feature_cols)}")

    # Replace any remaining NaN/Inf with 0
    for split_name in splits:
        splits[split_name][feature_cols] = (
            splits[split_name][feature_cols]
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    # Create LSTM sequences
    sequences = {}
    for split_name in ["train", "val", "test"]:
        if verbose:
            print(f"  Creating sequences for {split_name}...")
        X, y_rul, y_anom, info = create_sequences(
            splits[split_name], feature_cols
        )
        sequences[split_name] = (X, y_rul, y_anom, info)
        if verbose:
            print(f"    Shape: {X.shape}")

    # Save if requested
    if save:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        for split_name in ["train", "val", "test"]:
            # Save feature-engineered CSV
            csv_path = os.path.join(
                PROCESSED_DIR, f"{subset_id}_{split_name}_features.csv"
            )
            splits[split_name].to_csv(csv_path, index=False)

            # Save LSTM sequences as .npy
            X, y_rul, y_anom, info = sequences[split_name]
            np.save(
                os.path.join(PROCESSED_DIR, f"{subset_id}_{split_name}_X.npy"),
                X,
            )
            np.save(
                os.path.join(PROCESSED_DIR, f"{subset_id}_{split_name}_y_rul.npy"),
                y_rul,
            )
            np.save(
                os.path.join(PROCESSED_DIR, f"{subset_id}_{split_name}_y_anomaly.npy"),
                y_anom,
            )

        if verbose:
            print(f"  Saved features and sequences to {PROCESSED_DIR}")

    return {
        "train_feat": splits["train"],
        "val_feat": splits["val"],
        "test_feat": splits["test"],
        "X_train": sequences["train"][0],
        "y_train_rul": sequences["train"][1],
        "y_train_anomaly": sequences["train"][2],
        "X_val": sequences["val"][0],
        "y_val_rul": sequences["val"][1],
        "y_val_anomaly": sequences["val"][2],
        "X_test": sequences["test"][0],
        "y_test_rul": sequences["test"][1],
        "y_test_anomaly": sequences["test"][2],
        "test_info": sequences["test"][3],
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_prep import prepare_all_subsets

    print("Step 1: Data Preparation")
    all_data = prepare_all_subsets(save=True, verbose=True)

    print("\nStep 2: Feature Engineering")
    for subset_id in ["FD001", "FD002", "FD003", "FD004"]:
        feat_result = process_subset_features(
            all_data[subset_id], subset_id, save=True, verbose=True
        )
        print(f"\n  {subset_id} feature summary:")
        print(f"    Features: {len(feat_result['feature_cols'])}")
        print(f"    Train sequences: {feat_result['X_train'].shape}")
        print(f"    Val sequences:   {feat_result['X_val'].shape}")
        print(f"    Test sequences:  {feat_result['X_test'].shape}")
