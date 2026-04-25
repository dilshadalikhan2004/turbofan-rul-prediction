"""
data_prep.py — Data Loading, Cleaning, RUL Labeling, and Normalization
========================================================================

Loads NASA CMAPSS turbofan engine degradation datasets (FD001–FD004),
applies column naming, removes uninformative sensors, computes piecewise-
linear RUL targets, clusters operating conditions, normalizes per-cluster,
and computes a composite health index.

Reference: Saxena et al., "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM08, Denver CO, Oct 2008.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

COLUMN_NAMES = (
    ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance across all engines/conditions.
# These carry no predictive signal for degradation.
SENSORS_TO_DROP = [
    "sensor_1", "sensor_5", "sensor_6",
    "sensor_10", "sensor_16", "sensor_18", "sensor_19",
]

RETAINED_SENSORS = [
    s for s in [f"sensor_{i}" for i in range(1, 22)]
    if s not in SENSORS_TO_DROP
]

SUBSET_IDS = ["FD001", "FD002", "FD003", "FD004"]

# Operating condition clusters per subset (from NASA readme)
N_CLUSTERS = {
    "FD001": 1,
    "FD002": 6,
    "FD003": 1,
    "FD004": 6,
}

RUL_CAP = 125  # Piecewise-linear cap — engines don't degrade from day 1


# ──────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────

def load_raw_file(filepath):
    """
    Load a space-delimited CMAPSS data file (no headers).

    Parameters
    ----------
    filepath : str
        Absolute or relative path to the .txt file.

    Returns
    -------
    pd.DataFrame
        DataFrame with named columns.
    """
    df = pd.read_csv(
        filepath, sep=r"\s+", header=None,
        names=COLUMN_NAMES, dtype=np.float32,
    )
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_rul_file(filepath):
    """
    Load a RUL ground-truth file (one integer per line).

    Parameters
    ----------
    filepath : str
        Path to RUL_FDxxx.txt.

    Returns
    -------
    np.ndarray
        Array of true remaining-useful-life values.
    """
    return np.loadtxt(filepath, dtype=int)


def load_dataset(subset_id):
    """
    Load train, test, and RUL ground truth for a given CMAPSS subset.

    Parameters
    ----------
    subset_id : str
        One of 'FD001', 'FD002', 'FD003', 'FD004'.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame, np.ndarray)
        (train_df, test_df, rul_truth)
    """
    train_path = os.path.join(RAW_DIR, f"train_{subset_id}.txt")
    test_path = os.path.join(RAW_DIR, f"test_{subset_id}.txt")
    rul_path = os.path.join(RAW_DIR, f"RUL_{subset_id}.txt")

    train_df = load_raw_file(train_path)
    test_df = load_raw_file(test_path)
    rul_truth = load_rul_file(rul_path)

    return train_df, test_df, rul_truth


# ──────────────────────────────────────────────────────────────
# Cleaning & Sensor Pruning
# ──────────────────────────────────────────────────────────────

def drop_constant_sensors(df):
    """
    Remove sensors known to have near-zero variance (uninformative).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with all 21 sensor columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with 7 constant sensors removed (14 sensors retained).
    """
    cols_to_drop = [c for c in SENSORS_TO_DROP if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ──────────────────────────────────────────────────────────────
# RUL Labeling
# ──────────────────────────────────────────────────────────────

def compute_rul(df, cap=RUL_CAP):
    """
    Compute Remaining Useful Life for each cycle of each engine.

    RUL = max_cycle_for_engine - current_cycle, capped at `cap`.
    The cap implements piecewise-linear degradation — engines operate
    normally until wear-in period ends, then degrade linearly.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with 'unit_id' and 'cycle' columns.
    cap : int
        Maximum RUL value (default 125).

    Returns
    -------
    pd.DataFrame
        Input dataframe with 'RUL' column added.
    """
    max_cycles = df.groupby("unit_id")["cycle"].max()
    df = df.copy()
    df["RUL"] = df.apply(
        lambda row: max_cycles[row["unit_id"]] - row["cycle"], axis=1
    )
    df["RUL"] = df["RUL"].clip(upper=cap)
    return df


def compute_test_rul(test_df, rul_truth):
    """
    Attach ground-truth RUL to test data (only last cycle per engine).

    Parameters
    ----------
    test_df : pd.DataFrame
        Test data with 'unit_id' and 'cycle' columns.
    rul_truth : np.ndarray
        True RUL values (one per engine).

    Returns
    -------
    pd.DataFrame
        Test dataframe with 'RUL' column (computed for ALL cycles).
    """
    test_df = test_df.copy()
    max_cycles = test_df.groupby("unit_id")["cycle"].max()

    # Build a mapping: unit_id → true RUL at last cycle
    unit_ids = sorted(test_df["unit_id"].unique())
    rul_map = dict(zip(unit_ids, rul_truth))

    # For each engine, RUL at cycle t = rul_at_last_cycle + (max_cycle - t)
    rul_values = []
    for _, row in test_df.iterrows():
        uid = row["unit_id"]
        remaining_at_end = rul_map[uid]
        rul_at_cycle = remaining_at_end + (max_cycles[uid] - row["cycle"])
        rul_values.append(rul_at_cycle)

    test_df["RUL"] = np.array(rul_values, dtype=int)
    test_df["RUL"] = test_df["RUL"].clip(upper=RUL_CAP)
    return test_df


# ──────────────────────────────────────────────────────────────
# Operating Condition Clustering & Normalization
# ──────────────────────────────────────────────────────────────

def cluster_operating_conditions(df, n_clusters, random_state=42):
    """
    Cluster operating conditions using KMeans on the 3 op_setting columns.

    For FD001/FD003 (single condition), assigns all rows to cluster 0.
    For FD002/FD004 (six conditions), identifies 6 distinct regimes.

    Parameters
    ----------
    df : pd.DataFrame
        Data with op_setting_1, op_setting_2, op_setting_3 columns.
    n_clusters : int
        Number of operating condition clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (pd.DataFrame, KMeans or None)
        DataFrame with 'op_cluster' column added, and fitted KMeans model.
    """
    df = df.copy()
    op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]

    if n_clusters <= 1:
        df["op_cluster"] = 0
        return df, None

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["op_cluster"] = kmeans.fit_predict(df[op_cols])
    return df, kmeans


def normalize_per_cluster(df, sensor_cols, fit=True, scaler_dict=None):
    """
    Apply MinMaxScaler normalization per operating condition cluster.

    This is critical: global normalization conflates altitude/speed effects
    with degradation signals. Per-cluster normalization isolates true
    degradation trends.

    Parameters
    ----------
    df : pd.DataFrame
        Data with sensor columns and 'op_cluster' column.
    sensor_cols : list of str
        Sensor column names to normalize.
    fit : bool
        If True, fit new scalers. If False, use provided scaler_dict.
    scaler_dict : dict or None
        Pre-fitted scalers keyed by cluster ID (used for test data).

    Returns
    -------
    tuple of (pd.DataFrame, dict)
        Normalized DataFrame and scaler dictionary.
    """
    df = df.copy()
    if scaler_dict is None:
        scaler_dict = {}

    for cluster_id in df["op_cluster"].unique():
        mask = df["op_cluster"] == cluster_id
        if fit:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df.loc[mask, sensor_cols] = scaler.fit_transform(
                df.loc[mask, sensor_cols]
            )
            scaler_dict[cluster_id] = scaler
        else:
            scaler = scaler_dict.get(cluster_id)
            if scaler is not None:
                df.loc[mask, sensor_cols] = scaler.transform(
                    df.loc[mask, sensor_cols]
                )

    return df, scaler_dict


# ──────────────────────────────────────────────────────────────
# Health Index Computation
# ──────────────────────────────────────────────────────────────

def compute_sensor_weights(df, sensor_cols):
    """
    Compute sensor importance weights based on Pearson correlation with RUL.

    Sensors with stronger (negative) correlation with RUL get higher weights.
    Weights are normalized to sum to 1.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with sensor columns and 'RUL' column.
    sensor_cols : list of str
        Sensor column names.

    Returns
    -------
    dict
        Mapping of sensor_name → weight.
    """
    correlations = {}
    for col in sensor_cols:
        corr = abs(df[col].corr(df["RUL"]))
        correlations[col] = corr if not np.isnan(corr) else 0.0

    total = sum(correlations.values())
    if total == 0:
        # Fallback: equal weights
        return {col: 1.0 / len(sensor_cols) for col in sensor_cols}

    return {col: val / total for col, val in correlations.items()}


def compute_health_index(df, sensor_cols, weights):
    """
    Compute a composite health index from weighted sensor readings.

    Health index ranges from 1.0 (healthy, new engine) to 0.0 (failure).
    This is the weighted average of normalized sensor values, inverted
    so that degradation shows as a decrease.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized data with sensor columns.
    sensor_cols : list of str
        Sensor column names.
    weights : dict
        Sensor importance weights (from compute_sensor_weights).

    Returns
    -------
    pd.DataFrame
        DataFrame with 'health_index' column added.
    """
    df = df.copy()
    weighted_sum = np.zeros(len(df), dtype=np.float32)

    for col in sensor_cols:
        weighted_sum += df[col].values * weights.get(col, 0.0)

    # Per-engine: scale so that cycle 1 ≈ 1.0, last cycle ≈ 0.0
    hi_values = np.zeros(len(df), dtype=np.float32)
    for uid in df["unit_id"].unique():
        mask = df["unit_id"] == uid
        engine_vals = weighted_sum[mask]
        val_min = engine_vals.min()
        val_max = engine_vals.max()
        if val_max - val_min > 1e-8:
            # Invert: high raw weighted sum at failure → low health index
            normalized = (engine_vals - val_min) / (val_max - val_min)
            hi_values[mask] = 1.0 - normalized
        else:
            hi_values[mask] = 1.0

    df["health_index"] = hi_values
    return df


# ──────────────────────────────────────────────────────────────
# Train / Validation Split
# ──────────────────────────────────────────────────────────────

def split_train_val(df, val_ratio=0.2, random_state=42):
    """
    Split data into train and validation sets BY ENGINE UNIT.

    All cycles of a given engine stay together — prevents data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Full training data.
    val_ratio : float
        Fraction of engines to hold out for validation.
    random_state : int
        Random seed.

    Returns
    -------
    tuple of (pd.DataFrame, pd.DataFrame)
        (train_split, val_split)
    """
    rng = np.random.RandomState(random_state)
    unit_ids = df["unit_id"].unique()
    rng.shuffle(unit_ids)

    n_val = max(1, int(len(unit_ids) * val_ratio))
    val_ids = set(unit_ids[:n_val])
    train_ids = set(unit_ids[n_val:])

    train_split = df[df["unit_id"].isin(train_ids)].reset_index(drop=True)
    val_split = df[df["unit_id"].isin(val_ids)].reset_index(drop=True)

    return train_split, val_split


# ──────────────────────────────────────────────────────────────
# Master Pipeline
# ──────────────────────────────────────────────────────────────

def prepare_subset(subset_id, verbose=True):
    """
    Run the full data preparation pipeline for one CMAPSS subset.

    Steps:
    1. Load raw data (train + test + RUL truth)
    2. Drop uninformative sensors
    3. Compute RUL labels (piecewise-linear with cap)
    4. Cluster operating conditions
    5. Normalize sensors per operating condition cluster
    6. Compute health index
    7. Split train → train + validation

    Parameters
    ----------
    subset_id : str
        'FD001', 'FD002', 'FD003', or 'FD004'.
    verbose : bool
        Print progress info.

    Returns
    -------
    dict
        Keys: 'train', 'val', 'test', 'rul_truth', 'scaler_dict',
              'sensor_weights', 'kmeans_model'
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Preparing subset: {subset_id}")
        print(f"{'='*60}")

    # 1. Load raw data
    train_df, test_df, rul_truth = load_dataset(subset_id)
    if verbose:
        print(f"  Loaded: {len(train_df)} train rows, "
              f"{len(test_df)} test rows, "
              f"{len(rul_truth)} test engines")

    # 2. Drop constant sensors
    train_df = drop_constant_sensors(train_df)
    test_df = drop_constant_sensors(test_df)
    sensor_cols = [c for c in RETAINED_SENSORS if c in train_df.columns]
    if verbose:
        print(f"  Retained {len(sensor_cols)} informative sensors")

    # 3. Compute RUL
    train_df = compute_rul(train_df, cap=RUL_CAP)
    test_df = compute_test_rul(test_df, rul_truth)
    if verbose:
        print(f"  RUL computed (cap={RUL_CAP})")

    # 4. Cluster operating conditions
    n_clust = N_CLUSTERS[subset_id]
    train_df, kmeans_model = cluster_operating_conditions(train_df, n_clust)
    if kmeans_model is not None:
        test_df = test_df.copy()
        op_cols = ["op_setting_1", "op_setting_2", "op_setting_3"]
        test_df["op_cluster"] = kmeans_model.predict(test_df[op_cols])
    else:
        test_df = test_df.copy()
        test_df["op_cluster"] = 0
    if verbose:
        print(f"  Operating conditions: {n_clust} cluster(s)")

    # 5. Normalize per cluster
    train_df, scaler_dict = normalize_per_cluster(
        train_df, sensor_cols, fit=True
    )
    test_df, _ = normalize_per_cluster(
        test_df, sensor_cols, fit=False, scaler_dict=scaler_dict
    )
    if verbose:
        print(f"  Normalized per operating condition cluster")

    # 6. Health index
    sensor_weights = compute_sensor_weights(train_df, sensor_cols)
    train_df = compute_health_index(train_df, sensor_cols, sensor_weights)
    test_df = compute_health_index(test_df, sensor_cols, sensor_weights)
    if verbose:
        top_sensors = sorted(sensor_weights.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join([f"{s}: {w:.3f}" for s, w in top_sensors])
        print(f"  Health index computed. Top sensors: {top_str}")

    # 7. Train/val split
    train_split, val_split = split_train_val(train_df, val_ratio=0.2)
    if verbose:
        n_train_eng = train_split["unit_id"].nunique()
        n_val_eng = val_split["unit_id"].nunique()
        print(f"  Split: {n_train_eng} train engines, {n_val_eng} val engines")

    return {
        "train": train_split,
        "val": val_split,
        "test": test_df,
        "rul_truth": rul_truth,
        "scaler_dict": scaler_dict,
        "sensor_weights": sensor_weights,
        "kmeans_model": kmeans_model,
        "sensor_cols": sensor_cols,
    }


def prepare_all_subsets(save=True, verbose=True):
    """
    Run data preparation for all 4 CMAPSS subsets and optionally save.

    Parameters
    ----------
    save : bool
        If True, save processed CSVs to data/processed/.
    verbose : bool
        Print progress info.

    Returns
    -------
    dict
        Mapping of subset_id → result dict from prepare_subset().
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    all_results = {}

    for subset_id in SUBSET_IDS:
        result = prepare_subset(subset_id, verbose=verbose)
        all_results[subset_id] = result

        if save:
            for split_name in ["train", "val", "test"]:
                fpath = os.path.join(
                    PROCESSED_DIR, f"{subset_id}_{split_name}.csv"
                )
                result[split_name].to_csv(fpath, index=False)

            if verbose:
                print(f"  Saved processed data to {PROCESSED_DIR}")

    if verbose:
        print(f"\n{'='*60}")
        print("  All subsets prepared successfully!")
        print(f"{'='*60}")

    return all_results


# ──────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = prepare_all_subsets(save=True, verbose=True)

    # Summary statistics
    print("\n  Summary:")
    print(f"  {'Subset':<8} {'Train':>8} {'Val':>8} {'Test':>8} {'Sensors':>8}")
    print(f"  {'-'*40}")
    for sid in SUBSET_IDS:
        r = results[sid]
        print(
            f"  {sid:<8} {len(r['train']):>8,} {len(r['val']):>8,} "
            f"{len(r['test']):>8,} {len(r['sensor_cols']):>8}"
        )
