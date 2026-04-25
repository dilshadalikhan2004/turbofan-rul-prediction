"""
evaluate.py — Evaluation & Benchmarking with NASA Scoring Function
====================================================================

Computes comprehensive metrics for all models across all subsets:
- RMSE, MAE (standard regression metrics)
- NASA asymmetric scoring function (penalizes late predictions more)
- Anomaly detection: Precision, Recall, F1, AUC-ROC
- Early warning rate: % failures detected >20 cycles in advance
- False alarm rate: % healthy engines incorrectly flagged

NASA Score Formula:
  For each engine i:
    d_i = predicted_RUL_i - actual_RUL_i
    if d_i < 0 (early):  score += exp(-d_i/13) - 1
    if d_i >= 0 (late):   score += exp(d_i/10) - 1

Late predictions are penalized ~3x more because predicting failure
too late is catastrophic in aviation.

CLI: python src/evaluate.py --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, f1_score, roc_auc_score,
)

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")


# ──────────────────────────────────────────────────────────────
# NASA Scoring Function
# ──────────────────────────────────────────────────────────────

def nasa_score(y_true, y_pred):
    """
    Compute NASA's asymmetric scoring function for RUL prediction.

    This is THE metric for CMAPSS — standard in all published work.
    Late predictions are penalized more heavily because in aviation,
    missing a failure is far worse than a false alarm.

    Parameters
    ----------
    y_true : np.ndarray
        Actual RUL values.
    y_pred : np.ndarray
        Predicted RUL values.

    Returns
    -------
    float
        Total NASA score (lower is better).
    """
    d = y_pred - y_true  # difference
    score = 0.0
    for di in d:
        if di < 0:
            # Early prediction — less severe
            score += np.exp(-di / 13.0) - 1.0
        else:
            # Late prediction — more severe (denominator 10 vs 13)
            score += np.exp(di / 10.0) - 1.0
    return score


# ──────────────────────────────────────────────────────────────
# Regression Metrics
# ──────────────────────────────────────────────────────────────

def compute_regression_metrics(y_true, y_pred):
    """
    Compute RMSE, MAE, and NASA score for RUL predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Actual RUL values.
    y_pred : np.ndarray
        Predicted RUL values.

    Returns
    -------
    dict
        Keys: 'rmse', 'mae', 'nasa_score'
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }


# ──────────────────────────────────────────────────────────────
# Classification Metrics (Anomaly Detection)
# ──────────────────────────────────────────────────────────────

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute Precision, Recall, F1, and AUC-ROC for anomaly detection.

    Parameters
    ----------
    y_true : np.ndarray
        Actual anomaly labels (0 or 1).
    y_pred : np.ndarray
        Predicted anomaly labels.
    y_prob : np.ndarray or None
        Predicted anomaly probabilities (for AUC-ROC).

    Returns
    -------
    dict
        Keys: 'precision', 'recall', 'f1', 'auc_roc'
    """
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": 0.0,
    }

    if y_prob is not None and len(np.unique(y_true)) > 1:
        y_prob = y_prob[:min_len]
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["auc_roc"] = 0.0

    return metrics


# ──────────────────────────────────────────────────────────────
# Early Warning & False Alarm Metrics
# ──────────────────────────────────────────────────────────────

def compute_early_warning_rate(test_feat, pred_anomaly, lead_cycles=20):
    """
    Compute what percentage of engine failures were detected
    more than `lead_cycles` in advance.

    Parameters
    ----------
    test_feat : pd.DataFrame
        Test data with 'unit_id', 'cycle', 'RUL', 'anomaly' columns.
    pred_anomaly : np.ndarray
        Predicted anomaly labels.
    lead_cycles : int
        Minimum number of cycles ahead for "early" warning.

    Returns
    -------
    float
        Fraction of failures detected early (0.0 to 1.0).
    """
    test_df = test_feat.copy()
    test_df = test_df.iloc[:len(pred_anomaly)]
    test_df["pred_anomaly"] = pred_anomaly

    engines_detected_early = 0
    total_failing_engines = 0

    for uid in test_df["unit_id"].unique():
        engine = test_df[test_df["unit_id"] == uid].sort_values("cycle")
        # Only count engines that actually reach the anomaly zone
        if engine["RUL"].min() >= 30:
            continue

        total_failing_engines += 1

        # Find first anomaly detection
        anomaly_detections = engine[engine["pred_anomaly"] == 1]
        if len(anomaly_detections) == 0:
            continue

        first_detection_rul = anomaly_detections["RUL"].max()
        if first_detection_rul > lead_cycles:
            engines_detected_early += 1

    if total_failing_engines == 0:
        return 0.0

    return engines_detected_early / total_failing_engines


def compute_false_alarm_rate(test_feat, pred_anomaly):
    """
    Compute what percentage of healthy engine cycles were
    incorrectly flagged as anomalies.

    Parameters
    ----------
    test_feat : pd.DataFrame
        Test data with 'RUL' column.
    pred_anomaly : np.ndarray
        Predicted anomaly labels.

    Returns
    -------
    float
        False alarm rate (0.0 to 1.0).
    """
    test_df = test_feat.iloc[:len(pred_anomaly)].copy()
    test_df["pred_anomaly"] = pred_anomaly

    healthy_mask = test_df["RUL"] >= 30
    if healthy_mask.sum() == 0:
        return 0.0

    false_alarms = ((test_df["pred_anomaly"] == 1) & healthy_mask).sum()
    return false_alarms / healthy_mask.sum()


# ──────────────────────────────────────────────────────────────
# Full Evaluation Pipeline
# ──────────────────────────────────────────────────────────────

def evaluate_model(model_name, y_true_rul, pred_rul, y_true_anomaly,
                   pred_anomaly, pred_prob=None, test_feat=None,
                   subset_id="", verbose=True):
    """
    Compute all metrics for a single model on a single subset.

    Parameters
    ----------
    model_name : str
        Name of the model.
    y_true_rul : np.ndarray
        True RUL values.
    pred_rul : np.ndarray
        Predicted RUL.
    y_true_anomaly : np.ndarray
        True anomaly labels.
    pred_anomaly : np.ndarray
        Predicted anomaly labels.
    pred_prob : np.ndarray or None
        Predicted anomaly probabilities.
    test_feat : pd.DataFrame or None
        Test data for early warning / false alarm computation.
    subset_id : str
        Dataset identifier.
    verbose : bool
        Print metrics.

    Returns
    -------
    dict
        All computed metrics.
    """
    if model_name == "Autoencoder":
        reg_metrics = {"rmse": "N/A", "mae": "N/A", "nasa_score": "N/A"}
    else:
        reg_metrics = compute_regression_metrics(y_true_rul, pred_rul)

    clf_metrics = compute_classification_metrics(
        y_true_anomaly, pred_anomaly, pred_prob
    )

    result = {
        "model": model_name,
        "subset": subset_id,
        **reg_metrics,
        **clf_metrics,
    }

    # Early warning and false alarm
    if test_feat is not None:
        result["early_warning_rate"] = compute_early_warning_rate(
            test_feat, pred_anomaly
        )
        result["false_alarm_rate"] = compute_false_alarm_rate(
            test_feat, pred_anomaly
        )
    else:
        result["early_warning_rate"] = 0.0
        result["false_alarm_rate"] = 0.0

    if verbose:
        print(f"\n  {model_name} — {subset_id}:")
        if model_name == "Autoencoder":
            print(f"    RMSE:               N/A")
            print(f"    MAE:                N/A")
            print(f"    NASA Score:         N/A")
        else:
            print(f"    RMSE:               {result['rmse']:.2f}")
            print(f"    MAE:                {result['mae']:.2f}")
            print(f"    NASA Score:         {result['nasa_score']:.2f}")
        print(f"    Precision:          {result['precision']:.4f}")
        print(f"    Recall:             {result['recall']:.4f}")
        print(f"    F1:                 {result['f1']:.4f}")
        print(f"    AUC-ROC:            {result['auc_roc']:.4f}")
        print(f"    Early Warning Rate: {result['early_warning_rate']:.2%}")
        print(f"    False Alarm Rate:   {result['false_alarm_rate']:.2%}")

    return result


def save_benchmark_metrics(all_metrics):
    """
    Save all model metrics to benchmark_metrics.csv.

    Parameters
    ----------
    all_metrics : list of dict
        List of metric dictionaries from evaluate_model().
    """
    os.makedirs(METRICS_DIR, exist_ok=True)
    df = pd.DataFrame(all_metrics)
    path = os.path.join(METRICS_DIR, "benchmark_metrics.csv")
    df.to_csv(path, index=False)
    print(f"\n  Benchmark metrics saved to: {path}")
    return df


# ──────────────────────────────────────────────────────────────
# CLI: Run Full Pipeline
# ──────────────────────────────────────────────────────────────

def run_full_pipeline(subsets=None, verbose=True):
    """
    Execute the complete evaluation pipeline for all models and subsets.

    This is the main entry point — runs data prep, feature engineering,
    trains all models, evaluates, and saves results.

    Parameters
    ----------
    subsets : list of str or None
        Subsets to process. Default: all 4 (FD001–FD004).
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Complete benchmark metrics table.
    """
    from data_prep import prepare_all_subsets, RETAINED_SENSORS
    from features import process_subset_features, get_feature_columns
    from eda import run_all_eda
    from models.random_forest import train_and_evaluate_rf, get_flat_features
    from models.lstm_model import train_and_evaluate_lstm
    from models.autoencoder import train_and_evaluate_autoencoder
    from models.ensemble import build_ensemble

    if subsets is None:
        subsets = ["FD001", "FD002", "FD003", "FD004"]

    all_metrics = []

    # ── Step 1: Data Preparation ──
    print("\n" + "=" * 70)
    print("  STEP 1: Data Preparation")
    print("=" * 70)
    all_data = prepare_all_subsets(save=True, verbose=verbose)

    # ── Step 2: EDA ──
    print("\n" + "=" * 70)
    print("  STEP 2: Exploratory Data Analysis")
    print("=" * 70)
    run_all_eda(all_data)

    for subset_id in subsets:
        print("\n" + "#" * 70)
        print(f"  PROCESSING: {subset_id}")
        print("#" * 70)

        # ── Step 3: Feature Engineering ──
        print("\n  STEP 3: Feature Engineering")
        feat_result = process_subset_features(
            all_data[subset_id], subset_id, save=True, verbose=verbose
        )
        feature_cols = feat_result["feature_cols"]

        # Get true labels for test
        y_test_rul = feat_result["y_test_rul"]
        y_test_anomaly = feat_result["y_test_anomaly"]
        y_val_rul = feat_result["y_val_rul"]

        # ── Step 4a: Random Forest ──
        print("\n  STEP 4a: Random Forest")
        rf_results = train_and_evaluate_rf(
            feat_result["train_feat"], feat_result["val_feat"],
            feat_result["test_feat"], feature_cols, subset_id,
            verbose=verbose,
        )

        # Evaluate RF — use flat features for anomaly labels
        _, _, y_test_anom_flat = get_flat_features(
            feat_result["test_feat"], feature_cols
        )
        rf_metrics = evaluate_model(
            "Random Forest", y_test_rul,
            rf_results["pred_rul_test"][:len(y_test_rul)],
            y_test_anom_flat[:len(y_test_rul)],
            rf_results["pred_anomaly_test"][:len(y_test_rul)],
            rf_results["pred_prob_test"][:len(y_test_rul)],
            feat_result["test_feat"], subset_id, verbose=verbose,
        )
        all_metrics.append(rf_metrics)

        # ── Step 4b: LSTM ──
        print("\n  STEP 4b: LSTM")
        lstm_results = train_and_evaluate_lstm(
            feat_result, subset_id, epochs=100, verbose=verbose,
        )

        lstm_metrics = evaluate_model(
            "LSTM", y_test_rul,
            lstm_results["pred_rul_test"][:len(y_test_rul)],
            y_test_anomaly[:len(lstm_results["pred_anomaly_test"])],
            lstm_results["pred_anomaly_test"][:len(y_test_anomaly)],
            lstm_results["pred_prob_test"][:len(y_test_anomaly)],
            feat_result["test_feat"], subset_id, verbose=verbose,
        )
        all_metrics.append(lstm_metrics)

        # ── Step 4c: Autoencoder ──
        print("\n  STEP 4c: Autoencoder")
        ae_results = train_and_evaluate_autoencoder(
            feat_result["train_feat"], feat_result["val_feat"],
            feat_result["test_feat"], feature_cols, subset_id,
            verbose=verbose,
        )

        ae_metrics = evaluate_model(
            "Autoencoder",
            feat_result["test_feat"]["RUL"].values,
            feat_result["test_feat"]["RUL"].values,  # AE doesn't predict RUL
            (feat_result["test_feat"]["RUL"] < 30).astype(int).values,
            ae_results["pred_anomaly_test"],
            ae_results["recon_errors_test"],
            feat_result["test_feat"], subset_id, verbose=verbose,
        )
        all_metrics.append(ae_metrics)

        # ── Step 4d: Ensemble ──
        print("\n  STEP 4d: Ensemble")
        ensemble_results = build_ensemble(
            rf_results, lstm_results, y_val_rul, y_test_rul,
            subset_id, verbose=verbose,
        )

        ensemble_metrics = evaluate_model(
            "Ensemble", y_test_rul,
            ensemble_results["pred_rul_test"][:len(y_test_rul)],
            y_test_anom_flat[:len(ensemble_results["pred_anomaly_test"])],
            ensemble_results["pred_anomaly_test"],
            ensemble_results["pred_prob_test"],
            feat_result["test_feat"], subset_id, verbose=verbose,
        )
        all_metrics.append(ensemble_metrics)

    # ── Save all metrics ──
    metrics_df = save_benchmark_metrics(all_metrics)

    # ── Print summary table ──
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(metrics_df.to_string(index=False))

    return metrics_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turbofan RUL Prediction — Full Evaluation Pipeline"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run full pipeline on all 4 subsets",
    )
    parser.add_argument(
        "--subset", type=str, nargs="+",
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="Run on specific subsets",
    )
    args = parser.parse_args()

    if args.all:
        run_full_pipeline(verbose=True)
    elif args.subset:
        run_full_pipeline(subsets=args.subset, verbose=True)
    else:
        # Default: run all
        run_full_pipeline(verbose=True)
