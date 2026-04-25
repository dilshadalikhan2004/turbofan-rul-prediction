"""
main.py — Master Training & Benchmarking Pipeline
=================================================

This script executes the clean-data benchmarking for all 4 CMAPSS subsets.
It saves models and performance metrics to the results/ folder.
"""

import os
import pandas as pd
import numpy as np
from data_prep import prepare_subset
from features import process_subset_features
from models.random_forest import train_and_evaluate_rf
from models.lstm_model import train_and_evaluate_lstm
from models.autoencoder import train_and_evaluate_autoencoder
from models.ensemble import build_ensemble
from evaluate import evaluate_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")

def main():
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    all_benchmarks = []
    
    print("Starting Clean-Data Benchmarking Pipeline...")
    
    for subset_id in subsets:
        print(f"\n{'#'*60}")
        print(f"  PROCESSING SUBSET: {subset_id}")
        print(f"{'#'*60}")
        
        # 1. Prep Data & Features
        data = prepare_subset(subset_id, verbose=False)
        feat_result = process_subset_features(data, subset_id, save=True, verbose=True)
        
        feature_cols = feat_result["feature_cols"]
        train_feat = feat_result["train_feat"]
        val_feat = feat_result["val_feat"]
        test_feat = feat_result["test_feat"]
        y_test_rul = feat_result["y_test_rul"]
        y_test_anomaly = feat_result["y_test_anomaly"]
        
        # 2. Train Models
        print(f"--- Training Models for {subset_id} ---")
        
        # RF
        rf_results = train_and_evaluate_rf(
            train_feat, val_feat, test_feat, feature_cols, subset_id
        )
        rf_metrics = rf_results["metrics"]
        rf_metrics["subset"] = subset_id
        all_benchmarks.append(rf_metrics)
        
        # LSTM
        lstm_results = train_and_evaluate_lstm(feat_result, subset_id, epochs=50)
        lstm_metrics = lstm_results["metrics"]
        lstm_metrics["subset"] = subset_id
        all_benchmarks.append(lstm_metrics)
        
        # Autoencoder
        ae_results = train_and_evaluate_autoencoder(
            train_feat, val_feat, test_feat, feature_cols, subset_id
        )
        ae_metrics = ae_results["metrics"]
        ae_metrics["subset"] = subset_id
        all_benchmarks.append(ae_metrics)
        
        # Ensemble (Weighted LSTM + RF)
        ens_metrics = build_ensemble(rf_results, lstm_results, test_feat, y_test_rul, subset_id)
        ens_metrics["subset"] = subset_id
        all_benchmarks.append(ens_metrics)
        
    # 3. Save Master Benchmark Report
    os.makedirs(METRICS_DIR, exist_ok=True)
    master_df = pd.DataFrame(all_benchmarks)
    
    # Clean up display: Autoencoder N/A values
    for col in ['rmse', 'mae', 'nasa_score']:
        master_df.loc[master_df['model'] == 'Autoencoder', col] = "N/A"
        
    master_df.to_csv(os.path.join(METRICS_DIR, "benchmark_metrics.csv"), index=False)
    print(f"\nMaster Benchmark saved to: {METRICS_DIR}/benchmark_metrics.csv")
    print(master_df.to_string(index=False))

if __name__ == "__main__":
    main()
