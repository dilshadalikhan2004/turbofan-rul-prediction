"""
visualize.py — Publication-Quality Visualizations for Model Results
====================================================================

Generates 5 key visualizations saved to results/plots/ at 150 DPI:
1. RUL prediction curves (actual vs predicted per model)
2. Anomaly heatmap (engines × cycles, colored by reconstruction error)
3. Early warning timeline (alarm timing vs actual failure)
4. Model comparison bar chart (RMSE, NASA Score, F1)
5. Reconstruction error curve with inflection point
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")
PLOT_DPI = 150

COLORS = {
    "rf": "#2563EB", "lstm": "#8B5CF6",
    "ensemble": "#10B981", "autoencoder": "#F59E0B",
    "actual": "#1F2937", "danger": "#DC2626",
}


def _save(fig, name):
    os.makedirs(PLOT_DIR, exist_ok=True)
    p = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(p, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {p}")


def plot_rul_predictions(test_feat, predictions, subset_id, n_engines=5):
    """Plot actual vs predicted RUL for multiple engines per model."""
    max_c = test_feat.groupby("unit_id")["cycle"].max()
    sel = max_c.sort_values().index
    step = max(1, len(sel) // n_engines)
    engines = sel[::step][:n_engines]

    n_models = len(predictions)
    fig, axes = plt.subplots(n_engines, 1, figsize=(14, 3.5*n_engines), sharex=False)
    if n_engines == 1: axes = [axes]

    for i, uid in enumerate(engines):
        ax = axes[i]
        eng = test_feat[test_feat["unit_id"]==uid].sort_values("cycle")
        ax.plot(eng["cycle"], eng["RUL"], color=COLORS["actual"],
                linewidth=2.5, label="Actual RUL", linestyle="-")
        for mname, pred in predictions.items():
            eng_pred = pred[test_feat["unit_id"]==uid]
            if len(eng_pred) == len(eng):
                ax.plot(eng["cycle"].values, eng_pred, linewidth=1.8,
                        label=mname, alpha=0.85,
                        color=COLORS.get(mname.lower().split()[0], "#6B7280"))
        ax.set_ylabel("RUL", fontsize=11)
        ax.set_title(f"Engine {uid}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=n_models+1, loc="upper right")

    axes[-1].set_xlabel("Cycle", fontsize=12)
    fig.suptitle(f"RUL Predictions — {subset_id}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"rul_predictions_{subset_id}")


def plot_anomaly_heatmap(anomaly_df, subset_id, max_engines=50):
    """Heatmap of engines × cycles colored by reconstruction error."""
    uids = anomaly_df["unit_id"].unique()[:max_engines]
    fig, ax = plt.subplots(figsize=(16, max(6, len(uids)*0.25)))
    max_cycle = anomaly_df["cycle"].max()
    heatmap_data = np.full((len(uids), max_cycle), np.nan)

    for i, uid in enumerate(uids):
        eng = anomaly_df[anomaly_df["unit_id"]==uid]
        for _, row in eng.iterrows():
            c = int(row["cycle"]) - 1
            if c < max_cycle:
                heatmap_data[i, c] = row["reconstruction_error"]

    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Engine", fontsize=12)
    ax.set_yticks(range(0, len(uids), max(1, len(uids)//10)))
    ax.set_yticklabels([str(uids[i]) for i in range(0, len(uids), max(1, len(uids)//10))])
    plt.colorbar(im, ax=ax, label="Reconstruction Error", shrink=0.8)
    ax.set_title(f"Anomaly Heatmap — {subset_id}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, f"anomaly_heatmap_{subset_id}")


def plot_early_warning_timeline(test_feat, model_alerts, subset_id, n_engines=3):
    """Show when each model raised alarm vs actual failure for selected engines."""
    max_c = test_feat.groupby("unit_id")["cycle"].max()
    sel = max_c.sort_values().index
    step = max(1, len(sel) // n_engines)
    engines = sel[::step][:n_engines]

    fig, axes = plt.subplots(n_engines, 1, figsize=(14, 3*n_engines))
    if n_engines == 1: axes = [axes]

    for i, uid in enumerate(engines):
        ax = axes[i]
        eng = test_feat[test_feat["unit_id"]==uid].sort_values("cycle")
        total = eng["cycle"].max()

        ax.barh(0, total, height=0.4, color="#E5E7EB", edgecolor="none")
        fail_start = eng[eng["RUL"]<30]["cycle"].min()
        if not pd.isna(fail_start):
            ax.barh(0, total-fail_start, left=fail_start, height=0.4,
                    color="#FEE2E2", edgecolor="#DC2626", linewidth=1)
            ax.axvline(x=fail_start, color="#DC2626", linewidth=2, linestyle="--")

        y_offset = 0.6
        for mname, alert_cycle in model_alerts.get(uid, {}).items():
            color = COLORS.get(mname.lower().split()[0], "#6B7280")
            if alert_cycle:
                ax.scatter([alert_cycle], [y_offset], marker="v", s=100,
                          color=color, zorder=5)
                ax.annotate(f"{mname}: {alert_cycle}", (alert_cycle, y_offset),
                           fontsize=8, ha="center", va="bottom")
            y_offset += 0.15

        ax.set_title(f"Engine {uid}", fontsize=12, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, total+5)

    axes[-1].set_xlabel("Cycle", fontsize=12)
    fig.suptitle(f"Early Warning Timeline — {subset_id}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, f"early_warning_timeline_{subset_id}")


def plot_model_comparison(metrics_df, subset_id=None):
    """Bar chart comparing RMSE, NASA Score, F1 across models."""
    if subset_id:
        df = metrics_df[metrics_df["subset"]==subset_id]
    else:
        df = metrics_df.groupby("model").mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    models = df["model"].values

    for ax, metric, title in zip(
        axes,
        ["rmse", "nasa_score", "f1"],
        ["RMSE (lower=better)", "NASA Score (lower=better)", "F1 Score (higher=better)"]
    ):
        # Filter out N/A values for this specific metric
        valid_mask = df[metric].astype(str) != "N/A"
        valid_df = df[valid_mask]
        
        vals = pd.to_numeric(valid_df[metric]).values
        valid_models = valid_df["model"].values
        
        colors = [COLORS.get(m.lower().split()[0], "#6B7280") for m in valid_models]
        bars = ax.bar(range(len(valid_models)), vals, color=colors, edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(len(valid_models)))
        ax.set_xticklabels(valid_models, rotation=15, fontsize=10)
        ax.set_title(title, fontsize=13, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2., bar.get_height()+0.01*max(vals),
                   f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    label = subset_id or "All Subsets (Average)"
    fig.suptitle(f"Model Comparison — {label}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    name = f"model_comparison_{subset_id}" if subset_id else "model_comparison_all"
    _save(fig, name)


def plot_reconstruction_error_detail(anomaly_df, subset_id, threshold):
    """Detailed reconstruction error for one engine with threshold and inflection."""
    max_c = anomaly_df.groupby("unit_id")["cycle"].max()
    uid = max_c.sort_values().index[len(max_c)//2]
    eng = anomaly_df[anomaly_df["unit_id"]==uid].sort_values("cycle")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(eng["cycle"], eng["reconstruction_error"],
            linewidth=2, color=COLORS["rf"], label="Reconstruction Error")
    ax.axhline(y=threshold, color=COLORS["danger"], linestyle="--",
               linewidth=2, label=f"Threshold ({threshold:.5f})")

    cross_idx = eng[eng["reconstruction_error"]>threshold]["cycle"]
    if len(cross_idx)>0:
        inflection = cross_idx.iloc[0]
        ax.axvline(x=inflection, color=COLORS["autoencoder"], linestyle=":",
                   linewidth=2, label=f"Inflection: cycle {inflection}")
        ax.annotate("Anomaly begins", xy=(inflection, threshold),
                   xytext=(inflection-20, threshold*1.5),
                   arrowprops=dict(arrowstyle="->", color=COLORS["autoencoder"]),
                   fontsize=12, fontweight="bold", color=COLORS["autoencoder"])

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Reconstruction Error", fontsize=12)
    ax.set_title(f"Reconstruction Error Detail — Engine {uid} ({subset_id})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save(fig, f"recon_error_detail_{subset_id}")


def generate_all_visualizations(metrics_csv_path=None):
    """Generate all visualizations from saved metrics/results."""
    if metrics_csv_path is None:
        metrics_csv_path = os.path.join(METRICS_DIR, "benchmark_metrics.csv")
    if os.path.exists(metrics_csv_path):
        metrics_df = pd.read_csv(metrics_csv_path)
        for sid in metrics_df["subset"].unique():
            plot_model_comparison(metrics_df, subset_id=sid)
        plot_model_comparison(metrics_df, subset_id=None)
    print("  Visualization generation complete.")
