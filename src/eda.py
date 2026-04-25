"""
eda.py — Exploratory Data Analysis for CMAPSS Turbofan Dataset
================================================================

Generates publication-quality visualizations for understanding
engine degradation patterns, sensor behaviors, and data distributions.

All plots saved to results/plots/ as 150 DPI PNGs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle

# Style configuration
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
PLOT_DPI = 150
FIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "plots",
)

COLORS = {
    "primary": "#2563EB",
    "danger": "#DC2626",
    "warning": "#F59E0B",
    "success": "#10B981",
    "info": "#8B5CF6",
    "muted": "#6B7280",
    "bg_danger": "#FEE2E2",
}


def _save_fig(fig, name):
    """Save figure to results/plots/ with consistent settings."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ──────────────────────────────────────────────────────────────
# 1. Sensor Degradation Curves
# ──────────────────────────────────────────────────────────────

def plot_sensor_degradation(df, subset_id, n_engines=5, sensors=None):
    """
    Plot sensor readings over engine lifecycle for representative engines.

    Shows clear degradation trends from stable operation to failure.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with sensor columns, unit_id, cycle, RUL.
    subset_id : str
        Dataset identifier for title.
    n_engines : int
        Number of engines to plot.
    sensors : list of str or None
        Specific sensors to plot (default: auto-select 4 most varying).
    """
    if sensors is None:
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        # Pick sensors with highest variance
        variances = df[sensor_cols].var().sort_values(ascending=False)
        sensors = variances.index[:4].tolist()

    engine_ids = df["unit_id"].unique()
    # Select engines with diverse lifespans
    max_cycles = df.groupby("unit_id")["cycle"].max()
    sorted_engines = max_cycles.sort_values().index
    step = max(1, len(sorted_engines) // n_engines)
    selected = sorted_engines[::step][:n_engines]

    fig, axes = plt.subplots(len(sensors), 1, figsize=(14, 3.5 * len(sensors)),
                             sharex=False)
    if len(sensors) == 1:
        axes = [axes]

    for idx, sensor in enumerate(sensors):
        ax = axes[idx]
        for i, uid in enumerate(selected):
            engine = df[df["unit_id"] == uid].sort_values("cycle")
            # Normalize cycle to percentage of life
            max_c = engine["cycle"].max()
            pct = engine["cycle"] / max_c * 100
            ax.plot(pct, engine[sensor], alpha=0.8,
                    label=f"Engine {uid} ({max_c} cycles)",
                    linewidth=1.5)

        ax.set_ylabel(sensor.replace("_", " ").title(), fontsize=11)
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.axvspan(80, 100, alpha=0.1, color=COLORS["danger"],
                   label="Degradation zone")

    axes[-1].set_xlabel("Engine Life (%)", fontsize=12)
    fig.suptitle(f"Sensor Degradation Curves — {subset_id}",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, f"sensor_degradation_{subset_id}")


# ──────────────────────────────────────────────────────────────
# 2. Correlation Heatmap
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df, subset_id):
    """
    Heatmap showing Pearson correlation of sensors with RUL.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with sensor columns and 'RUL'.
    subset_id : str
        Dataset identifier for title.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    cols_with_rul = sensor_cols + ["RUL", "health_index"]
    cols_present = [c for c in cols_with_rul if c in df.columns]

    corr_matrix = df[cols_present].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
    )
    ax.set_title(f"Sensor Correlation Heatmap — {subset_id}",
                 fontsize=14, fontweight="bold", pad=15)

    fig.tight_layout()
    _save_fig(fig, f"correlation_heatmap_{subset_id}")


# ──────────────────────────────────────────────────────────────
# 3. Health Index Decay Curve
# ──────────────────────────────────────────────────────────────

def plot_health_index_decay(df, subset_id):
    """
    Average health index vs normalized lifecycle for all engines.

    Should show clear sigmoid-like degradation pattern.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'health_index', 'unit_id', 'cycle' columns.
    subset_id : str
        Dataset identifier.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Normalize cycle to percentage of engine life
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].transform("max")
    df["life_pct"] = df["cycle"] / max_cycles * 100

    # Bin into percentiles and compute mean + std
    df["life_bin"] = pd.cut(df["life_pct"], bins=50, labels=False)
    stats = df.groupby("life_bin")["health_index"].agg(["mean", "std"])
    x = np.linspace(0, 100, len(stats))

    ax.fill_between(x, stats["mean"] - stats["std"],
                    stats["mean"] + stats["std"],
                    alpha=0.2, color=COLORS["primary"])
    ax.plot(x, stats["mean"], linewidth=2.5, color=COLORS["primary"],
            label="Mean Health Index")

    # Mark critical zones
    ax.axhline(y=0.3, color=COLORS["danger"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Critical threshold (0.3)")
    ax.axhspan(0, 0.3, alpha=0.08, color=COLORS["danger"])

    ax.set_xlabel("Engine Life (%)", fontsize=12)
    ax.set_ylabel("Health Index", fontsize=12)
    ax.set_title(f"Health Index Decay — {subset_id}",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_fig(fig, f"health_index_decay_{subset_id}")


# ──────────────────────────────────────────────────────────────
# 4. Operating Condition Clusters
# ──────────────────────────────────────────────────────────────

def plot_operating_clusters(df, subset_id):
    """
    Scatter plot of op_setting_1 vs op_setting_2, colored by cluster.

    Only meaningful for FD002/FD004 with multiple operating conditions.

    Parameters
    ----------
    df : pd.DataFrame
        Data with op_setting columns and 'op_cluster'.
    subset_id : str
        Dataset identifier.
    """
    if "op_cluster" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    n_clusters = df["op_cluster"].nunique()

    scatter = ax.scatter(
        df["op_setting_1"], df["op_setting_2"],
        c=df["op_cluster"], cmap="Set2" if n_clusters > 1 else "Blues",
        alpha=0.3, s=10, edgecolors="none",
    )

    if n_clusters > 1:
        cbar = plt.colorbar(scatter, ax=ax, label="Cluster ID")
        cbar.set_ticks(range(n_clusters))

    ax.set_xlabel("Operational Setting 1", fontsize=12)
    ax.set_ylabel("Operational Setting 2", fontsize=12)
    ax.set_title(
        f"Operating Condition Clusters — {subset_id} ({n_clusters} clusters)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    _save_fig(fig, f"operating_clusters_{subset_id}")


# ──────────────────────────────────────────────────────────────
# 5. RUL Distribution
# ──────────────────────────────────────────────────────────────

def plot_rul_distribution(df, subset_id, cap=125):
    """
    Histogram of RUL values in training data with KDE overlay.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with 'RUL' column.
    subset_id : str
        Dataset identifier.
    cap : int
        RUL cap value to annotate.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(df["RUL"], bins=60, density=True, alpha=0.7,
            color=COLORS["primary"], edgecolor="white", linewidth=0.5,
            label="RUL distribution")

    # KDE overlay
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(df["RUL"].values)
    x_range = np.linspace(0, df["RUL"].max() + 5, 200)
    ax.plot(x_range, kde(x_range), linewidth=2.5,
            color=COLORS["info"], label="KDE")

    # Annotate cap and anomaly threshold
    ax.axvline(x=cap, color=COLORS["warning"], linestyle="--",
               linewidth=2, label=f"RUL cap ({cap})")
    ax.axvline(x=30, color=COLORS["danger"], linestyle="--",
               linewidth=2, label="Anomaly zone (RUL < 30)")
    ax.axvspan(0, 30, alpha=0.1, color=COLORS["danger"])

    ax.set_xlabel("Remaining Useful Life (cycles)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"RUL Distribution — {subset_id}",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    _save_fig(fig, f"rul_distribution_{subset_id}")


# ──────────────────────────────────────────────────────────────
# 6. Anomaly Zone Timeline
# ──────────────────────────────────────────────────────────────

def plot_anomaly_timeline(df, subset_id, engine_id=None):
    """
    Timeline of one engine showing when it enters the danger zone.

    Overlays a key sensor trend with shaded anomaly region.

    Parameters
    ----------
    df : pd.DataFrame
        Training data with sensor columns, RUL, anomaly.
    subset_id : str
        Dataset identifier.
    engine_id : int or None
        Specific engine to plot. If None, picks one with ~200 cycles.
    """
    if engine_id is None:
        max_cycles = df.groupby("unit_id")["cycle"].max()
        # Pick engine closest to median lifespan
        median_life = max_cycles.median()
        engine_id = (max_cycles - median_life).abs().idxmin()

    engine = df[df["unit_id"] == engine_id].sort_values("cycle")
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # Pick the sensor most correlated with RUL for this engine
    if len(sensor_cols) > 0:
        corrs = engine[sensor_cols].corrwith(engine["RUL"]).abs()
        best_sensor = corrs.idxmax()
    else:
        best_sensor = sensor_cols[0] if sensor_cols else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    sharex=True, height_ratios=[2, 1])

    # Top: sensor trend with anomaly zone
    if best_sensor:
        ax1.plot(engine["cycle"], engine[best_sensor],
                 linewidth=2, color=COLORS["primary"],
                 label=best_sensor.replace("_", " ").title())

    # Shade anomaly zone
    anomaly_start = engine[engine["RUL"] < 30]["cycle"].min()
    if not pd.isna(anomaly_start):
        ax1.axvspan(anomaly_start, engine["cycle"].max(),
                    alpha=0.15, color=COLORS["danger"],
                    label="Danger zone (RUL < 30)")
        ax1.axvline(x=anomaly_start, color=COLORS["danger"],
                    linestyle="--", linewidth=1.5)

    ax1.set_ylabel("Sensor Reading", fontsize=12)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_title(
        f"Anomaly Zone Timeline — Engine {engine_id} ({subset_id})",
        fontsize=14, fontweight="bold",
    )

    # Bottom: RUL countdown
    ax2.fill_between(engine["cycle"], engine["RUL"],
                     alpha=0.3, color=COLORS["primary"])
    ax2.plot(engine["cycle"], engine["RUL"],
             linewidth=2, color=COLORS["primary"])
    ax2.axhline(y=30, color=COLORS["danger"], linestyle="--",
                linewidth=1.5, label="Anomaly threshold")
    ax2.set_xlabel("Cycle", fontsize=12)
    ax2.set_ylabel("RUL (cycles)", fontsize=12)
    ax2.legend(fontsize=11)

    fig.tight_layout()
    _save_fig(fig, f"anomaly_timeline_{subset_id}")


# ──────────────────────────────────────────────────────────────
# Master EDA Pipeline
# ──────────────────────────────────────────────────────────────

def run_eda(data_dict, subset_id):
    """
    Generate all 6 EDA visualizations for one CMAPSS subset.

    Parameters
    ----------
    data_dict : dict
        Output from data_prep.prepare_subset() with 'train' key.
    subset_id : str
        Dataset identifier.
    """
    train_df = data_dict["train"]
    print(f"\n  Generating EDA plots for {subset_id}...")

    plot_sensor_degradation(train_df, subset_id)
    plot_correlation_heatmap(train_df, subset_id)
    plot_health_index_decay(train_df, subset_id)
    plot_operating_clusters(train_df, subset_id)
    plot_rul_distribution(train_df, subset_id)
    plot_anomaly_timeline(train_df, subset_id)

    print(f"  EDA complete for {subset_id}")


def run_all_eda(all_data):
    """
    Generate EDA for all 4 CMAPSS subsets.

    Parameters
    ----------
    all_data : dict
        Mapping of subset_id → data_dict from prepare_all_subsets().
    """
    for subset_id in ["FD001", "FD002", "FD003", "FD004"]:
        run_eda(all_data[subset_id], subset_id)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_prep import prepare_all_subsets

    print("Loading and preparing data...")
    all_data = prepare_all_subsets(save=False, verbose=True)

    print("\nGenerating EDA visualizations...")
    run_all_eda(all_data)
    print("\nAll EDA visualizations saved to results/plots/")
