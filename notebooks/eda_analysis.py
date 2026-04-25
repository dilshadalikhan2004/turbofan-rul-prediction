"""
eda_analysis.py — Full Exploratory Data Analysis for CMAPSS Dataset
====================================================================

Run this script to generate all EDA visualizations:
    python notebooks/eda_analysis.py

Or convert to Jupyter notebook:
    pip install jupytext
    jupytext --to notebook notebooks/eda_analysis.py
"""

# %% [markdown]
# # NASA CMAPSS Turbofan Engine Degradation — Exploratory Data Analysis
#
# This analysis explores sensor degradation patterns, operating conditions,
# and health index dynamics across all 4 CMAPSS subsets.

# %%
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from data_prep import prepare_all_subsets
from eda import run_all_eda

# %% [markdown]
# ## Step 1: Load and Prepare All Datasets

# %%
print("Loading all 4 CMAPSS subsets...")
all_data = prepare_all_subsets(save=False, verbose=True)

# %% [markdown]
# ## Step 2: Generate All EDA Visualizations

# %%
print("\nGenerating visualizations...")
run_all_eda(all_data)
print("\nAll EDA plots saved to results/plots/")

# %% [markdown]
# ## Step 3: Data Summary Statistics

# %%
import pandas as pd

for sid in ["FD001", "FD002", "FD003", "FD004"]:
    train = all_data[sid]["train"]
    print(f"\n{'='*50}")
    print(f"  {sid} Summary")
    print(f"{'='*50}")
    print(f"  Engines: {train['unit_id'].nunique()}")
    print(f"  Total cycles: {len(train):,}")
    print(f"  Avg lifecycle: {train.groupby('unit_id')['cycle'].max().mean():.0f} cycles")
    print(f"  RUL range: {train['RUL'].min()} - {train['RUL'].max()}")
    print(f"  Health index range: {train['health_index'].min():.3f} - {train['health_index'].max():.3f}")
    print(f"  Op clusters: {train['op_cluster'].nunique()}")
