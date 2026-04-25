"""
noise_injection.py — Simulates Real-World Sensor Degradation
============================================================

Aerospace sensors in the real world suffer from calibration drift, 
communication dropouts, random noise, and electrical spikes. This 
module injects these 4 specific types of noise into the CMAPSS 
dataset to test model robustness.
"""

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOT_DIR = os.path.join(BASE_DIR, "results", "plots")


class NoiseInjector:
    def __init__(self, noise_level='medium', g_mult=None, d_mult=None, s_mult=None, dr_mult=None, random_state=42):
        """
        Initialize the Noise Injector.
        
        Parameters
        ----------
        noise_level : str
            'low' (50%), 'medium' (100%), 'high' (200%)
        g_mult, d_mult, s_mult, dr_mult : float, optional
            Explicit multipliers for Gaussian, Dropout, Spikes, and Drift.
            If provided, they override the noise_level defaults.
        random_state : int
            Random seed for reproducibility.
        """
        self.noise_level = noise_level
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        random.seed(random_state)
        
        # Intensity multipliers
        levels = {'low': 0.5, 'medium': 1.0, 'high': 2.0, 'custom': 1.0}
        base_mult = levels.get(noise_level, 1.0)
            
        self.g_mult = g_mult if g_mult is not None else base_mult
        self.d_mult = d_mult if d_mult is not None else base_mult
        self.s_mult = s_mult if s_mult is not None else base_mult
        self.dr_mult = dr_mult if dr_mult is not None else base_mult
        
        # Base parameters
        self.base_gaussian_sigma = 0.02  # 2% of std
        self.base_dropout_rate = 0.10    # 10% dropout
        self.base_spike_rate = 0.02      # 2% spikes
        self.drift_sensors_per_engine = 3
        
        self.report = {}

    def inject(self, df, sensor_cols):
        """
        Apply all 4 noise types to the dataframe.
        
        Parameters
        ----------
        df : pd.DataFrame
            Clean dataframe.
        sensor_cols : list of str
            Columns to apply noise to.
            
        Returns
        -------
        pd.DataFrame
            Noisy dataframe.
        """
        noisy_df = df.copy()
        
        # Track spikes for later evaluation
        noisy_df['is_injected_spike'] = 0
        
        stats = {
            'gaussian_noise_applied': True,
            'dropout_events': 0,
            'drift_engines_affected': 0,
            'spikes_injected': 0
        }
        
        # 1. Gaussian Noise
        sigma_multiplier = self.base_gaussian_sigma * self.g_mult
        for col in sensor_cols:
            std = df[col].std()
            if std > 0:
                noise = self.rng.normal(0, std * sigma_multiplier, size=len(df))
                noisy_df[col] += noise

        # 2. Sudden Spike Anomalies
        spike_rate = self.base_spike_rate * self.s_mult
        for col in sensor_cols:
            std = df[col].std()
            if std > 0:
                # Random locations
                spike_mask = self.rng.random(len(df)) < spike_rate
                n_spikes = spike_mask.sum()
                
                if n_spikes > 0:
                    # Magnitude 3x to 5x std
                    magnitudes = self.rng.uniform(3, 5, size=n_spikes) * std * self.s_mult
                    # Random direction (positive or negative)
                    directions = self.rng.choice([-1, 1], size=n_spikes)
                    
                    noisy_df.loc[spike_mask, col] += (magnitudes * directions)
                    noisy_df.loc[spike_mask, 'is_injected_spike'] = 1
                    stats['spikes_injected'] += n_spikes

        # 3. Gradual Sensor Drift
        for uid in noisy_df['unit_id'].unique():
            engine_mask = noisy_df['unit_id'] == uid
            engine_idx = noisy_df[engine_mask].index
            
            # Select 3 random sensors for this engine
            drift_cols = self.rng.choice(sensor_cols, size=self.drift_sensors_per_engine, replace=False)
            stats['drift_engines_affected'] += 1
            
            for col in drift_cols:
                drift_rate = self.rng.uniform(0.001, 0.005) * self.dr_mult
                
                # Drift starts at cycle 20
                cycles = noisy_df.loc[engine_idx, 'cycle'].values
                drift_amount = np.maximum(0, cycles - 20) * drift_rate
                
                # Random direction
                direction = self.rng.choice([-1, 1])
                noisy_df.loc[engine_idx, col] += (drift_amount * direction)

        # 4. Random Sensor Dropout (NaN then forward fill)
        dropout_rate = self.base_dropout_rate * self.d_mult
        # Apply dropout to sensor columns
        mask = self.rng.random(noisy_df[sensor_cols].shape) < dropout_rate
        stats['dropout_events'] = mask.sum()
        
        # We need to operate on the numpy array to set NaNs quickly
        sensor_data = noisy_df[sensor_cols].values
        sensor_data[mask] = np.nan
        noisy_df[sensor_cols] = sensor_data
        
        # Forward fill (group by engine so we don't leak across engines)
        noisy_df[sensor_cols] = noisy_df.groupby('unit_id')[sensor_cols].ffill()
        # If the first value is NaN, bfill
        noisy_df[sensor_cols] = noisy_df.groupby('unit_id')[sensor_cols].bfill()
        
        self.report = stats
        return noisy_df

    def get_noise_report(self):
        return self.report
        
    def plot_comparison(self, clean_df, noisy_df, engine_id, sensor_col):
        """
        Plot clean vs noisy sensor readings for a single engine.
        """
        clean_eng = clean_df[clean_df['unit_id'] == engine_id].sort_values('cycle')
        noisy_eng = noisy_df[noisy_df['unit_id'] == engine_id].sort_values('cycle')
        
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(clean_eng['cycle'], clean_eng[sensor_col], 
                color='#1F2937', linewidth=2, label='Clean Signal')
        ax.plot(noisy_eng['cycle'], noisy_eng[sensor_col], 
                color='#EF4444', linewidth=1.5, alpha=0.7, label=f'Noisy Signal ({self.noise_level})')
                
        # Mark spikes if any
        spikes = noisy_eng[noisy_eng['is_injected_spike'] == 1]
        if not spikes.empty:
            ax.scatter(spikes['cycle'], spikes[sensor_col], 
                      color='#F59E0B', s=50, zorder=5, label='Injected Spikes')
                      
        ax.set_title(f"Sensor Noise Injection - {sensor_col} (Engine {engine_id})", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Sensor Value")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        os.makedirs(PLOT_DIR, exist_ok=True)
        path = os.path.join(PLOT_DIR, f"noise_comparison_{engine_id}_{sensor_col}_{self.noise_level}.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved noise comparison to: {path}")

if __name__ == "__main__":
    # Quick test
    from data_prep import load_dataset, drop_constant_sensors, RETAINED_SENSORS
    train_df, _, _ = load_dataset("FD001")
    train_df = drop_constant_sensors(train_df)
    sensor_cols = [c for c in RETAINED_SENSORS if c in train_df.columns]
    
    injector = NoiseInjector(noise_level='high')
    noisy_df = injector.inject(train_df, sensor_cols)
    print("Noise Report:", injector.get_noise_report())
    
    # Plot example
    injector.plot_comparison(train_df, noisy_df, engine_id=1, sensor_col=sensor_cols[0])
