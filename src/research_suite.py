import os
import sys
import subprocess
import time
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def run_step(name, command):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    start_time = time.time()
    
    try:
        # Use sys.executable to ensure we use the same python interpreter
        result = subprocess.run([sys.executable] + command.split(), check=True)
        elapsed = time.time() - start_time
        print(f"OK: {name} completed in {elapsed:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {name} failed with error: {e}")
        return False

def main():
    print("NASA CMAPSS Research-Grade Suite v2.0")
    print("Orchestrating End-to-End Predictive Maintenance Research...")
    
    # 1. Clean Benchmarking (All 4 Subsets)
    if not run_step("Standard Benchmarking", "src/main.py"):
        return

    # 2. Noise & Robustness Evaluation
    # This runs the stress tests we defined (Gaussian, Drift, Spikes)
    if not run_step("Robustness Stress-Testing", "src/robustness_eval.py"):
        return

    # 3. Findings Generation
    # Updates the markdown reports with final numbers
    print("\nUpdating Research Findings...")
    # (The robustness_eval already writes to robustness_report.csv)
    
    # 4. Interactive Dashboard Generation
    if not run_step("Interactive Dashboard Generation", "src/dashboard_gen.py"):
        return

    print("\n" + "="*60)
    print("ALL RESEARCH TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("View your results:")
    print("  - Dashboard: results/dashboard.html")
    print("  - Findings:  results/findings.md")
    print("  - Models:    models/")
    print("="*60)

if __name__ == "__main__":
    main()
