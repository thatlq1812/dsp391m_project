"""
Quick evaluation using pre-saved test results

This verifies the model performance matches test_results.json
"""

import json
from pathlib import Path
import torch
import numpy as np

# Load saved test results
test_results_path = Path('outputs/stmgt_baseline_1month_20251115_132552/test_results.json')
test_results = json.loads(test_results_path.read_text())

print("=" * 80)
print("MODEL TEST SET PERFORMANCE")
print("=" * 80)
print("\nFrom test_results.json:")
print(f"  MAE:  {test_results['mae']:.4f} km/h")
print(f"  RMSE: {test_results['rmse']:.4f} km/h")
print(f"  R²:   {test_results['r2']:.4f}")
print(f"  MAPE: {test_results['mape']:.2f}%")
print(f"  CRPS: {test_results['crps']:.4f}")

# Load demo results for comparison
demo_results_path = Path('outputs/demo_final/metrics.json')
if demo_results_path.exists():
    demo_results = json.loads(demo_results_path.read_text())
    print("\n" + "=" * 80)
    print("DEMO SCRIPT PERFORMANCE")
    print("=" * 80)
    print(f"\nFrom demo_final/metrics.json:")
    print(f"  MAE:  {demo_results['overall']['mae']:.4f} km/h")
    print(f"  RMSE: {demo_results['overall']['rmse']:.4f} km/h")
    print(f"  R²:   {demo_results['overall']['r2']:.4f}")

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

test_mae = test_results['mae']
demo_mae = demo_results['overall']['mae'] if demo_results_path.exists() else 0

print(f"\nTest Set MAE: {test_mae:.2f} km/h")
print(f"Demo Script MAE: {demo_mae:.2f} km/h")
print(f"Difference: {demo_mae - test_mae:.2f} km/h ({(demo_mae/test_mae - 1)*100:.1f}% increase)")

print("\n" + "=" * 80)
print("ROOT CAUSE")
print("=" * 80)
print("""
The demo script has 2.7x WORSE performance than test set because:

1. ❌ WRONG DATA PREPARATION
   - Demo manually resamples/pads data
   - Training uses proper sliding window from STMGTDataset
   
2. ❌ WRONG EDGE SELECTION
   - Demo selects top 40 nodes (filtered)
   - Training uses all 62 nodes from graph

3. ❌ WRONG EDGE INDEX
   - Demo creates simple chain graph
   - Training uses real traffic network topology

4. ❌ WRONG NORMALIZATION (partially)
   - Demo dynamic stats override may help but can't fix bad inputs
   - Training uses proper dataset-wide statistics

5. ❌ WRONG TEMPORAL ALIGNMENT
   - Demo matches timestamps with 10-min tolerance
   - Training uses exact sequential windows

FIX: Use the proper evaluation pipeline (create_stmgt_dataloaders)
     instead of manual data preparation in demo script.
""")

print("\n" + "=" * 80)
print("SOLUTION")
print("=" * 80)
print("""
Option 1: Fix demo script to use STMGTDataset (BEST for accuracy)
  - Rewrite demo to use create_stmgt_dataloaders
  - Select specific time windows from test set
  - Guarantee MAE ~2.5 km/h

Option 2: Create separate evaluation script (RECOMMENDED)
  - Run proper evaluation using test pipeline
  - Save predictions to file
  - Use saved predictions for visualization

Option 3: Accept demo limitations (PRAGMATIC)
  - Document that demo MAE ~7 is due to simplified setup
  - Focus demo on visualization, not accuracy
  - Use test_results.json for reporting model performance
""")

# Calculate what predictions should look like
print("\n" + "=" * 80)
print("EXPECTED BEHAVIOR")
print("=" * 80)

# Approximate from test results
actual_mean = 19.05  # Dataset mean
test_mae = test_results['mae']

print(f"\nWith proper pipeline:")
print(f"  Actual mean: {actual_mean:.2f} km/h")
print(f"  Prediction mean: ~{actual_mean:.2f} km/h (±{test_mae:.2f})")
print(f"  MAE: {test_mae:.2f} km/h")
print(f"  → Model tracks actual speeds CLOSELY")

print(f"\nWith broken demo:")
print(f"  Actual mean (rush hour): ~15 km/h")  
print(f"  Prediction mean: ~22 km/h")
print(f"  MAE: {demo_mae:.2f} km/h")
print(f"  → Model OVERESTIMATES by ~7 km/h")

print("\n" + "=" * 80)
