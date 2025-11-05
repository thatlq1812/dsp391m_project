import pandas as pd
import numpy as np
import sys

# Read training history
df = pd.read_csv('outputs/stmgt_v2_20251102_200308/training_history.csv')

print('='*80)
print('TRAINING ANALYSIS - Run 20251102_200308')
print('='*80)
print(f'\nCurrent Epoch: {len(df)}/? (still running)')
print(f'Latest Epoch: {df.iloc[-1]["epoch"]:.0f}')

print('\n' + '='*80)
print('LATEST METRICS (Epoch {})'.format(int(df.iloc[-1]['epoch'])))
print('='*80)
latest = df.iloc[-1]
print(f'Train MAE:  {latest["train_mae"]:.4f} km/h')
print(f'Train RMSE: {latest["train_rmse"]:.4f} km/h')
print(f'Train R²:   {latest["train_r2"]:.4f}')
print(f'Train Coverage (80%): {latest["train_coverage_80"]:.2%}')
print()
print(f'Val MAE:    {latest["val_mae"]:.4f} km/h')
print(f'Val RMSE:   {latest["val_rmse"]:.4f} km/h')
print(f'Val R²:     {latest["val_r2"]:.4f}')
print(f'Val Coverage (80%): {latest["val_coverage_80"]:.2%}')

print('\n' + '='*80)
print('BEST PERFORMANCE SO FAR')
print('='*80)
best_idx = df['val_mae'].idxmin()
best = df.iloc[best_idx]
print(f'Best Epoch: {int(best["epoch"])}')
print(f'Val MAE:    {best["val_mae"]:.4f} km/h')
print(f'Val RMSE:   {best["val_rmse"]:.4f} km/h')
print(f'Val R²:     {best["val_r2"]:.4f}')

print('\n' + '='*80)
print('CONVERGENCE ANALYSIS')
print('='*80)
first_5 = df.iloc[:5]['val_mae'].mean()
last_5 = df.iloc[-5:]['val_mae'].mean()
improvement = ((first_5 - last_5) / first_5) * 100
print(f'First 5 epochs avg MAE: {first_5:.4f} km/h')
print(f'Last 5 epochs avg MAE:  {last_5:.4f} km/h')
print(f'Improvement: {improvement:.2f}%')

mae_std_last5 = df.iloc[-5:]['val_mae'].std()
print(f'Last 5 epochs MAE std:  {mae_std_last5:.4f} (stability indicator)')

print('\n' + '='*80)
print('OVERFITTING CHECK')
print('='*80)
gap = latest['val_mae'] - latest['train_mae']
gap_pct = (gap / latest['val_mae']) * 100
print(f'Train/Val Gap: {gap:.4f} km/h ({gap_pct:.2f}%)')
if gap < 1.0:
    print('✅ Excellent generalization (gap < 1.0)')
elif gap < 2.0:
    print('✅ Good generalization (gap < 2.0)')
else:
    print('⚠️  Possible overfitting (gap >= 2.0)')

print('\n' + '='*80)
print('CALIBRATION QUALITY')
print('='*80)
target_cov = 0.80
actual_cov = latest['val_coverage_80']
cov_error = abs(actual_cov - target_cov)
print(f'Target 80% CI Coverage: 80.0%')
print(f'Actual Coverage:        {actual_cov:.2%}')
print(f'Calibration Error:      {cov_error:.2%}')
if cov_error < 0.05:
    print('✅ Well-calibrated (error < 5%)')
elif cov_error < 0.10:
    print('✅ Acceptable calibration (error < 10%)')
else:
    print('⚠️  Poor calibration (error >= 10%)')

print('\n' + '='*80)
print('REALISTIC PERFORMANCE CHECK')
print('='*80)
print('Comparison with known baselines:')
print('  - Naive persistence: ~7.2 km/h MAE')
print('  - Previous best STMGT: 3.91 km/h MAE (Exp 20251102_182710)')
print('  - Graph WaveNet claim: 1.55 km/h MAE (UNREALISTIC)')
print()
print(f'Current run: {latest["val_mae"]:.2f} km/h')
if latest['val_mae'] < 3.0:
    print('⚠️  SUSPICIOUS: Too good, verify data leakage')
elif latest['val_mae'] < 5.0:
    print('✅ EXCELLENT: Realistic and competitive')
elif latest['val_mae'] < 7.0:
    print('✅ GOOD: Reasonable performance')
else:
    print('⚠️  WEAK: Below naive baseline')

print('\n' + '='*80)
print('PREDICTION')
print('='*80)
recent_trend = df.iloc[-3:]['val_mae'].values
print(f'Last 3 epochs MAE: {recent_trend}')
if len(recent_trend) >= 3:
    if recent_trend[-1] < recent_trend[-2] < recent_trend[-3]:
        print('📉 Still improving - likely to continue')
    elif recent_trend[-1] > recent_trend[-2]:
        print('📈 Starting to plateau/degrade')
    else:
        print('➡️  Stable/fluctuating')

estimated_final = df['val_mae'].min() * 0.95  # optimistic 5% more improvement
print(f'\nEstimated final MAE: {estimated_final:.2f}-{df["val_mae"].min():.2f} km/h')
print('='*80)
