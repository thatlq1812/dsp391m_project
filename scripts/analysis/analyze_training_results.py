"""
Analyze STMGT Training Results

Purpose:
    - Load training history
    - Visualize loss curves
    - Compute learning statistics
    - Generate analysis report

Author: DSP391m Team
Date: November 1, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
run_dir = Path('outputs/stmgt_20251101_002822')

with open(run_dir / 'history.json') as f:
    history = json.load(f)

with open(run_dir / 'config.json') as f:
    config = json.load(f)

train_loss = np.array(history['train_loss'])

# Analysis
print("=" * 70)
print("STMGT TRAINING ANALYSIS")
print("=" * 70)

print("\nCONFIGURATION:")
print(f"  Sequence Length: {config['seq_len']} timesteps (12 runs)")
print(f"  Prediction Length: {config['pred_len']} timesteps (12 runs)")
print(f"  Batch Size: {config['batch_size']}")
print(f"  Learning Rate: {config['learning_rate']}")
print(f"  Weight Decay: {config['weight_decay']}")
print(f"  DropEdge Probability: {config['drop_edge_p']}")
print(f"  Max Epochs: {config['max_epochs']}")

print("\nTRAINING SUMMARY:")
print(f"  Total Epochs: {len(train_loss)}")
print(f"  Initial Loss: {train_loss[0]:.4f}")
print(f"  Final Loss: {train_loss[-1]:.4f}")
print(f"  Best Loss: {train_loss.min():.4f} (Epoch {train_loss.argmin() + 1})")
print(f"  Loss Reduction: {train_loss[0] - train_loss[-1]:.4f} ({(1 - train_loss[-1]/train_loss[0])*100:.1f}%)")

# Learning phases
early_phase = train_loss[:20]
mid_phase = train_loss[20:60]
late_phase = train_loss[60:]

print("\nLEARNING PHASES:")
print(f"  Early (1-20): {early_phase.mean():.4f} ± {early_phase.std():.4f}")
print(f"  Mid (21-60): {mid_phase.mean():.4f} ± {mid_phase.std():.4f}")
print(f"  Late (61-100): {late_phase.mean():.4f} ± {late_phase.std():.4f}")

# Convergence analysis
last_10 = train_loss[-10:]
print(f"\nCONVERGENCE (Last 10 epochs):")
print(f"  Mean Loss: {last_10.mean():.4f}")
print(f"  Std Dev: {last_10.std():.4f}")
print(f"  Min: {last_10.min():.4f}")
print(f"  Max: {last_10.max():.4f}")
print(f"  Variance: {last_10.std()**2:.6f}")

# Learning rate analysis
epoch_diff = np.diff(train_loss)
print(f"\nLEARNING DYNAMICS:")
print(f"  Improving Epochs: {(epoch_diff < 0).sum()} / {len(epoch_diff)} ({(epoch_diff < 0).sum()/len(epoch_diff)*100:.1f}%)")
print(f"  Average Improvement: {epoch_diff[epoch_diff < 0].mean():.4f} per epoch")
print(f"  Largest Jump: {epoch_diff.max():.4f} (Epoch {epoch_diff.argmax() + 1} -> {epoch_diff.argmax() + 2})")
print(f"  Largest Drop: {epoch_diff.min():.4f} (Epoch {epoch_diff.argmin() + 1} -> {epoch_diff.argmin() + 2})")

# Data limitations
print("\nDATA LIMITATIONS:")
print(f"  Training Samples: 3 (very limited!)")
print(f"  Validation Samples: 0 (no validation set)")
print(f"  Test Samples: 0 (no test set)")
print(f"  WARNING: Results may indicate overfitting due to tiny dataset")

print("\nKEY OBSERVATIONS:")

# Check convergence
if last_10.std() < 0.02:
    print("  ✓ Model has converged (stable loss in last 10 epochs)")
else:
    print("  ! Model still fluctuating (high variance in last 10 epochs)")

# Check improvement
total_improvement = (train_loss[0] - train_loss[-1]) / train_loss[0]
if total_improvement > 0.95:
    print(f"  ✓ Excellent loss reduction: {total_improvement*100:.1f}%")
elif total_improvement > 0.80:
    print(f"  ✓ Good loss reduction: {total_improvement*100:.1f}%")
else:
    print(f"  ! Limited loss reduction: {total_improvement*100:.1f}%")

# Check for overfitting signs
if len(train_loss) > 50:
    recent_trend = np.polyfit(range(len(late_phase)), late_phase, 1)[0]
    if recent_trend > 0.01:
        print("  ! Potential overfitting: loss increasing in late phase")
    elif abs(recent_trend) < 0.005:
        print("  ✓ Stable late-phase training (no overfitting signs)")
    else:
        print("  ✓ Still improving in late phase")

# Recommendations
print("\nRECOMMENDATIONS:")
print("  1. URGENT: Collect more data (need >>3 samples for valid training)")
print("  2. Create proper train/val/test splits (currently 0 validation)")
print("  3. Reduce seq_len/pred_len to create more samples from existing runs")
print("  4. Consider data augmentation or synthetic data generation")
print("  5. Current model is likely overfitting to 3 training samples")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curve
ax = axes[0, 0]
ax.plot(train_loss, linewidth=2, color='#2E86AB')
ax.axhline(y=train_loss[-1], color='red', linestyle='--', alpha=0.5, label=f'Final: {train_loss[-1]:.2f}')
ax.axhline(y=train_loss.min(), color='green', linestyle='--', alpha=0.5, label=f'Best: {train_loss.min():.2f}')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss (NLL)', fontsize=11)
ax.set_title('Training Loss Curve', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Loss improvement per epoch
ax = axes[0, 1]
ax.plot(epoch_diff, linewidth=2, color='#A23B72')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax.fill_between(range(len(epoch_diff)), epoch_diff, 0, 
                 where=(epoch_diff < 0), alpha=0.3, color='green', label='Improvement')
ax.fill_between(range(len(epoch_diff)), epoch_diff, 0, 
                 where=(epoch_diff >= 0), alpha=0.3, color='red', label='Degradation')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss Change', fontsize=11)
ax.set_title('Per-Epoch Loss Change', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Moving average (window=5)
ax = axes[1, 0]
window = 5
moving_avg = np.convolve(train_loss, np.ones(window)/window, mode='valid')
ax.plot(train_loss, alpha=0.3, color='gray', label='Raw')
ax.plot(range(window-1, len(train_loss)), moving_avg, linewidth=2, 
        color='#F18F01', label=f'{window}-epoch MA')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.set_title('Smoothed Loss Trend', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Learning phases comparison
ax = axes[1, 1]
phases = ['Early\n(1-20)', 'Mid\n(21-60)', 'Late\n(61-100)']
phase_means = [early_phase.mean(), mid_phase.mean(), late_phase.mean()]
phase_stds = [early_phase.std(), mid_phase.std(), late_phase.std()]
colors = ['#06A77D', '#F18F01', '#D72638']

bars = ax.bar(phases, phase_means, yerr=phase_stds, capsize=5, 
              color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('Mean Loss', fontsize=11)
ax.set_title('Loss by Training Phase', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, mean, std in zip(bars, phase_means, phase_stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.2f}\n±{std:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(run_dir / 'training_analysis.png', dpi=150, bbox_inches='tight')
print(f"\nVisualization saved to {run_dir / 'training_analysis.png'}")

print("\n" + "=" * 70)
