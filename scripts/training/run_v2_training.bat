@echo off
REM STMGT v2 Training Script
REM Date: 2025-11-10
REM Config: train_normalized_v2.json (hidden_dim=128, heads=8, K=7)

echo ================================================================================
echo STMGT v2 Training
echo ================================================================================
echo.
echo Model: hidden_dim=128, num_heads=8, num_blocks=3, K=7
echo Expected: MAE 2.85-2.95 km/h, R2 0.82-0.85
echo Training time: ~13 min/epoch, ~22 hours total
echo.
echo Starting training...
echo.

cd /d D:\UNI\DSP391m\project

C:\ProgramData\miniconda3\Scripts\conda.exe run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_normalized_v2.json

echo.
echo ================================================================================
echo Training completed!
echo Check outputs/ directory for results
echo ================================================================================
pause
