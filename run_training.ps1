# PowerShell script to run STMGT training in background
# Usage: .\run_training.ps1

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$outputDir = "outputs/stmgt_baseline_1month_$timestamp"
$logFile = "training_$timestamp.log"

Write-Host "Starting STMGT training..."
Write-Host "Output directory: $outputDir"
Write-Host "Log file: $logFile"
Write-Host ""

# Run training process
Start-Process -FilePath "python" `
    -ArgumentList "scripts/training/train_stmgt.py", `
                  "--config", "configs/training/stmgt_baseline_1month.json", `
                  "--output", $outputDir `
    -NoNewWindow `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "${logFile}.err"

Write-Host "Training started successfully!"
Write-Host ""
Write-Host "To monitor progress:"
Write-Host "  Get-Content $logFile -Wait -Tail 20"
Write-Host ""
Write-Host "To check output:"
Write-Host "  ls $outputDir"
