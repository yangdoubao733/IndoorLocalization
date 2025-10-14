# Indoor Localization System GUI Launcher

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Indoor Localization System - Starting..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Initialize conda
Write-Host "[INFO] Initializing Conda..." -ForegroundColor Yellow
& "D:\Anaconda\shell\condabin\conda-hook.ps1"

# Activate conda environment
Write-Host "[INFO] Activating Conda environment: inDoorLocalization" -ForegroundColor Yellow
conda activate inDoorLocalization

if ($LASTEXITCODE -eq 0) {
    Write-Host "[SUCCESS] Conda environment activated" -ForegroundColor Green
    Write-Host "[INFO] Starting GUI..." -ForegroundColor Yellow
    Write-Host ""

    # Run GUI
    python gui.py

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "GUI Closed" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan
} else {
    Write-Host "[ERROR] Failed to activate Conda environment" -ForegroundColor Red
    Write-Host "Please check environment name: inDoorLocalization" -ForegroundColor Red
}

Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
