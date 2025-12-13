# Echo Startup Script
Write-Host "Initializing Echo Protocol..." -ForegroundColor Cyan

# Activate the virtual environment
& ".\venv311\Scripts\Activate.ps1"

# Run the assistant
python -m src.assistant
