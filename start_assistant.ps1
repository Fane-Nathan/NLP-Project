# Echo Startup Script
Write-Host "Initializing Echo Protocol..." -ForegroundColor Cyan

# Activate the virtual environment
& ".\DocumentRAG\Scripts\Activate.ps1"

# Run the assistant
python -m src.assistant
