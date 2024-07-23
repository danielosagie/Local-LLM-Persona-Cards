# install_dependencies.ps1

# Function to check if a command is available
function Test-Command($command) {
    try { if (Get-Command $command -ErrorAction Stop) { return $true } }
    catch { return $false }
}

$ProgressPreference = 'Continue'

function Write-ProgressToFile($message) {
    Add-Content -Path "$env:TEMP\install_progress.txt" -Value $message
}

Write-ProgressToFile "Starting installation..."

# Check if Python is installed
$minVersion = [Version]"3.9.0"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand -or ($pythonCommand.Version -lt $minVersion)) {
    Write-ProgressToFile "Installing Python 3.11.7..."
    $pythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
    Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $pythonInstaller
}

Write-ProgressToFile "Python installed. Refreshing environment..."

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install required packages
$packages = @("streamlit", "streamlit_survey", "langchain", "langchain_community", "transformers", "torch", "sentence_transformers")
$totalPackages = $packages.Count
$currentPackage = 0

foreach ($package in $packages) {
    $currentPackage++
    $percentComplete = [math]::Round(($currentPackage / $totalPackages) * 100)
    Write-ProgressToFile "Installing $package... ($percentComplete% complete)"
    pip install $package
}

Write-ProgressToFile "All dependencies installed successfully."