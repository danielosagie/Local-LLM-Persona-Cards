# install_dependencies.ps1

# Function to check if a command is available
function Test-Command($command) {
    try { if (Get-Command $command -ErrorAction Stop) { return $true } }
    catch { return $false }
}

# Install Python if not present
if (-not (Test-Command python)) {
    Write-Host "Installing Python..."
    $pythonUrl = "https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller
    Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $pythonInstaller
}

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Install pip if not present
if (-not (Test-Command pip)) {
    Write-Host "Installing pip..."
    python -m ensurepip --upgrade
}

# Install required Python packages
$packages = @("streamlit", "streamlit_survey", "langchain", "langchain_community", "transformers", "torch", "sentence_transformers")
foreach ($package in $packages) {
    Write-Host "Installing $package..."
    pip install $package
}

Write-Host "All dependencies installed successfully."