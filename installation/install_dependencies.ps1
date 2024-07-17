# install_dependencies.ps1

# Function to check if a command is available
function Test-Command($command) {
    try { if (Get-Command $command -ErrorAction Stop) { return $true } }
    catch { return $false }
}

# Check if Python is installed
$minVersion = [Version]"3.9.0"
$maxVersion = [Version]"3.11.99"  # Adjust this as newer versions become stable
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand -or ($pythonCommand.Version -lt $minVersion) -or ($pythonCommand.Version -ge $maxVersion)) {
    Write-Host "Installing Python 3.11.7..."
    $pythonUrl = "https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe"
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

# Ensure Streamlit is in the PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
if (-not (Test-Command streamlit)) {
    Write-Host "Adding Streamlit to PATH..."
    $pythonScriptsPath = [System.IO.Path]::Combine([System.IO.Path]::GetDirectoryName((Get-Command python).Source), "Scripts")
    [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";" + $pythonScriptsPath, [System.EnvironmentVariableTarget]::User)
}

Write-Host "All dependencies installed successfully."