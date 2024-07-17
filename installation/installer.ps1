# installer.ps1

$ErrorActionPreference = "Stop"

# Function to check if a command is available
function Test-Command($command) {
    try { if (Get-Command $command -ErrorAction Stop) { return $true } }
    catch { return $false }
}

# Function to download a file
function Get-FileFromWeb($url, $outputPath) {
    Invoke-WebRequest -Uri $url -OutFile $outputPath
}

# Create installation directory
$installDir = "$env:LOCALAPPDATA\PersonaCard"
New-Item -ItemType Directory -Force -Path $installDir | Out-Null

# Check and install Python
if (-not (Test-Command python)) {
    Write-Host "Python is not installed. Installing Python..."
    $pythonUrl = "https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    Download-File $pythonUrl $pythonInstaller
    Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $pythonInstaller
}

# Refresh environment variables
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Check and install pip
if (-not (Test-Command pip)) {
    Write-Host "pip is not installed. Installing pip..."
    python -m ensurepip --upgrade
}

# Install required Python packages
$packages = @("streamlit", "streamlit_survey", "langchain", "langchain_community", "transformers", "torch", "sentence_transformers")
foreach ($package in $packages) {
    Write-Host "Installing $package..."
    pip install $package
}

# Download PersonaCard.py and other necessary files
$files = @{
    "PersonaCard.py" = "https://raw.githubusercontent.com/danielosagie/Local-LLM-Persona-Cards/main/PersonaCard.py"
    # Add other files as needed
}

foreach ($file in $files.GetEnumerator()) {
    Write-Host "Downloading $($file.Key)..."
    Download-File $file.Value "$installDir\$($file.Key)"
}

# Create the launcher script
$launcherScript = @"
import subprocess
import os
import sys

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)
subprocess.call(["streamlit", "run", "PersonaCard.py"])
"@

Set-Content -Path "$installDir\PersonaCardLauncher.py" -Value $launcherScript

# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\PersonaCard.lnk")
$Shortcut.TargetPath = "python"
$Shortcut.Arguments = """$installDir\PersonaCardLauncher.py"""
$Shortcut.WorkingDirectory = $installDir
$Shortcut.Save()

Write-Host "Installation complete! A desktop shortcut has been created for PersonaCard."

# Ask if user wants to start the app
$startApp = Read-Host "Do you want to start the app now? (y/n)"
if ($startApp -eq 'y') {
    Start-Process python -ArgumentList """$installDir\PersonaCardLauncher.py"""
}

# Create update checker script
$updateCheckerScript = @"
import os
import urllib.request
import json

def check_for_updates():
    local_version_file = os.path.join(os.environ['LOCALAPPDATA'], 'PersonaCard', 'version.json')
    if os.path.exists(local_version_file):
        with open(local_version_file, 'r') as f:
            local_version = json.load(f)['version']
    else:
        local_version = '0.0.0'

    remote_version_url = "https://raw.githubusercontent.com/danielosagie/Local-LLM-Persona-Cards/main/version.json"
    with urllib.request.urlopen(remote_version_url) as response:
        remote_version = json.loads(response.read())['version']

    if remote_version > local_version:
        print(f"A new version ({remote_version}) is available. You have version {local_version}.")
        update = input("Do you want to update? (y/n): ")
        if update.lower() == 'y':
            os.system('powershell -ExecutionPolicy Bypass -File "%LOCALAPPDATA%\\PersonaCard\\installer.ps1"')
    else:
        print("You have the latest version.")

if __name__ == "__main__":
    check_for_updates()
"@

Set-Content -Path "$installDir\update_checker.py" -Value $updateCheckerScript

# Create a version file
$version = @{version="1.0.0"}
$version | ConvertTo-Json | Set-Content "$installDir\version.json"

# Add update checker to the launcher
$updatedLauncherScript = @"
import subprocess
import os
import sys

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)

# Run update checker
subprocess.call(["python", "update_checker.py"])

# Run the main app
subprocess.call(["streamlit", "run", "PersonaCard.py"])
"@

Set-Content -Path "$installDir\PersonaCardLauncher.py" -Value $updatedLauncherScript

Write-Host "Update checker has been added. The app will check for updates each time it's launched."