# PersonaCardInstaller.ps1

# Function to check if a command is available
function Test-Command($command) {
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = 'stop'
    try { if (Get-Command $command) { return $true } }
    catch { return $false }
    finally { $ErrorActionPreference = $oldPreference }
}

# Function to download a file
function Get-FileFromWeb($url, $outputPath) {
    Invoke-WebRequest -Uri $url -OutFile $outputPath
}

# Create installation directory
$installDir = "$env:LOCALAPPDATA\PersonaCard"
New-Item -ItemType Directory -Force -Path $installDir

# Check and install Python
if (-not (Test-Command python)) {
    Write-Host "Python is not installed. Installing Python..."
    $pythonUrl = "https://www.python.org/ftp/python/3.9.7/python-3.9.7-amd64.exe"
    $pythonInstaller = "$env:TEMP\python-installer.exe"
    Download-File $pythonUrl $pythonInstaller
    Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1" -Wait
    Remove-Item $pythonInstaller
}

# Check and install pip
if (-not (Test-Command pip)) {
    Write-Host "pip is not installed. Installing pip..."
    python -m ensurepip --upgrade
}

# Install required Python packages
$packages = @("streamlit", "streamlit_survey", "langchain", "langchain_community", "transformers", "torch", "sentence_transformers", "pyinstaller")
foreach ($package in $packages) {
    if (-not (python -c "import $package" -gt $null)) {
        Write-Host "Installing $package..."
        pip install $package
    }
}

# Download PersonaCard.py from GitHub
$personaCardUrl = "https://raw.githubusercontent.com/danielosagie/Local-LLM-Persona-Cards/main/PersonaCard.py"
$personaCardPath = "$installDir\PersonaCard.py"
Get-FileFromWeb $personaCardUrl $personaCardPath

# Create the launcher script
$launcherScript = @"
import subprocess
import sys
import os

if getattr(sys, 'frozen', False):
    application_path = sys._MEIPASS
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

os.chdir(application_path)
subprocess.call(["streamlit", "run", "PersonaCard.py"])
"@

Set-Content -Path "$installDir\PersonaCardLauncher.py" -Value $launcherScript

# Create the executable
Set-Location $installDir
pyinstaller --name="PersonaCard" --onefile --add-data "PersonaCard.py;." --add-data "$env:LOCALAPPDATA\Programs\Python\Python39\Lib\site-packages\streamlit;streamlit" --hidden-import=streamlit.web.bootstrap --hidden-import=streamlit.runtime.scriptrunner.magic_funcs --noconsole PersonaCardLauncher.py

# Create desktop shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\PersonaCard.lnk")
$Shortcut.TargetPath = "$installDir\dist\PersonaCard.exe"
$Shortcut.Save()

Write-Host "Installation complete! A desktop shortcut has been created for PersonaCard."
pause