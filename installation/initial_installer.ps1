# initial_installer.ps1

$ErrorActionPreference = "Stop"

function Get-FileFromWeb($url, $outputPath) {
    Invoke-WebRequest -Uri $url -OutFile $outputPath
}

Write-Host "Downloading installation files..."
$scriptUrl = "https://raw.githubusercontent.com/danielosagie/Local-LLM-Persona-Cards/main/installer.ps1"
$scriptPath = Join-Path $env:TEMP "installer.ps1"
Get-FileFromWeb $scriptUrl $scriptPath

Write-Host "Starting installation..."
PowerShell -ExecutionPolicy Bypass -File $scriptPath