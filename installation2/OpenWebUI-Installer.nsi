!include "MUI2.nsh"
!include "FileFunc.nsh"

; Custom name for your app
!define APP_NAME "OpenWebUI"
!define COMP_NAME "OpenWebUI"

; Change installer name
Name "${APP_NAME} Installer"
OutFile "${APP_NAME}-Installer.exe"
InstallDir "$PROGRAMFILES\${APP_NAME}"

; Add your custom icon to the installer
!define MUI_ICON "custom_logo.ico"
!define MUI_UNICON "custom_logo.ico"

; Add a custom banner to the installer pages
!define MUI_WELCOMEFINISHPAGE_BITMAP "installer_banner.bmp"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    
    ; Copy your custom icon to the installation directory
    File "custom_logo.ico"
    
    ; Download and install Docker if not present
    nsExec::ExecToLog 'powershell -Command "if (!(Get-Command docker -ErrorAction SilentlyContinue)) { Invoke-WebRequest -Uri https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe -OutFile dockerinstaller.exe; Start-Process dockerinstaller.exe -Wait }"'
    
    ; Download and install Ollama
    nsExec::ExecToLog 'powershell -Command "Invoke-WebRequest -Uri https://ollama.ai/download/ollama-windows-amd64.exe -OutFile ollama.exe; Start-Process ollama.exe -Wait"'
    
    ; Pull OpenWebUI Docker image
    nsExec::ExecToLog 'docker pull ghcr.io/open-webui/open-webui:main'
    
    ; Create batch file to run OpenWebUI
    FileOpen $0 "$INSTDIR\run-${APP_NAME}.bat" w
    FileWrite $0 '@echo off$\r$\n'
    FileWrite $0 'docker start open-webui || docker run -d -p 3000:8080 --add-host=host.docker.internal:host-gateway -v open-webui:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main$\r$\n'
    FileWrite $0 'start http://localhost:3000$\r$\n'
    FileClose $0
    
    ; Create desktop shortcut with custom name and icon
    CreateShortCut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\run-${APP_NAME}.bat" "" "$INSTDIR\custom_logo.ico"
    
    ; Start the application
    Exec '"$INSTDIR\run-${APP_NAME}.bat"'
SectionEnd

; Add information about your application
VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "CompanyName" "${COMP_NAME}"
VIAddVersionKey "LegalCopyright" "© ${COMP_NAME}"
VIAddVersionKey "FileDescription" "${APP_NAME}"
VIAddVersionKey "FileVersion" "1.0.0"