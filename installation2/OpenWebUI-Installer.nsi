!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

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
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_LANGUAGE "English"

Section "OpenWebUI" SecOpenWebUI
    SetOutPath $INSTDIR
    
    ; Copy your custom icon to the installation directory
    File "custom_logo.ico"
    
    ; Download and install Docker if not present
    nsExec::ExecToLog 'powershell -Command "if (!(Get-Command docker -ErrorAction SilentlyContinue)) { Invoke-WebRequest -Uri https://desktop.docker.com/win/stable/Docker%20Desktop%20Installer.exe -OutFile dockerinstaller.exe; Start-Process dockerinstaller.exe -Wait }"'
    
    ; Download and install Ollama
    nsExec::ExecToLog 'powershell -Command "Invoke-WebRequest -Uri https://ollama.ai/download/ollama-windows-amd64.exe -OutFile ollama.exe; Start-Process ollama.exe -Wait"'
    
    ; Pull OpenWebUI Docker image
    nsExec::ExecToLog 'docker pull ghcr.io/open-webui/open-webui:main'
    
    ; Copy the manager script
    File "openwebui_manager.py"
    
    ; Create batch file to run OpenWebUI manager
    FileOpen $0 "$INSTDIR\run-${APP_NAME}-manager.bat" w
    FileWrite $0 '@echo off$\r$\n'
    FileWrite $0 'start "" pythonw "$INSTDIR\openwebui_manager.py"$\r$\n'
    FileClose $0
    
    ; Create desktop shortcut for OpenWebUI Manager
    CreateShortCut "$DESKTOP\${APP_NAME} Manager.lnk" "$INSTDIR\run-${APP_NAME}-manager.bat" "" "$INSTDIR\custom_logo.ico"
    
    ; Create Start Menu shortcuts for OpenWebUI Manager
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\${APP_NAME} Manager.lnk" "$INSTDIR\run-${APP_NAME}-manager.bat" "" "$INSTDIR\custom_logo.ico"
    
    ; Create desktop shortcut for Docker Desktop
    CreateShortCut "$DESKTOP\Docker Desktop.lnk" "$PROGRAMFILES\Docker\Docker\Docker Desktop.exe" "" "$PROGRAMFILES\Docker\Docker\Docker Desktop.exe" 0
    
    ; Create Start Menu shortcut for Docker Desktop
    CreateShortCut "$SMPROGRAMS\${APP_NAME}\Docker Desktop.lnk" "$PROGRAMFILES\Docker\Docker\Docker Desktop.exe" "" "$PROGRAMFILES\Docker\Docker\Docker Desktop.exe" 0
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    ; Add to startup
    WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${APP_NAME} Manager" "$INSTDIR\run-${APP_NAME}-manager.bat"
SectionEnd

Section "Run OpenWebUI Manager after installation" SecRunApp
    Exec '"$INSTDIR\run-${APP_NAME}-manager.bat"'
SectionEnd

Section "Uninstall"
    ; Remove OpenWebUI
    ExecWait 'docker stop open-webui'
    ExecWait 'docker rm open-webui'
    ExecWait 'docker rmi ghcr.io/open-webui/open-webui:main'
    
    ; Remove files and shortcuts
    Delete "$DESKTOP\${APP_NAME} Manager.lnk"
    Delete "$DESKTOP\Docker Desktop.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME} Manager.lnk"
    Delete "$SMPROGRAMS\${APP_NAME}\Docker Desktop.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    Delete "$INSTDIR\run-${APP_NAME}-manager.bat"
    Delete "$INSTDIR\openwebui_manager.py"
    Delete "$INSTDIR\custom_logo.ico"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir "$INSTDIR"
    
    ; Remove from startup
    DeleteRegValue HKCU "Software\Microsoft\Windows\CurrentVersion\Run" "${APP_NAME} Manager"
SectionEnd

; Descriptions
LangString DESC_SecOpenWebUI ${LANG_ENGLISH} "Install OpenWebUI and Docker Desktop"
LangString DESC_SecRunApp ${LANG_ENGLISH} "Run OpenWebUI Manager after installation"

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SecOpenWebUI} $(DESC_SecOpenWebUI)
  !insertmacro MUI_DESCRIPTION_TEXT ${SecRunApp} $(DESC_SecRunApp)
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; Add information about your application
VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "${APP_NAME}"
VIAddVersionKey "CompanyName" "${COMP_NAME}"
VIAddVersionKey "LegalCopyright" "© 2024 ${COMP_NAME}"
VIAddVersionKey "FileDescription" "${APP_NAME} Installer"
VIAddVersionKey "FileVersion" "1.0.0"