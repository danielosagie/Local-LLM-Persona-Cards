#define MyAppName "PersonaCard"
#define MyAppVersion "1.0"
#define MyAppPublisher "GTRI"
#define MyAppURL "https://github.com/danielosagie/Local-LLM-Persona-Cards"
#define MyAppExeName "PersonaCardLauncher.vbs"
#define MyAppIcon "PersonaCard.ico"

[Setup]
AppId={{XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE.txt
OutputDir=.
OutputBaseFilename=PersonaCardInstaller
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile={#MyAppIcon}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "PersonaCard.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "PersonaCardLauncher.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "install_dependencies.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "README.md"; DestDir: "{app}"; Flags: ignoreversion isreadme
Source: "{#MyAppIcon}"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"; WorkingDir: "{app}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIcon}"; Tasks: desktopicon; WorkingDir: "{app}"

[Run]
Filename: "powershell.exe"; Parameters: "-ExecutionPolicy Bypass -File ""{app}\install_dependencies.ps1"""; Description: "Install dependencies"; Flags: runhidden
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not RegKeyExists(HKEY_LOCAL_MACHINE, 'SOFTWARE\Microsoft\PowerShell\1\PowerShellEngine') then
  begin
    MsgBox('This installer requires PowerShell to be installed.' #13#13 'Please install PowerShell and run this installer again.', mbInformation, MB_OK);
    Result := False;
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}"