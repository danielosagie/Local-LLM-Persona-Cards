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
Filename: "powershell.exe"; \
    Parameters: "-ExecutionPolicy Bypass -File ""{app}\install_dependencies.ps1"""; \
    StatusMsg: "Installing Python and dependencies..."; \
    Flags: runhidden

[Code]
var
  ProgressPage: TOutputProgressWizardPage;

procedure InitializeWizard;
begin
  ProgressPage := CreateOutputProgressPage('Installing Dependencies', 'Please wait while we install necessary components...');
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
  begin
    ProgressPage.Show;
    try
      ProgressPage.SetText('Installing Python and dependencies...', '');
      ProgressPage.SetProgress(0, 0);

      // Run the PowerShell script
      if not Exec('powershell.exe', 
          '-ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install_dependencies.ps1') + '"', 
          '', SW_HIDE, ewWaitUntilTerminated, ResultCode) then
      begin
        MsgBox('Failed to install dependencies. Error code: ' + IntToStr(ResultCode), mbError, MB_OK);
      end;

      ProgressPage.SetProgress(100, 100);
    finally
      ProgressPage.Hide;
    end;
  end;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}"