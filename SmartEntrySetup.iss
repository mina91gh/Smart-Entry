#define MyAppName "Smart Entry"
#define MyAppVersion "1.0.0"

[Setup]
AppId={{A3D9F7C1-5B21-4D6E-9A42-8F61C0B7E235}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Smart Entry

DefaultDirName={autopf}\SmartEntry
DefaultGroupName=Smart Entry

OutputDir=installer-output
OutputBaseFilename=SmartEntrySetup

PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

WizardStyle=modern
Compression=lzma2
SolidCompression=yes

WizardSizePercent=120
WizardResizable=yes

Uninstallable=yes
CloseApplications=yes


[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Dirs]

; Main application data directory.
Name: "{commonappdata}\SmartEntry"

; InsightFace models directory.
Name: "{commonappdata}\SmartEntry\.insightface\models\buffalo_l"; Permissions: users-readexec; Check: IsServerInstall

; Face dataset directory.
Name: "{commonappdata}\SmartEntry\Face-Collection\dataset"; Permissions: users-modify; Check: IsServerInstall

; Profile images directory.
Name: "{commonappdata}\SmartEntry\Web-UI\static\uploads\people"; Permissions: users-modify; Check: IsServerInstall

; Application logs directory.
Name: "{commonappdata}\SmartEntry\Logs"; Permissions: users-modify; Check: IsServerInstall


[Files]

; ==============================
; SERVER INSTALLATION
; ==============================

Source: "dist\SmartEntryServer.exe"; DestDir: "{app}\Server"; Flags: ignoreversion; Check: IsServerInstall


; ==============================
; ADMIN / CLIENT INSTALLATION
; ==============================

Source: "dist\SmartEntryAdmin.exe"; DestDir: "{app}\Admin"; Flags: ignoreversion; Check: IsClientInstall


; ==============================
; INSIGHTFACE MODELS
; ==============================

Source: "installer-payload\models\buffalo_l\*"; DestDir: "{commonappdata}\SmartEntry\.insightface\models\buffalo_l"; Flags: onlyifdoesntexist; Check: IsServerInstall



[Icons]

; Server shortcut in Start Menu.
Name: "{autoprograms}\Smart Entry\Smart Entry Server"; Filename: "{app}\Server\SmartEntryServer.exe"; Check: IsServerInstall

; Admin shortcut in Start Menu.
Name: "{autoprograms}\Smart Entry\Smart Entry Admin"; Filename: "{app}\Admin\SmartEntryAdmin.exe"; Check: IsClientInstall



[Registry]

; Start Smart Entry Server automatically when Windows user logs in.
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "SmartEntryServer"; ValueData: """{app}\Server\SmartEntryServer.exe"""; Flags: uninsdeletevalue; Check: IsServerInstall



[Run]

; Allow Smart Entry Server API on private networks.
Filename: "{sys}\netsh.exe"; Parameters: "advfirewall firewall add rule name=""Smart Entry Server"" dir=in action=allow protocol=TCP localport=8002 profile=private"; Flags: runhidden; Check: IsServerInstall

; Start Server after installation.
Filename: "{app}\Server\SmartEntryServer.exe"; Description: "Start Smart Entry Server now"; Flags: postinstall nowait skipifsilent; Check: IsServerInstall

; Start Admin after installation.
Filename: "{app}\Admin\SmartEntryAdmin.exe"; Description: "Start Smart Entry Admin now"; Flags: postinstall nowait skipifsilent; Check: IsClientInstall

[Code]

var
  RolePage: TInputOptionWizardPage;
  ServerPage: TInputQueryWizardPage;
  CameraPage: TInputQueryWizardPage;
  ClientPage: TInputQueryWizardPage;

function IsServerInstall: Boolean;
begin
  Result := RolePage.SelectedValueIndex = 0;
end;


function IsClientInstall: Boolean;
begin
  Result := RolePage.SelectedValueIndex = 1;
end;


function ServerConfigPath: String;
begin
  Result := ExpandConstant(
    '{commonappdata}\SmartEntry\smart-entry.env'
  );
end;


function ClientConfigPath: String;
begin
  Result := ExpandConstant(
    '{commonappdata}\SmartEntry\client.env'
  );
end;



procedure InitializeWizard;
begin

  RolePage := CreateInputOptionPage(
    wpWelcome,
    'Installation Type',
    'Choose the installation type',
    'Select which Smart Entry component you want to install:',
    True,
    False
  );

  RolePage.Add('Server');
  RolePage.Add('Admin / Client');
  RolePage.SelectedValueIndex := 0;
  ServerPage := CreateInputQueryPage(
    RolePage.ID,
    'Server Configuration',
    'Configure the Smart Entry Server',
    'Enter the PostgreSQL and network camera settings.'
  );

  ServerPage.Add('Database name:', False);
  ServerPage.Add('Database user:', False);
  ServerPage.Add('Database password:', True);
  ServerPage.Add('Database host:', False);
  ServerPage.Add('Database port:', False);


  ServerPage.Values[0] := 'face-reco';
  ServerPage.Values[1] := 'postgres';
  ServerPage.Values[2] := '';
  ServerPage.Values[3] := 'localhost';
  ServerPage.Values[4] := '5432';
  
  CameraPage := CreateInputQueryPage(
    ServerPage.ID,
    'Camera Configuration',
    'Configure the Network Camera',
    'Enter the RTSP URL of your network camera.'
  );

  CameraPage.Add('RTSP camera URL:', False);
  CameraPage.Values[0] := '';
  ClientPage := CreateInputQueryPage(
    CameraPage.ID,
    'Admin Configuration',
    'Connect Admin to the Server',
    'Enter the IP address or URL of the Smart Entry Server.'
  );

  ClientPage.Add('Server URL:', False);
  ClientPage.Values[0] := 'http://192.168.1.100:8002';
end;

function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := False;

  if IsClientInstall then
  begin
    if (PageID = ServerPage.ID) or
       (PageID = CameraPage.ID) then
      Result := True;
  end;

  if IsServerInstall then
  begin
    if PageID = ClientPage.ID then
      Result := True;
  end;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  DBPort: Integer;
  ServerURL: String;
begin
  Result := True;

  if (CurPageID = ServerPage.ID) and IsServerInstall then
  begin
    if Trim(ServerPage.Values[0]) = '' then
    begin
      MsgBox('Database name is required.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(ServerPage.Values[1]) = '' then
    begin
      MsgBox('Database user is required.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(ServerPage.Values[2]) = '' then
    begin
      MsgBox('Database password is required.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Trim(ServerPage.Values[3]) = '' then
    begin
      MsgBox('Database host is required.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    DBPort := StrToIntDef(Trim(ServerPage.Values[4]), 0);

    if (DBPort < 1) or (DBPort > 65535) then
    begin
      MsgBox('Please enter a valid database port.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;
  
  if (CurPageID = CameraPage.ID) and IsServerInstall then
  begin
    if Trim(CameraPage.Values[0]) = '' then
    begin
      MsgBox('RTSP camera URL is required.', mbError, MB_OK);
      Result := False;
      Exit;
    end;

    if Pos('rtsp://', Lowercase(Trim(CameraPage.Values[0]))) <> 1 then
    begin
      MsgBox('Please enter a valid RTSP URL starting with rtsp://', mbError, MB_OK);
      Result := False;
      Exit;
    end;
  end;
  
  if (CurPageID = ClientPage.ID) and IsClientInstall then
  begin
    ServerURL := Trim(ClientPage.Values[0]);

    if (ServerURL = '') or
       ((Pos('http://', Lowercase(ServerURL)) <> 1) and
        (Pos('https://', Lowercase(ServerURL)) <> 1)) then
    begin
      MsgBox(
        'Please enter a valid Server URL starting with http:// or https://',
        mbError,
        MB_OK
      );
      Result := False;
      Exit;
    end;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ConfigContent: String;
begin
  if CurStep = ssPostInstall then
  begin

    if IsServerInstall then
    begin
      if not FileExists(ServerConfigPath) then
      begin
        ConfigContent :=
          'SMART_ENTRY_DB_NAME=' + Trim(ServerPage.Values[0]) + #13#10 +
          'SMART_ENTRY_DB_USER=' + Trim(ServerPage.Values[1]) + #13#10 +
          'SMART_ENTRY_DB_PASSWORD=' + ServerPage.Values[2] + #13#10 +
          'SMART_ENTRY_DB_HOST=' + Trim(ServerPage.Values[3]) + #13#10 +
          'SMART_ENTRY_DB_PORT=' + Trim(ServerPage.Values[4]) + #13#10 +
          'SMART_ENTRY_RTSP_URL=' + Trim(CameraPage.Values[0]) + #13#10;

        if not SaveStringToFile(ServerConfigPath, ConfigContent, False) then
          MsgBox('Could not create the Server configuration file.', mbError, MB_OK);
      end;
    end;

    if IsClientInstall then
    begin
      if not FileExists(ClientConfigPath) then
      begin
        ConfigContent :=
          'SMART_ENTRY_SERVER_URL=' + Trim(ClientPage.Values[0]) + #13#10;

        if not SaveStringToFile(ClientConfigPath, ConfigContent, False) then
          MsgBox('Could not create the Client configuration file.', mbError, MB_OK);
      end;
    end;

  end;
end;




