Set objShell = CreateObject("Wscript.Shell")
strPath = Wscript.ScriptFullName
Set objFSO = CreateObject("Scripting.FileSystemObject")
strFolder = objFSO.GetParentFolderName(strPath)
objShell.CurrentDirectory = strFolder
objShell.Run "PersonaCard.bat", 0, False