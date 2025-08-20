Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
batPath = fso.BuildPath(scriptDir, "run_app.bat")

cmd = "cmd /c """ & batPath & """"
CreateObject("WScript.Shell").Run cmd, 0, False
