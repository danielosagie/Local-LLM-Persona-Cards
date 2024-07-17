# create_exe.ps1

$code = @"
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;

class Program
{
    static void Main()
    {
        string tempPath = Path.GetTempPath();
        string scriptPath = Path.Combine(tempPath, "initial_installer.ps1");
        
        using (Stream resource = Assembly.GetExecutingAssembly().GetManifestResourceStream("initial_installer.ps1"))
        using (StreamReader reader = new StreamReader(resource))
        using (StreamWriter writer = new StreamWriter(scriptPath))
        {
            writer.Write(reader.ReadToEnd());
        }

        ProcessStartInfo startInfo = new ProcessStartInfo();
        startInfo.FileName = "powershell.exe";
        startInfo.Arguments = string.Format("-ExecutionPolicy Bypass -File \"{0}\"", scriptPath);
        startInfo.UseShellExecute = false;

        using (Process process = Process.Start(startInfo))
        {
            process.WaitForExit();
        }

        File.Delete(scriptPath);
    }
}
"@

$outputPath = "PersonaCardInstaller.exe"

Add-Type -OutputAssembly $outputPath -OutputType ConsoleApplication -TypeDefinition $code -ReferencedAssemblies "System.dll"

$assembly = [System.Reflection.Assembly]::LoadFile((Resolve-Path $outputPath))
$resourceName = "initial_installer.ps1"

if (-not ($assembly.GetManifestResourceInfo($resourceName))) {
    $assemblyBuilder = [System.Reflection.Emit.AssemblyBuilder]::DefineDynamicAssembly((New-Object System.Reflection.AssemblyName("ResourceAssembly")), [System.Reflection.Emit.AssemblyBuilderAccess]::Run)
    $moduleBuilder = $assemblyBuilder.DefineDynamicModule("ResourceModule", $outputPath)
    $resourceWriter = $moduleBuilder.DefineResource($resourceName, "Embedded")
    $resourceWriter.AddResource($resourceName, [System.IO.File]::ReadAllBytes("initial_installer.ps1"))
    $assemblyBuilder.Save($outputPath)
}

Write-Host "PersonaCardInstaller.exe created successfully."