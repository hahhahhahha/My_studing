param(
    [string]$PythonPath = 'X:\anaconda\ANACONDA\python.exe',
    [switch]$SkipCheck
)

# Process scope only: no system/user environment, packages, or drivers change.
$windows = (Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').SystemRoot
$programs = Get-ItemProperty -LiteralPath 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion'
$required = @{
    SystemRoot = $windows
    windir = $windows
    ProgramFiles = $programs.ProgramFilesDir
    ProgramW6432 = $programs.ProgramW6432Dir
}
foreach ($name in $required.Keys) {
    if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) {
        $value = $required[$name]
        if (-not $value -or -not (Test-Path -LiteralPath $value -PathType Container)) {
            throw "Cannot resolve required Windows path: $name"
        }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
        Write-Output "Restored process variable: $name"
    }
}
if (-not $env:USERNAME) {
    $env:USERNAME = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name.Split('\')[-1]
}
foreach ($name in @('TEMP', 'TMP')) {
    if (-not [Environment]::GetEnvironmentVariable($name, 'Process')) {
        $temporary = Join-Path $PSScriptRoot '.runtime_cache\tmp'
        New-Item -ItemType Directory -Path $temporary -Force | Out-Null
        [Environment]::SetEnvironmentVariable($name, $temporary, 'Process')
    }
}
if (-not $SkipCheck) {
    & $PythonPath -B (Join-Path $PSScriptRoot 'scripts\verify_pytorch_gpu.py')
    if ($LASTEXITCODE -ne 0) {
        throw "PyTorch/GPU verification failed; see protocol\PYTORCH_GPU_INITIALIZATION.json"
    }
}
