$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
function GetProjectRoot {
    param([string]$StartDir)
    $searchDir = (Resolve-Path $StartDir).Path
    while ($true) {
        $marker = Join-Path $searchDir "src\quadrotor_rl\__init__.py"
        if (Test-Path $marker) {
            return $searchDir
        }
        $parent = Split-Path -Parent $searchDir
        if ($parent -eq $searchDir -or [string]::IsNullOrWhiteSpace($parent)) {
            throw "Cannot locate project root from $StartDir"
        }
        $searchDir = $parent
    }
}
$ProjectRoot = GetProjectRoot $ScriptDir
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src;$env:PYTHONPATH"
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "1" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "1" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "1" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "1" }
$env:PYTHONUNBUFFERED = "1"
if (-not $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD) { $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1" }

python tests/task1/task1_env_test.py @args
