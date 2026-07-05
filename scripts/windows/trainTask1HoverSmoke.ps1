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

$NumEnvs = if ($env:NUM_ENVS) { $env:NUM_ENVS } else { "512" }
$TotalEnvSteps = if ($env:TOTAL_ENV_STEPS) { $env:TOTAL_ENV_STEPS } else { "5000" }
$SaveFreqEnvSteps = if ($env:SAVE_FREQ_ENV_STEPS) { $env:SAVE_FREQ_ENV_STEPS } else { "5000" }
python -m quadrotor_rl.tasks.task1.task1_train --num-envs $NumEnvs --total-env-steps $TotalEnvSteps --save-freq-env-steps $SaveFreqEnvSteps @args
