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

$CheckpointPath = $env:CHECKPOINT
if (-not $CheckpointPath) {
  $Latest = Get-ChildItem -Path "logs/task4" -Recurse -Directory -Filter final_checkpoint -ErrorAction SilentlyContinue | Sort-Object FullName | Select-Object -Last 1
  if ($Latest) { $CheckpointPath = $Latest.FullName }
}
if (-not $CheckpointPath) { throw "checkpoint not found. Set CHECKPOINT=/path/to/checkpoint" }
python -m quadrotor_rl.tasks.task4.task4_model_test --checkpoint $CheckpointPath --visualize @args
