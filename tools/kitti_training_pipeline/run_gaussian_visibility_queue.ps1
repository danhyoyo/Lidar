param(
    [ValidateSet("a0", "a1", "a2", "a3", "a4")]
    [string]$CurrentVariant = "a1",
    [int]$Seed = 42,
    [int]$Epochs = 50,
    [int]$PhysicalBatchSize = 16,
    [int]$AccumulationSteps = 1,
    [ValidateSet("fp32", "fp16", "bf16")]
    [string]$Precision = "bf16",
    [int]$NumWorkers = 0,
    [string]$Device = "cuda",
    [string]$PythonExe = "python",
    [string]$KittiRoot = "",
    [string]$OutputRoot = "",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptRoot "..\..")).Path
if ([string]::IsNullOrWhiteSpace($KittiRoot)) {
    $KittiRoot = Join-Path (Split-Path -Parent $repoRoot) "datasets\KITTI\object"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repoRoot "artifacts\kitti_50_16_batch_num3"
}

$runner = Join-Path $repoRoot "tools\kitti_training_pipeline\run_gaussian_visibility_ablation.py"
$evaluationTable = Join-Path $repoRoot "tools\kitti_training_pipeline\evaluation_table.py"
$efficiencyTable = Join-Path $repoRoot "tools\kitti_training_pipeline\efficiency_table.py"
$tablePrefix = Join-Path $repoRoot "evaluation\gaussian_visibility_table"
$efficiencyPrefix = Join-Path $repoRoot "evaluation\gaussian_visibility_efficiency"
$variants = @("a0", "a1", "a2", "a3", "a4")
$startIndex = [Array]::IndexOf($variants, $CurrentVariant)
$queue = @($variants[$startIndex..($variants.Count - 1)])

function Invoke-Python {
    param([string[]]$Arguments)

    Write-Host "`n>>> $PythonExe $($Arguments -join ' ')" -ForegroundColor Cyan
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-VariantStage {
    param(
        [string]$Variant,
        [ValidateSet("train", "select", "report")]
        [string]$Stage
    )

    $arguments = @(
        $runner,
        "--variant", $Variant,
        "--seed", $Seed,
        "--stage", $Stage,
        "--device", $Device,
        "--kitti-root", $KittiRoot,
        "--output-root", $OutputRoot
    )
    if ($Stage -eq "train") {
        $arguments += @(
            "--epochs", $Epochs,
            "--physical-batch-size", $PhysicalBatchSize,
            "--accumulation-steps", $AccumulationSteps,
            "--precision", $Precision,
            "--num-workers", $NumWorkers
        )
    }
    Invoke-Python $arguments
}

function Get-ActiveVariantProcesses {
    param([string]$Variant)

    try {
        $processes = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'")
    } catch {
        throw "Cannot inspect running Python processes; start this queue after the current training finishes. Details: $($_.Exception.Message)"
    }
    return @($processes | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "run_gaussian_visibility_ablation\.py" -and
        $_.CommandLine -match ("--variant\s+" + [regex]::Escape($Variant))
    })
}

function Wait-ForCurrentTraining {
    param([string]$Variant)

    $runDir = Join-Path $OutputRoot "gaussian_visibility_${Variant}_seed${Seed}"
    $finalCheckpoint = Join-Path $runDir "checkpoints\${Epochs}epoch.pt"
    Write-Host "Waiting for the current $Variant training process to finish..." -ForegroundColor Yellow
    while ($true) {
        $active = @(Get-ActiveVariantProcesses $Variant)
        if ($active.Count -eq 0) {
            break
        }
        Write-Host "  $($active.Count) matching Python process(es) still running; checking again in $PollSeconds seconds..."
        Start-Sleep -Seconds $PollSeconds
    }
    if (-not (Test-Path -LiteralPath $finalCheckpoint)) {
        return $false
    }
    Write-Host "$Variant training completed: $finalCheckpoint" -ForegroundColor Green
    return $true
}

function Update-EvaluationTable {
    $reports = @()
    foreach ($variant in $variants) {
        $report = Join-Path $OutputRoot "gaussian_visibility_${variant}_seed${Seed}\report.json"
        if (Test-Path -LiteralPath $report) {
            $reports += $report
        }
    }
    if ($reports.Count -eq 0) {
        return
    }
    $arguments = @($evaluationTable) + $reports + @("--output-prefix", $tablePrefix)
    Invoke-Python $arguments
    $efficiencyArguments = @($efficiencyTable) + $reports + @("--output-prefix", $efficiencyPrefix)
    Invoke-Python $efficiencyArguments
}

Write-Host "Queue: $($queue -join ' -> ')" -ForegroundColor Green
Write-Host "Output root: $OutputRoot"
Write-Host "Num workers: $NumWorkers"

# The current variant was started outside this queue, so wait for it and then
# process its selection/report stages without training it a second time.
if (-not (Wait-ForCurrentTraining $CurrentVariant)) {
    $currentRunDir = Join-Path $OutputRoot "gaussian_visibility_${CurrentVariant}_seed${Seed}"
    $currentMetrics = Join-Path $currentRunDir "metrics.jsonl"
    if (Test-Path -LiteralPath $currentMetrics) {
        throw "The current $CurrentVariant run has partial metrics but no final checkpoint. Use a fresh output root or clean that incomplete run before retrying."
    }
    Write-Warning "$CurrentVariant did not leave a final checkpoint; retrying its training once."
    Invoke-VariantStage $CurrentVariant "train"
    $finalCheckpoint = Join-Path $currentRunDir "checkpoints\${Epochs}epoch.pt"
    if (-not (Test-Path -LiteralPath $finalCheckpoint)) {
        throw "Retry of $CurrentVariant did not leave the expected final checkpoint: $finalCheckpoint"
    }
}
Invoke-VariantStage $CurrentVariant "select"
Invoke-VariantStage $CurrentVariant "report"
Update-EvaluationTable

foreach ($variant in @($queue | Select-Object -Skip 1)) {
    Invoke-VariantStage $variant "train"
    Invoke-VariantStage $variant "select"
    Invoke-VariantStage $variant "report"
    Update-EvaluationTable
}

Write-Host "`nQueue completed. Final tables: $tablePrefix.md, $tablePrefix.csv, $efficiencyPrefix.md, and $efficiencyPrefix.csv" -ForegroundColor Green
