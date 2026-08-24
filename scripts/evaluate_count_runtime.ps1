param(
    [string]$TestH5 = "",
    [double[]]$Snrs = @(-10, -5, 0, 5, 10, 15, 20),
    [int]$MaxBatches = 0
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $TestH5) {
    $TestH5 = Join-Path $Root "data\beiyou_test_10000_clean_joint.h5"
}
$Arguments = @(
    (Join-Path $Root "evaluation\evaluate_count_runtime.py"),
    "--test-h5", $TestH5,
    "--snrs"
) + $Snrs
if ($MaxBatches -gt 0) {
    $Arguments += @("--max-batches", $MaxBatches)
}
python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Count/runtime evaluation failed with exit code $LASTEXITCODE"
}
