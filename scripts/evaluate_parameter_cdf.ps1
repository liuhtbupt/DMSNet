param(
    [string]$TestH5 = "",
    [double[]]$Snrs = @(0),
    [int]$MaxBatches = 0
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $TestH5) {
    $TestH5 = Join-Path $Root "data\beiyou_test_2000_clean_K2_joint.h5"
}
$Arguments = @(
    (Join-Path $Root "evaluation\evaluate_parameter_cdf.py"),
    "--test-h5", $TestH5,
    "--snrs"
) + $Snrs
if ($MaxBatches -gt 0) {
    $Arguments += @("--max-batches", $MaxBatches)
}
python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Parameter-CDF evaluation failed with exit code $LASTEXITCODE"
}
