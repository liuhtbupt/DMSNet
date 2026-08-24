param(
    [Parameter(Mandatory = $true)][string]$InputNpz,
    [Parameter(Mandatory = $true)][string]$OutputH5
)

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
python (Join-Path $Root "data_tools\convert_sionna_npz_to_h5.py") `
    --npz $InputNpz `
    --out $OutputH5

if ($LASTEXITCODE -ne 0) {
    throw "Dataset conversion failed with exit code $LASTEXITCODE"
}
