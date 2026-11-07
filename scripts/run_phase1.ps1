param(
    [Parameter(Mandatory=$true)][string]$RawDir,
    [string]$OutDir = "data/processed",
    [string]$Python = "python"
)
$ErrorActionPreference = "Stop"
& $Python -m flight_delay.cli discover --raw-dir $RawDir --year 2025 --manifest "$OutDir/audit/input_manifest.json"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m flight_delay.cli preprocess --raw-dir $RawDir --out-dir $OutDir --year 2025 --config configs/preprocessing.yaml
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python -m flight_delay.cli audit --processed-dir $OutDir
exit $LASTEXITCODE
