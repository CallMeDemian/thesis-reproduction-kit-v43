# Lay out the E1 scoring run root outside the repository (project root and THESIS_REPRO_RUN_ROOT are the same folder).
# Source: frozen/distribution/THESIS_REPRO_KIT_v2.1.1_FINAL.zip, extracted to ..\runroot_zip.
param(
  [string]$Work = "C:\Users\Demian\Desktop\abl1102",
  [string]$Kit  = "C:\Users\Demian\Desktop\abl1102\thesis-reproduction-kit-v43"
)
$ErrorActionPreference = "Stop"
$zipRoot = Join-Path $Work "runroot_zip\THESIS_REPRO_KIT_v2.1.1_FINAL"
$run = Join-Path $Work "runroot"
if (-not (Test-Path (Join-Path $zipRoot "archive"))) {
  Expand-Archive -LiteralPath (Join-Path $Kit "frozen\distribution\THESIS_REPRO_KIT_v2.1.1_FINAL.zip") -DestinationPath (Join-Path $Work "runroot_zip")
}
New-Item -ItemType Directory -Force $run, (Join-Path $run "03_stage2") | Out-Null
$dep = Join-Path $zipRoot "archive\DEPLOYED_RELEASE"
function Link($path, $target) { if (-not (Test-Path $path)) { New-Item -ItemType Junction -Path $path -Target $target | Out-Null } }
New-Item -ItemType Directory -Force (Join-Path $run "archive") | Out-Null
Link (Join-Path $run "archive\DEPLOYED_RELEASE") $dep
Link (Join-Path $run "03_stage2\input_source") (Join-Path $dep "stage2_candidate_projection")
# 01_contract is a real copy: the financial-cost contract keys its source digests by the archive/... path, but the loader looks them up under 03_stage2/...; the copy gets an alias key with the SAME digest (path remapping only).
$src01 = Join-Path $dep "stage2_candidate_projection\v4_3_runtime\01_contract"
$dst01 = Join-Path $run "03_stage2\01_contract"
if (-not (Test-Path $dst01)) { Copy-Item $src01 $dst01 -Recurse }
$c = Join-Path $dst01 "r085_financial_cost_contract.json"
$j = Get-Content $c -Raw -Encoding utf8 | ConvertFrom-Json
$old = "archive/DEPLOYED_RELEASE/stage2_candidate_projection/v4_3_runtime/01_contract/financial_cost_sources_r2/history_financial_cost_sources.parquet"
$new = "03_stage2/01_contract/financial_cost_sources_r2/history_financial_cost_sources.parquet"
if (-not ($j.source_hashes.PSObject.Properties.Name -contains $new)) {
  $j.source_hashes | Add-Member -NotePropertyName $new -NotePropertyValue $j.source_hashes.$old
  [IO.File]::WriteAllText($c, ($j | ConvertTo-Json -Depth 50), (New-Object Text.UTF8Encoding $false))
}
Link (Join-Path $run "03_stage2\input_splits") (Join-Path $dep "stage2_candidate_projection\input_splits")
Copy-Item (Join-Path $dep "stage2_candidate_projection\phase_eval_candidate.parquet") (Join-Path $run "03_stage2\phase_eval_candidate.parquet") -Force
Copy-Item (Join-Path $dep "stage2_candidate_projection\candidate_action_contract_v4_3.json") (Join-Path $run "03_stage2\candidate_action_contract_v4_3.json") -Force
if (-not (Test-Path (Join-Path $run "contracts"))) { Copy-Item (Join-Path $Kit "contracts") (Join-Path $run "contracts") -Recurse }
"run root ready: $run"
# The kit's fresh-run contract differs from the frozen September one; the scoring path must use the frozen contract
# (sha256 f722e189a265e111c018341254a648ddb9cecc05aa8c12c3547a023ea4c458c0), exposed under the path the simulator reads.
Copy-Item (Join-Path $run "03_stage2\candidate_action_contract_v4_3.json") (Join-Path $run "contracts\scientific\v43_action_contract.json") -Force
