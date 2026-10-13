# Launches the ten 30-seed capacity-sweep workers detached, one per
# (arm, units) slice, with the same slicing as the 2026-10-12 10:14 run.
# Run this through Win32_Process.Create (see the command at the bottom) so
# the workers are not inside the launching shell's job object.
$repo = Split-Path -Parent $PSScriptRoot
$py = Join-Path (Split-Path -Parent $repo) 'venv\Scripts\python.exe'
$env:PYTHONIOENCODING = 'utf-8'

$workers = @()
foreach ($u in 4, 8, 16, 32) {
    $workers += @{ name = "arm_a-${u}u"; args = "--arm full --units $u --config FULL_HYBRID PURE_ML" }
    $workers += @{ name = "arm_b-${u}u"; args = "--arm interior --units $u --config FULL_HYBRID PURE_ML" }
}
$workers += @{ name = 'arm_b_clean-4u-main'; args = '--arm interior_clean --units 4 --config FULL_HYBRID PURE_ML' }
$workers += @{ name = 'arm_b_clean-4u-ablation'; args = '--arm interior_clean --units 4 --config NO_TEMPORAL NO_STATIC_MECH ONLY_MECH' }

$rows = foreach ($w in $workers) {
    $base = Join-Path $PSScriptRoot "30seed.$($w.name)"
    $p = Start-Process -FilePath $py `
        -ArgumentList "src\02b_run_capacity_sweep.py $($w.args) --n-seeds 30 --log-file `"logs\30seed.$($w.name).log`"" `
        -WorkingDirectory $repo -WindowStyle Hidden `
        -RedirectStandardOutput "$base.out.log" -RedirectStandardError "$base.err.log" -PassThru
    [pscustomobject]@{ worker = $w.name; pid = $p.Id; started = (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'); args = $w.args }
}
$rows | Export-Csv -NoTypeInformation -Path (Join-Path $PSScriptRoot '30seed_worker_pids.csv')

# Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
#     CommandLine = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<repo>\logs\launch_30seed_workers.ps1"' }
