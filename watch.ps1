$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$status = ""
do {
    $status = kaggle kernels status mugishaalainpaisible/codegen-train-v1 2>&1 | Out-String
    Write-Host $status.Trim() -ForegroundColor Yellow
    if ($status -notmatch "KernelWorkerStatus\.(RUNNING|QUEUED|ERROR|COMPLETE)") {
        Write-Host "(transient/network hiccup, retrying...)" -ForegroundColor DarkGray
    }
    Start-Sleep -Seconds 15
} while ($status -notmatch "KernelWorkerStatus\.(ERROR|COMPLETE)")

if ($status -match "KernelWorkerStatus\.ERROR") {
    Write-Host "`nFAILED" -ForegroundColor Red
    Remove-Item kaggle_output\codegen-train-v1.log -Force -ErrorAction SilentlyContinue
    kaggle kernels output mugishaalainpaisible/codegen-train-v1 -p kaggle_output 2>&1 | Out-String -Width 4096
    if (Test-Path kaggle_output\codegen-train-v1.log) {
        Write-Host "`n--- Error section ---" -ForegroundColor Red
        Get-Content kaggle_output\codegen-train-v1.log -Encoding UTF8 | Select-String -Pattern "Traceback|Error" -Context 0,10
    } else {
        Write-Host "Log file was not created - output download failed silently." -ForegroundColor Magenta
    }
} else {
    Write-Host "`nSUCCESS" -ForegroundColor Green
}
