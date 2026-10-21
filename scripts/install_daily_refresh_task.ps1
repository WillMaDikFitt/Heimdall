# Register a Windows Task Scheduler job that refreshes the Weather Hub each morning.
# Run once in PowerShell (as your normal user; Admin only if you want "Run whether user is logged on").
#
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_refresh_task.ps1
#
# Optional:  -Time "16:00"   -TaskName "MechatronWeatherHubDailyRefresh"

param(
    [string]$Time = "16:00",
    [string]$TaskName = "MechatronWeatherHubDailyRefresh"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$batPath = Join-Path $PSScriptRoot "run_daily_refresh.bat"

if (-not (Test-Path $batPath)) {
    throw "Missing $batPath"
}

$action = New-ScheduledTaskAction -Execute $batPath -WorkingDirectory $projectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At $Time
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Description "Refresh Mechatron Weather Hub outlook, archive it, and score matured forecasts." `
    -Force | Out-Null

Write-Host "Scheduled task '$TaskName' registered to run daily at $Time."
Write-Host "Project: $projectRoot"
Write-Host "Command: $batPath"
Write-Host ""
Write-Host "To remove later:"
Write-Host "  Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
Write-Host "To run once now:"
Write-Host "  Start-ScheduledTask -TaskName '$TaskName'"
