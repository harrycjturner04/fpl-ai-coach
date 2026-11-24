# Registers (or replaces) the daily FPL data refresh on this PC. Run once: powershell -File deploy\register_pc_task.ps1
$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
$action = New-ScheduledTaskAction -Execute $python -Argument "-m orchestration.job refresh" -WorkingDirectory $repo
$triggers = @((New-ScheduledTaskTrigger -Daily -At 05:00), (New-ScheduledTaskTrigger -AtLogOn))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable
Register-ScheduledTask -TaskName "FPL Coach refresh" -Action $action -Trigger $triggers -Settings $settings -Force
