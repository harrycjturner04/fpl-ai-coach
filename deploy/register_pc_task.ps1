# Registers (or replaces) the daily FPL data refresh on this PC for the current user only (no admin needed).
# Run once: powershell -File deploy\register_pc_task.ps1
# Uses python.exe (not pythonw) so a failed run shows in Task Scheduler's Last Run Result; the console window is brief.
$repo = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source
if ($python -like "*\WindowsApps\*") {
    throw "python resolves to the Microsoft Store stub ($python). Install a real Python or edit `$python in this script to its full path."
}
$user = "$env:USERDOMAIN\$env:USERNAME"
$action = New-ScheduledTaskAction -Execute $python -Argument "-m orchestration.job refresh" -WorkingDirectory $repo
$triggers = @((New-ScheduledTaskTrigger -Daily -At 05:00), (New-ScheduledTaskTrigger -AtLogOn -User $user))
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "FPL Coach refresh" -Action $action -Trigger $triggers -Principal $principal -Settings $settings -Force | Out-Null
Get-ScheduledTask -TaskName "FPL Coach refresh" | Select-Object TaskName, State
(Get-ScheduledTaskInfo -TaskName "FPL Coach refresh").NextRunTime
