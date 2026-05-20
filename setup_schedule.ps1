# setup_schedule.ps1
# Run once to create a monthly Windows Task Scheduler job.
# The task runs 06_auto_update.py on the 1st of each month at 8:00 AM.

$python   = (Get-Command python).Source
$script   = "C:\Users\Thomas\AI\orlando_gun_violence\06_auto_update.py"
$taskName = "Orlando GVA Monthly Update"

$action  = New-ScheduledTaskAction  -Execute $python -Argument $script
$trigger = New-ScheduledTaskTrigger -Monthly -DaysOfMonth 1 -At "08:00"
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable

Register-ScheduledTask -TaskName $taskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Downloads latest Orlando GVA shooting data and merges it." `
    -Force

Write-Host "Scheduled task '$taskName' created."
Write-Host "Runs: 1st of each month at 08:00 AM"
Write-Host "Log:  C:\Users\Thomas\AI\orlando_gun_violence\output\update_log.txt"
