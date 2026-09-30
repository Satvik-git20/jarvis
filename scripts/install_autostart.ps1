<#
.SYNOPSIS
    Installs the JARVIS daemon to start with Windows, plus shortcuts.

.DESCRIPTION
    Creates two things:
      1. A Scheduled Task running the daemon at logon, so it is already up when
         opencode needs it. A scheduled task rather than a Startup-folder
         shortcut because it survives a Terminal window being closed and can be
         configured not to show a window at all.
      2. Start Menu shortcuts for the daemon and the voice loop.

.PARAMETER Remove
    Undo everything this script created.

.EXAMPLE
    pwsh -File scripts/install_autostart.ps1
    pwsh -File scripts/install_autostart.ps1 -Remove
#>

[CmdletBinding()]
param([switch]$Remove)

$ErrorActionPreference = "Stop"

$Repo      = Split-Path -Parent $PSScriptRoot
$TaskName  = "JARVIS daemon"
$StartMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$Python    = Join-Path $Repo ".venv\Scripts\python.exe"

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "  $msg" -ForegroundColor Yellow }

if (-not (Test-Path $Python)) {
    throw "venv not found at $Python - run 'uv sync' first."
}

$shortcuts = @{
    (Join-Path $StartMenu "JARVIS voice.lnk") = @{
        Target  = $Python
        Args    = "-m jarvis run"
        WorkDir = $Repo
        Desc    = "Talk to JARVIS"
    }
    (Join-Path $StartMenu "JARVIS daemon.lnk") = @{
        Target  = $Python
        Args    = "-m jarvis daemon"
        WorkDir = $Repo
        Desc    = "JARVIS API for opencode"
    }
}

if ($Remove) {
    Write-Step "removing..."
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Ok "scheduled task '$TaskName' removed"
    }
    foreach ($lnk in $shortcuts.Keys) {
        if (Test-Path $lnk) { Remove-Item $lnk -Force; Write-Ok "removed $lnk" }
    }
    return
}

Write-Step "installing JARVIS autostart"

# The daemon must not flash a console window at every logon.
$action = New-ScheduledTaskAction -Execute $Python -Argument "-m jarvis daemon" -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
# Delay so the desktop is not competing with Ollama and the daemon at boot.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)
$settings.Hidden = $true

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger `
    -Settings $settings -Principal $principal `
    -Description "JARVIS HTTP API on 127.0.0.1:8765, used by the opencode plugin" | Out-Null
Write-Ok "scheduled task '$TaskName' registered (runs at logon, hidden)"

New-Item -ItemType Directory -Force -Path $StartMenu | Out-Null
$shell = New-Object -ComObject WScript.Shell
foreach ($lnk in $shortcuts.Keys) {
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath       = $shortcuts[$lnk].Target
    $s.Arguments        = $shortcuts[$lnk].Args
    $s.WorkingDirectory = $shortcuts[$lnk].WorkDir
    $s.Description      = $shortcuts[$lnk].Desc
    $s.IconLocation     = "$Python,0"
    $s.Save()
    Write-Ok "shortcut: $(Split-Path $lnk -Leaf)"
}

Write-Host ""
Write-Ok "done. Start the daemon now without waiting for a reboot:"
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor DarkGray
Write-Host ""
Write-Warn2 "The daemon needs JARVIS_DAEMON_TOKEN set in $Repo\.env or it refuses to start."
