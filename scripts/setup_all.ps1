$ErrorActionPreference = "Stop"

$ScriptsDir = $PSScriptRoot

$SetupScripts = @(
    @{
        Name = "Python environment"
        Path = Join-Path $ScriptsDir "setup_python.ps1"
        Type = "PowerShell"
    },
    @{
        Name = "R environment"
        Path = Join-Path $ScriptsDir "setup_r.R"
        Type = "R"
    },
    @{
        Name = "VS Code"
        Path = Join-Path $ScriptsDir "setup_vscode.ps1"
        Type = "PowerShell"
    }
)

Write-Host "`n====================================" -ForegroundColor Cyan
Write-Host " OpenESM Nonresponse: Full Setup"
Write-Host "====================================`n" -ForegroundColor Cyan

foreach ($Script in $SetupScripts) {
    Write-Host "`n>>> Setting up $($Script.Name)..." -ForegroundColor Yellow

    if (-not (Test-Path $Script.Path)) {
        throw "Setup script not found: $($Script.Path)"
    }

    if ($Script.Type -eq "PowerShell") {
        & $Script.Path

        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
            throw "$($Script.Name) failed with exit code $LASTEXITCODE."
        }
    }
    elseif ($Script.Type -eq "R") {
        $Rscript = Get-Command Rscript.exe -ErrorAction SilentlyContinue

        if (-not $Rscript) {
            throw "Rscript.exe not found. Install R or add it to PATH."
        }

        & $Rscript.Source $Script.Path

        if ($LASTEXITCODE -ne 0) {
            throw "$($Script.Name) failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "Completed: $($Script.Name)" -ForegroundColor Green
}

Write-Host "`n====================================" -ForegroundColor Green
Write-Host " All setup scripts completed!"
Write-Host "====================================`n" -ForegroundColor Green
