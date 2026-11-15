$ErrorActionPreference = "Stop"

# ============================================================
# OpenESM Nonresponse — Python Environment Setup
# ============================================================

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentFile = Join-Path $ProjectRoot "environment.yml"
$EnvironmentName = "openesm-nonresponse"

Write-Host "`n=== OpenESM Python Setup ===" -ForegroundColor Cyan

# ------------------------------------------------------------
# 1. Find an existing Conda installation
# ------------------------------------------------------------

function Find-Conda {
    $command = Get-Command conda.exe -ErrorAction SilentlyContinue

    if ($command) {
        return $command.Source
    }

    $candidates = @(
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\anaconda3\Scripts\conda.exe",
        "$env:LOCALAPPDATA\miniforge3\Scripts\conda.exe",
        "C:\ProgramData\anaconda3\Scripts\conda.exe",
        "C:\ProgramData\miniconda3\Scripts\conda.exe"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    return $null
}

$Conda = Find-Conda

# ------------------------------------------------------------
# 2. Install Anaconda if Conda is unavailable
# ------------------------------------------------------------

if (-not $Conda) {
    Write-Host "Conda not found. Installing Anaconda..." `
        -ForegroundColor Yellow

    $Installer = Join-Path $env:TEMP "Anaconda3-Installer.exe"
    $InstallerUrl = "https://repo.anaconda.com/archive/Anaconda3-2025.06-1-Windows-x86_64.exe"

    Invoke-WebRequest -Uri $InstallerUrl -OutFile $Installer

    $InstallDir = Join-Path $env:USERPROFILE "anaconda3"

    $Process = Start-Process `
        -FilePath $Installer `
        -ArgumentList @(
            "/InstallationType=JustMe",
            "/RegisterPython=0",
            "/AddToPath=0",
            "/S",
            "/D=$InstallDir"
        ) `
        -Wait `
        -PassThru

    if ($Process.ExitCode -ne 0) {
        throw "Anaconda installation failed with exit code $($Process.ExitCode)."
    }

    Remove-Item $Installer -Force -ErrorAction SilentlyContinue

    $Conda = Join-Path $InstallDir "Scripts\conda.exe"

    if (-not (Test-Path $Conda)) {
        throw "Anaconda installer completed, but Conda was not found at $Conda."
    }
}

Write-Host "Conda: $Conda" -ForegroundColor Green

# ------------------------------------------------------------
# 3. Identify the base environment
# ------------------------------------------------------------

$BaseEnvironment = (& $Conda info --base).Trim()

if ($LASTEXITCODE -ne 0 -or -not $BaseEnvironment) {
    throw "Could not determine the Conda base environment."
}

Write-Host "Base environment: $BaseEnvironment"

# ------------------------------------------------------------
# 4. Ensure Mamba is installed in base
# ------------------------------------------------------------

$Mamba = Join-Path $BaseEnvironment "Scripts\mamba.exe"

if (-not (Test-Path $Mamba)) {
    Write-Host "Mamba not found in base. Installing..." `
        -ForegroundColor Yellow

    & $Conda install --name base --channel conda-forge mamba --yes

    if ($LASTEXITCODE -ne 0) {
        throw "Mamba installation failed."
    }
}

if (-not (Test-Path $Mamba)) {
    throw "Mamba installation could not be verified at $Mamba."
}

& $Mamba --version

# ------------------------------------------------------------
# 5. Verify the project environment file
# ------------------------------------------------------------

if (-not (Test-Path $EnvironmentFile)) {
    throw "Missing environment.yml: $EnvironmentFile"
}

# ------------------------------------------------------------
# 6. Create or update the project environment
# ------------------------------------------------------------

Write-Host "`nCreating/updating project environment..." `
    -ForegroundColor Cyan

& $Mamba env update `
    --name $EnvironmentName `
    --file $EnvironmentFile `
    --prune

if ($LASTEXITCODE -ne 0) {
    throw "Project environment creation/update failed."
}

# ------------------------------------------------------------
# 7. Verify the environment
# ------------------------------------------------------------

& $Conda run --name $EnvironmentName python --version

if ($LASTEXITCODE -ne 0) {
    throw "Python verification failed."
}

Write-Host "`nSetup completed successfully." -ForegroundColor Green
Write-Host "Project environment: $EnvironmentName"
Write-Host "`nActivate it with:"
Write-Host "conda activate $EnvironmentName"
