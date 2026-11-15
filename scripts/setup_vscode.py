from pathlib import Path

content = r'''$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$extensionsFile = Join-Path $repoRoot ".vscode\extensions.json"

if (-not (Get-Command code -ErrorAction SilentlyContinue)) {
    throw "The VS Code 'code' command was not found. Enable the VS Code command in PATH, restart PowerShell, and try again."
}

if (-not (Test-Path $extensionsFile)) {
    throw "Could not find $extensionsFile. Create .vscode\extensions.json first."
}

$config = Get-Content $extensionsFile -Raw | ConvertFrom-Json
$extensions = @($config.recommendations) | Where-Object { $_ }

if ($extensions.Count -eq 0) {
    throw "No extension recommendations were found in $extensionsFile."
}

foreach ($extension in $extensions) {
    Write-Host "`nInstalling $extension ..."
    & code --install-extension $extension

    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Installation failed for $extension. Continuing with the remaining extensions."
    }
}

Write-Host "`nFinished processing the recommended VS Code extensions."
Write-Host "Review any warnings above, then restart VS Code if needed."
'''
path = Path("/mnt/data/setup_vscode.ps1")
path.write_text(content, encoding="utf-8")
print(f"Created {path}")
