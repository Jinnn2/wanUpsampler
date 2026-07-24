$ErrorActionPreference = "Stop"

$figureDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $figureDir "fig_overall_framework_local.html"
$pngPath = Join-Path $figureDir "fig_overall_framework_local.png"
$pdfPath = Join-Path $figureDir "fig_overall_framework_local.pdf"
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$profilePath = Join-Path $figureDir ".chrome-overall-local-profile"

if (-not (Test-Path -LiteralPath $chromePath)) {
    throw "Google Chrome was not found at $chromePath"
}

$resolvedFigureDir = [System.IO.Path]::GetFullPath($figureDir)
$resolvedProfile = [System.IO.Path]::GetFullPath($profilePath)
if (-not $resolvedProfile.StartsWith($resolvedFigureDir, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Chrome profile path is outside the figure directory: $resolvedProfile"
}

if (Test-Path -LiteralPath $resolvedProfile) {
    Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
}
New-Item -ItemType Directory -Path $resolvedProfile | Out-Null

try {
    $htmlUri = ([System.Uri]$htmlPath).AbsoluteUri
    $pngArgs = @(
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--window-size=2200,650",
        "--user-data-dir=$resolvedProfile",
        "--screenshot=$pngPath",
        $htmlUri
    )
    $pngProcess = Start-Process -FilePath $chromePath -ArgumentList $pngArgs `
        -Wait -PassThru -WindowStyle Hidden
    if ($pngProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pngPath)) {
        throw "Chrome PNG rendering failed with exit code $($pngProcess.ExitCode)"
    }

    $pdfArgs = @(
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--user-data-dir=$resolvedProfile",
        "--print-to-pdf=$pdfPath",
        $htmlUri
    )
    $pdfProcess = Start-Process -FilePath $chromePath -ArgumentList $pdfArgs `
        -Wait -PassThru -WindowStyle Hidden
    if ($pdfProcess.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $pdfPath)) {
        throw "Chrome PDF rendering failed with exit code $($pdfProcess.ExitCode)"
    }
}
finally {
    if (Test-Path -LiteralPath $resolvedProfile) {
        Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
    }
}

Write-Output "Rendered $pngPath"
Write-Output "Rendered $pdfPath"
