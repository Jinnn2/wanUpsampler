$ErrorActionPreference = "Stop"

$figureDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$htmlPath = Join-Path $figureDir "fig_overall_framework_style_a.html"
$pngPath = Join-Path $figureDir "fig_overall_framework_style_a.png"
$pdfPath = Join-Path $figureDir "fig_overall_framework_style_a.pdf"
$chromePath = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$profilePath = Join-Path $figureDir ".chrome-style-a-profile"

if (-not (Test-Path -LiteralPath $chromePath)) {
    throw "Google Chrome was not found at $chromePath"
}

$htmlUri = ([System.Uri]$htmlPath).AbsoluteUri

if (Test-Path -LiteralPath $profilePath) {
    $resolvedFigureDir = [System.IO.Path]::GetFullPath($figureDir)
    $resolvedProfile = [System.IO.Path]::GetFullPath($profilePath)
    if (-not $resolvedProfile.StartsWith($resolvedFigureDir, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove Chrome profile outside the figure directory: $resolvedProfile"
    }
    Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
}

New-Item -ItemType Directory -Path $profilePath | Out-Null

try {
    $pngArgs = @(
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--window-size=2100,520",
        "--user-data-dir=$profilePath",
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
        "--user-data-dir=$profilePath",
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
    $resolvedFigureDir = [System.IO.Path]::GetFullPath($figureDir)
    $resolvedProfile = [System.IO.Path]::GetFullPath($profilePath)
    if ($resolvedProfile.StartsWith($resolvedFigureDir, [System.StringComparison]::OrdinalIgnoreCase) -and
        (Test-Path -LiteralPath $resolvedProfile)) {
        Remove-Item -LiteralPath $resolvedProfile -Recurse -Force
    }
}

Write-Output "Rendered $pngPath"
Write-Output "Rendered $pdfPath"
