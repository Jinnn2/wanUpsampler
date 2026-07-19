param(
    [switch]$SkipSupplement
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$build = Join-Path $root 'build'
if (-not (Test-Path $build)) {
    New-Item -ItemType Directory -Path $build | Out-Null
}

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
if (-not $pdflatex -or -not $bibtex) {
    $tinytex = Join-Path $env:LOCALAPPDATA 'TinyTeX\bin\windows'
    $pdflatexPath = Join-Path $tinytex 'pdflatex.exe'
    $bibtexPath = Join-Path $tinytex 'bibtex.exe'
    if (-not (Test-Path $pdflatexPath) -or -not (Test-Path $bibtexPath)) {
        throw 'PDFLaTeX/BibTeX not found. Install TeX Live or TinyTeX first.'
    }
} else {
    $pdflatexPath = $pdflatex.Source
    $bibtexPath = $bibtex.Source
}

Push-Location $root
try {
    & $pdflatexPath -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$build" main.tex
    if ($LASTEXITCODE -ne 0) { throw 'Initial PDFLaTeX pass failed.' }
    & $bibtexPath build/main
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    1..2 | ForEach-Object {
        & $pdflatexPath -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$build" main.tex
        if ($LASTEXITCODE -ne 0) { throw "PDFLaTeX pass $_ failed." }
    }
    Copy-Item -Force (Join-Path $build 'main.pdf') (Join-Path $root 'main.pdf')

    if (-not $SkipSupplement) {
        1..2 | ForEach-Object {
            & $pdflatexPath -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$build" supplementary.tex
            if ($LASTEXITCODE -ne 0) { throw "Supplement PDFLaTeX pass $_ failed." }
        }
        Copy-Item -Force (Join-Path $build 'supplementary.pdf') (Join-Path $root 'supplementary.pdf')
    }
} finally {
    Pop-Location
}

Write-Host "Main PDF: $(Join-Path $root 'main.pdf')"
if (-not $SkipSupplement) {
    Write-Host "Supplement PDF: $(Join-Path $root 'supplementary.pdf')"
}
