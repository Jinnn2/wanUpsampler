$ErrorActionPreference = "Stop"

$Here = (Resolve-Path $PSScriptRoot).Path
$RepoRoot = (Resolve-Path (Join-Path $Here "..\..\..")).Path
$RewriteRoot = (Resolve-Path (Join-Path $Here "..\rewrite")).Path
$SupplementRoot = Join-Path $Here "supplementary_document"
$ChecklistRoot = Join-Path $Here "reproducibility_checklist"
$CodeDataRoot = Join-Path $Here "code_data_package"
$MediaRoot = Join-Path $Here "media_archive"
$PackageRoot = Join-Path $Here "packages"

if (-not $Here.StartsWith($RepoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Submission workspace is outside the expected repository."
}
if (-not $PackageRoot.StartsWith($Here, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Package output path escaped the submission workspace."
}

function Get-TreeSnapshot {
    param([Parameter(Mandatory = $true)][string]$Root)
    $items = Get-ChildItem -LiteralPath $Root -Recurse -File |
        Sort-Object FullName |
        ForEach-Object {
            [PSCustomObject]@{
                Path = $_.FullName.Substring($Root.Length).TrimStart("\")
                Size = $_.Length
                SHA256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
            }
        }
    return ($items | ConvertTo-Json -Compress)
}

function Write-TreeManifest {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Output
    )
    $lines = Get-ChildItem -LiteralPath $Root -Recurse -File |
        Where-Object {
            $_.FullName -ne $Output -and
            $_.Extension -ne ".pyc" -and
            $_.FullName -notmatch "[\\/]__pycache__[\\/]"
        } |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($Root.Length).TrimStart("\").Replace("\", "/")
            $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
            "$hash  $relative"
        }
    [System.IO.File]::WriteAllLines($Output, $lines, [System.Text.UTF8Encoding]::new($false))
}

function Invoke-LatexBuild {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$MainFile
    )
    Push-Location $Root
    try {
        & latexmk -pdf -interaction=nonstopmode -halt-on-error $MainFile
        if ($LASTEXITCODE -ne 0) {
            throw "LaTeX build failed for $MainFile"
        }
    }
    finally {
        Pop-Location
    }
}

function Assert-PdfFonts {
    param([Parameter(Mandatory = $true)][string]$Path)
    $fontReport = (& pdffonts $Path 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        throw "pdffonts failed for $Path"
    }
    if ($fontReport -match "Type 3|Identity-H|CID (TrueType|Type 0)|\sno\s+(yes|no)\s*$") {
        throw "Disallowed or unembedded font detected in $Path`n$fontReport"
    }
}

function Assert-ZipPaths {
    param([Parameter(Mandatory = $true)][string]$Path)
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($Path)
    try {
        foreach ($entry in $archive.Entries) {
            $name = $entry.FullName.Replace("\", "/")
            if ($name.StartsWith("/") -or $name -match "(^|/)\.\.(/|$)" -or $name -match "^[A-Za-z]:") {
                throw "Unsafe ZIP member path in $Path`: $name"
            }
        }
    }
    finally {
        $archive.Dispose()
    }
}

$RewriteBefore = Get-TreeSnapshot -Root $RewriteRoot

New-Item -ItemType Directory -Force -Path $PackageRoot | Out-Null

Invoke-LatexBuild -Root $SupplementRoot -MainFile "main.tex"
Invoke-LatexBuild -Root $ChecklistRoot -MainFile "ReproducibilityChecklist.tex"

$SupplementLog = Get-Content -Raw (Join-Path $SupplementRoot "main.log")
$ChecklistLog = Get-Content -Raw (Join-Path $ChecklistRoot "ReproducibilityChecklist.log")
foreach ($entry in @(
    [PSCustomObject]@{ Name = "supplement"; Log = $SupplementLog },
    [PSCustomObject]@{ Name = "checklist"; Log = $ChecklistLog }
)) {
    if ($entry.Log -match "undefined references|undefined citations|Citation .* undefined") {
        throw "$($entry.Name) PDF has unresolved references or citations."
    }
    if ($entry.Log -match "Overfull \\hbox|Overfull \\vbox") {
        Write-Warning "$($entry.Name) PDF contains an overfull box; inspect the PDF."
    }
}

$ExpectedVideos = @(
    "prompt_00_seed9700\Trilinear_at_40.mp4",
    "prompt_00_seed9700\ITU_only_at_40.mp4",
    "prompt_00_seed9700\InTraScale_at_40.mp4",
    "prompt_07_seed9707\Trilinear_at_45.mp4",
    "prompt_07_seed9707\ITU_only_at_45.mp4",
    "prompt_07_seed9707\InTraScale_at_45.mp4"
)
foreach ($relative in $ExpectedVideos) {
    $path = Join-Path $MediaRoot $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing media file: $relative"
    }
}
$ActualVideoCount = @(Get-ChildItem -LiteralPath $MediaRoot -Recurse -File -Filter *.mp4).Count
if ($ActualVideoCount -ne $ExpectedVideos.Count) {
    throw "Expected $($ExpectedVideos.Count) media files, found $ActualVideoCount."
}

$ChecklistText = Get-Content -Raw (Join-Path $ChecklistRoot "ReproducibilityChecklist.tex")
$PlaceholderCount = ([regex]::Matches($ChecklistText, "Type your response here")).Count
if ($PlaceholderCount -ne 3) {
    throw "Unexpected checklist placeholder count: $PlaceholderCount. Expected 3 template examples only."
}
$QuestionCount = ([regex]::Matches($ChecklistText, "\\question\{")).Count
if ($QuestionCount -ne 31) {
    throw "Unexpected checklist question count: $QuestionCount. Expected all 31 AAAI-27 questions."
}

$IdentityPattern = "houze|jinho|C:\\Users|/mnt/afs|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
$ScanOutput = & rg -n $IdentityPattern $CodeDataRoot $MediaRoot -g "!*.mp4" 2>$null
if ($LASTEXITCODE -eq 0) {
    $ScanOutput | ForEach-Object { Write-Error $_ }
    throw "Potential identity or machine-specific path found in an upload package."
}
if ($LASTEXITCODE -gt 1) {
    throw "Anonymity scan failed with rg exit code $LASTEXITCODE."
}

Write-TreeManifest -Root $CodeDataRoot -Output (Join-Path $CodeDataRoot "FILE_MANIFEST_SHA256.txt")
Write-TreeManifest -Root $MediaRoot -Output (Join-Path $MediaRoot "FILE_MANIFEST_SHA256.txt")

$env:PYTHONDONTWRITEBYTECODE = "1"
& python (Join-Path $CodeDataRoot "tools\check_source_closure.py")
if ($LASTEXITCODE -ne 0) {
    throw "Code-and-data source snapshot has missing internal imports."
}

$Outputs = @(
    (Join-Path $PackageRoot "AAAI27_InTraScale_Main_Paper.pdf"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Reproducibility_Checklist.pdf"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Technical_Supplement.pdf"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Media_Supplement.zip"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Code_and_Data_Supplement.zip"),
    (Join-Path $PackageRoot "SHA256SUMS.txt")
)
Get-ChildItem -LiteralPath $PackageRoot -File | ForEach-Object {
    Remove-Item -LiteralPath $_.FullName -Force
}

Copy-Item -LiteralPath (Join-Path $RewriteRoot "main.pdf") `
    -Destination $Outputs[0]
Copy-Item -LiteralPath (Join-Path $ChecklistRoot "ReproducibilityChecklist.pdf") `
    -Destination $Outputs[1]
Copy-Item -LiteralPath (Join-Path $SupplementRoot "main.pdf") `
    -Destination $Outputs[2]

Compress-Archive -Path (Join-Path $MediaRoot "*") `
    -DestinationPath $Outputs[3] -CompressionLevel Optimal

$CodeStage = Join-Path $Here ".package_stage_code_data"
if (-not $CodeStage.StartsWith($Here, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe code staging path: $CodeStage"
}
if (Test-Path -LiteralPath $CodeStage) {
    Remove-Item -LiteralPath $CodeStage -Recurse -Force
}
New-Item -ItemType Directory -Path $CodeStage | Out-Null
try {
    Get-ChildItem -LiteralPath $CodeDataRoot -Recurse -File |
        Where-Object {
            $_.Extension -ne ".pyc" -and
            $_.FullName -notmatch "[\\/]__pycache__[\\/]"
        } |
        ForEach-Object {
            $relative = $_.FullName.Substring($CodeDataRoot.Length).TrimStart("\")
            $target = Join-Path $CodeStage $relative
            $parent = Split-Path -Parent $target
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
            Copy-Item -LiteralPath $_.FullName -Destination $target
        }
    Compress-Archive -Path (Join-Path $CodeStage "*") `
        -DestinationPath $Outputs[4] -CompressionLevel Optimal
}
finally {
    if (Test-Path -LiteralPath $CodeStage) {
        Remove-Item -LiteralPath $CodeStage -Recurse -Force
    }
}

Assert-PdfFonts -Path $Outputs[0]
Assert-PdfFonts -Path $Outputs[1]
Assert-PdfFonts -Path $Outputs[2]
Assert-ZipPaths -Path $Outputs[3]
Assert-ZipPaths -Path $Outputs[4]

$Limits = @{
    $Outputs[2] = 10000000
    $Outputs[3] = 50000000
    $Outputs[4] = 50000000
}
foreach ($entry in $Limits.GetEnumerator()) {
    $size = (Get-Item -LiteralPath $entry.Key).Length
    if ($size -gt $entry.Value) {
        throw "Upload limit exceeded: $($entry.Key) is $size bytes (limit $($entry.Value))."
    }
}

$HashLines = $Outputs[0..4] | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($item.Name)"
}
[System.IO.File]::WriteAllLines($Outputs[5], $HashLines, [System.Text.UTF8Encoding]::new($false))

$RewriteAfter = Get-TreeSnapshot -Root $RewriteRoot
if ($RewriteBefore -ne $RewriteAfter) {
    throw "The protected rewrite directory changed during package construction."
}

Write-Host "Built upload artifacts:"
Get-ChildItem -LiteralPath $PackageRoot -File |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "Protected rewrite directory: unchanged."
