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
        Where-Object { $_.FullName -ne $Output } |
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

$Outputs = @(
    (Join-Path $PackageRoot "AAAI27_InTraScale_Supplementary_Document.pdf"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Reproducibility_Checklist.pdf"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Code_and_Data.zip"),
    (Join-Path $PackageRoot "AAAI27_InTraScale_Supplementary_Media.zip"),
    (Join-Path $PackageRoot "SHA256SUMS.txt")
)
foreach ($path in $Outputs) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

Copy-Item -LiteralPath (Join-Path $SupplementRoot "main.pdf") `
    -Destination $Outputs[0]
Copy-Item -LiteralPath (Join-Path $ChecklistRoot "ReproducibilityChecklist.pdf") `
    -Destination $Outputs[1]

Compress-Archive -Path (Join-Path $CodeDataRoot "*") `
    -DestinationPath $Outputs[2] -CompressionLevel Optimal
Compress-Archive -Path (Join-Path $MediaRoot "*") `
    -DestinationPath $Outputs[3] -CompressionLevel Optimal

$HashLines = $Outputs[0..3] | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $hash = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($item.Name)"
}
[System.IO.File]::WriteAllLines($Outputs[4], $HashLines, [System.Text.UTF8Encoding]::new($false))

$RewriteAfter = Get-TreeSnapshot -Root $RewriteRoot
if ($RewriteBefore -ne $RewriteAfter) {
    throw "The protected rewrite directory changed during package construction."
}

Write-Host "Built upload artifacts:"
Get-ChildItem -LiteralPath $PackageRoot -File |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
Write-Host "Protected rewrite directory: unchanged."

