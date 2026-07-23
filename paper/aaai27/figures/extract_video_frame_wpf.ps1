param(
    [Parameter(Mandatory = $true)]
    [string]$InputVideo,

    [Parameter(Mandatory = $true)]
    [string]$OutputPng,

    [ValidateRange(0.0, 1.0)]
    [double]$Fraction = 0.5,

    [switch]$LastFrame,

    [ValidateRange(1, 120)]
    [int]$TimeoutSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

Add-Type -AssemblyName PresentationCore
Add-Type -AssemblyName WindowsBase

$resolvedInput = (Resolve-Path -LiteralPath $InputVideo).Path
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPng)
$outputDirectory = [System.IO.Path]::GetDirectoryName($resolvedOutput)
if (-not [System.IO.Directory]::Exists($outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}

$player = [System.Windows.Media.MediaPlayer]::new()
$player.Volume = 0
$player.ScrubbingEnabled = $true

try {
    $player.Open([System.Uri]::new($resolvedInput))

    $deadline = [DateTime]::UtcNow.AddSeconds($TimeoutSeconds)
    while (
        ($player.NaturalVideoWidth -le 0 -or
         $player.NaturalVideoHeight -le 0 -or
         -not $player.NaturalDuration.HasTimeSpan) -and
        [DateTime]::UtcNow -lt $deadline
    ) {
        $player.Play()
        Start-Sleep -Milliseconds 100
        $player.Pause()
        [System.Windows.Threading.Dispatcher]::CurrentDispatcher.Invoke(
            [Action]{},
            [System.Windows.Threading.DispatcherPriority]::Background
        )
    }

    if (
        $player.NaturalVideoWidth -le 0 -or
        $player.NaturalVideoHeight -le 0 -or
        -not $player.NaturalDuration.HasTimeSpan
    ) {
        throw "Timed out while opening video: $resolvedInput"
    }

    $duration = $player.NaturalDuration.TimeSpan
    if ($LastFrame) {
        # Random seeking can stop at an earlier keyframe. Decode continuously
        # from the beginning and pause inside the display interval of the last
        # frame instead.
        $targetPosition = $duration - [TimeSpan]::FromMilliseconds(20)
        if ($targetPosition -lt [TimeSpan]::Zero) {
            $targetPosition = [TimeSpan]::Zero
        }
        $player.Position = [TimeSpan]::Zero
        $player.Play()
        $playbackDeadline = [DateTime]::UtcNow.AddSeconds(
            [Math]::Max($TimeoutSeconds, [Math]::Ceiling($duration.TotalSeconds) + 5)
        )
        while (
            $player.Position -lt $targetPosition -and
            [DateTime]::UtcNow -lt $playbackDeadline
        ) {
            Start-Sleep -Milliseconds 5
            [System.Windows.Threading.Dispatcher]::CurrentDispatcher.Invoke(
                [Action]{},
                [System.Windows.Threading.DispatcherPriority]::Background
            )
        }
        $player.Pause()
        Start-Sleep -Milliseconds 80
        if ($player.Position -lt $targetPosition) {
            throw "Timed out while decoding the final frame: $resolvedInput"
        }
        $targetTicks = $player.Position.Ticks
    }
    else {
        $targetTicks = [long]($duration.Ticks * $Fraction)
        if ($targetTicks -ge $duration.Ticks) {
            $targetTicks = [Math]::Max(
                0,
                $duration.Ticks - [TimeSpan]::TicksPerMillisecond
            )
        }

        $player.Position = [TimeSpan]::FromTicks($targetTicks)
        $player.Play()
        Start-Sleep -Milliseconds 350
        $player.Pause()
        Start-Sleep -Milliseconds 100
    }

    $width = $player.NaturalVideoWidth
    $height = $player.NaturalVideoHeight
    $visual = [System.Windows.Media.DrawingVisual]::new()
    $drawingContext = $visual.RenderOpen()
    $drawingContext.DrawVideo(
        $player,
        [System.Windows.Rect]::new(0, 0, $width, $height)
    )
    $drawingContext.Close()

    $bitmap = [System.Windows.Media.Imaging.RenderTargetBitmap]::new(
        $width,
        $height,
        96,
        96,
        [System.Windows.Media.PixelFormats]::Pbgra32
    )
    $bitmap.Render($visual)

    $encoder = [System.Windows.Media.Imaging.PngBitmapEncoder]::new()
    $encoder.Frames.Add(
        [System.Windows.Media.Imaging.BitmapFrame]::Create($bitmap)
    )
    $stream = [System.IO.File]::Open(
        $resolvedOutput,
        [System.IO.FileMode]::Create,
        [System.IO.FileAccess]::Write
    )
    try {
        $encoder.Save($stream)
    }
    finally {
        $stream.Dispose()
    }

    [pscustomobject]@{
        Input = $resolvedInput
        Output = $resolvedOutput
        Width = $width
        Height = $height
        DurationSeconds = [Math]::Round($duration.TotalSeconds, 4)
        Fraction = $Fraction
        LastFrame = [bool]$LastFrame
        TimestampSeconds = [Math]::Round(
            $player.Position.TotalSeconds,
            4
        )
    } | ConvertTo-Json -Compress
}
finally {
    $player.Close()
}
