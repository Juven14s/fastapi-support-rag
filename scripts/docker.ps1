$DockerArgs = $args
$DockerCommand = Get-Command docker -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1

if ($DockerCommand) {
    $DockerPath = $DockerCommand.Source
} else {
    $DockerPath = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
    if (-not (Test-Path -LiteralPath $DockerPath -PathType Leaf)) {
        throw "Client Docker introuvable. Vérifie le PATH ou l’installation de Docker Desktop."
    }
}

& $DockerPath @DockerArgs
exit $LASTEXITCODE
