# Dot-sourced by up.ps1 / down.ps1 / run.ps1 (all run from canopy/): one place that knows the compose files.
# CVAT's own compose files are one level up (this folder lives inside the CVAT fork).
$env:COMPOSE_PROJECT_NAME = "canopy-cvat"
function Compose {
    # compose writes progress to stderr; Windows PowerShell would turn that into terminating errors under "Stop",
    # so judge success by the exit code only
    $ErrorActionPreference = "Continue"
    docker compose -p canopy-cvat `
        -f ../docker-compose.yml `
        -f ../components/serverless/docker-compose.serverless.yml `
        -f compose.canopy.yml @args 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE) { throw "docker compose $args failed ($LASTEXITCODE)" }
}
