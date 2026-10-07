# Stop the canopy CVAT stack.
#   .\down.ps1           stop containers, keep data (CVAT database, annotations)
#   .\down.ps1 -Purge    remove EVERYTHING this project created: containers, volumes (all CVAT data!), the network,
#                        Nuclio function containers and every canopy-cvat/* image. D:\canopy-evidence is untouched.
param([switch]$Purge)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\compose.ps1

# Nuclio function containers are started by the dashboard, not by compose
$fn = docker ps -aq --filter "name=nuclio-nuclio-canopy-"
if ($fn) { docker rm -f $fn | Out-Null }
if ($Purge) {
    Compose --profile tools --profile vlm --profile ocr down -v --remove-orphans
    $img = docker images --format "{{.Repository}}:{{.Tag}}" | Where-Object { $_ -like "canopy-cvat/*" -or $_ -like "nuclio/processor-canopy-*" }
    if ($img) { docker rmi -f $img | Out-Null }
    Write-Host "purged. Remove the checkout too if you are done:  Remove-Item -Recurse D:\canopy-cvat"
} else {
    Compose --profile tools --profile vlm --profile ocr down
}
