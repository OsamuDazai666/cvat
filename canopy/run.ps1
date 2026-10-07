# Run something in the disposable tools container (removed after each run).
#   .\run.ps1 deploy                      (re)deploy all canopy Nuclio functions
#   .\run.ps1 labeler <command> [args]    see labeler/labeler.py, e.g.  .\run.ps1 labeler seed --task 3 --source auto_v1
#   .\run.ps1 sh                          shell inside the tools container
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\compose.ps1
if (-not (Test-Path .env)) { throw ".env missing (CVAT_USER / CVAT_PASSWORD), see README" }
$what = $args[0]
$rest = @($args | Select-Object -Skip 1)       # always an array, so @rest never splats a string
switch ($what) {
    "deploy"  { Compose --profile tools run --rm tools bash /work/functions/deploy.sh @rest }
    "labeler" { Compose --profile tools run --rm tools python /work/labeler/labeler.py @rest }
    "sh"      { Compose --profile tools run --rm tools bash }
    default   { throw "usage: run.ps1 deploy | labeler <cmd> ... | sh" }
}
