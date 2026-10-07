# Start the canopy CVAT stack (local test). Everything lives in compose project "canopy-cvat".
#   .\up.ps1                    CVAT + Nuclio + canopy_models + Qwen3.8 (default: plates, SAM 3 click tool)
#   .\up.ps1 -Mode none         no VLM loaded
#   .\up.ps1 -Build             rebuild canopy images (models server, function base, tools)
# Batch runs (labeler pre-annotate / ocr-fill) switch modes themselves and always end back in the default.
param([ValidateSet("qwen", "none")][string]$Mode = "qwen", [switch]$Build)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
. .\compose.ps1

if ($Build) {
    docker build -t canopy-cvat/fn-base:latest functions/base
    Compose --profile tools --profile vlm --profile ocr build
}
# create every service (VLM containers too, stopped) so the labeler can start/stop them by name
Compose --profile vlm --profile ocr up --no-start
Compose up -d
function Unload { try { Invoke-RestMethod -Method Post -Uri http://127.0.0.1:9090/unload -Body "{}" -ContentType application/json | Out-Null } catch {} }
switch ($Mode) {
    # Qwen3.8 at 72% of the GPU + the SAM 3 click model fit together (31 of 32.6 GB, measured)
    "qwen" { docker stop canopy_vllm_ocr 2>$null | Out-Null; Unload; docker start canopy_vllm | Out-Null }
    "none" { docker stop canopy_vllm canopy_vllm_ocr 2>$null | Out-Null }
}
Write-Host "CVAT: http://localhost:9080   Nuclio: http://localhost:9070   models: http://localhost:9090/health"
Write-Host "Qwen takes ~5 min to load after a start; check with  docker logs -f canopy_vllm"
