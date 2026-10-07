#!/usr/bin/env bash
# (Re)deploy the canopy Nuclio functions. Runs inside the tools container (run.ps1 deploy [name ...]).
set -euo pipefail
cd /work/functions
python gen.py                                   # function.yaml from specs.py (classes.json, plate value lists)
NET="${COMPOSE_PROJECT_NAME:-canopy-cvat}_cvat"
nuctl create project cvat --platform local 2>/dev/null || true
FUNCS=("$@")
[ ${#FUNCS[@]} -eq 0 ] && FUNCS=(sam3-interactor canopy-vehicles plate-reader)
for f in "${FUNCS[@]}"; do
  echo "== deploying $f"
  nuctl deploy --project-name cvat --path "/work/functions/$f" --file "/work/functions/$f/function.yaml" \
    --platform local --platform-config "{\"attributes\": {\"network\": \"$NET\"}}"
done
nuctl get function --platform local
