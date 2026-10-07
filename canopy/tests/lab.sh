#!/usr/bin/env bash
# Run the labeler from Git Bash (tests only; users use run.ps1). MSYS_NO_PATHCONV keeps /work paths intact.
cd /d/canopy-cvat/cvat/canopy && MSYS_NO_PATHCONV=1 docker compose -p canopy-cvat -f ../docker-compose.yml \
  -f ../components/serverless/docker-compose.serverless.yml -f compose.canopy.yml --profile tools \
  run --rm tools python /work/labeler/labeler.py "$@" 2>&1 | grep -vE "Container |Creating|Created"
