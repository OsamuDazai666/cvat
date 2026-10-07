# Disposable tool box: nuctl (deploys the Nuclio functions through the host docker socket) + cvat-sdk labeler.
FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/*
ARG NUCTL_VERSION=1.16.3
RUN curl -fsSL -o /usr/local/bin/nuctl \
      https://github.com/nuclio/nuclio/releases/download/${NUCTL_VERSION}/nuctl-${NUCTL_VERSION}-linux-amd64 \
    && chmod +x /usr/local/bin/nuctl
RUN pip install cvat-sdk==2.78.0 "numpy<2.5" opencv-python-headless pyyaml pillow
ENV PYTHONPATH=/work/labeler:/work/functions:/work/functions/common
