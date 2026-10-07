# Thin CPU base for the canopy Nuclio functions: they only adapt CVAT's contract and call canopy-models / vLLM.
FROM python:3.12-slim-bookworm
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN pip install "numpy<2.5" opencv-python-headless pyyaml
