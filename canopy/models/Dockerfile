# canopy-models: GPU model server for the CVAT functions. Same stack as the Windows env that ran SAM 3 for
# auto_v1 (torch 2.11 cu128, ultralytics 8.4.115) + onnxruntime (CPU) for the two tiny plate models
# (onnxruntime-gpu >= 1.24 wants CUDA 13, the torch wheels bring CUDA 12; CPU fp32 is the reference anyway).
# Weights and tools/ are bind-mounted read-only (compose.canopy.yml); nothing model-related is baked in.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    YOLO_OFFLINE=1 HF_HUB_OFFLINE=1 YOLO_CONFIG_DIR=/tmp/ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install ultralytics==8.4.115 timm==1.0.30 "numpy<2.5" pyyaml pillow onnxruntime \
        git+https://github.com/ultralytics/CLIP.git \
    && pip uninstall -y opencv-python && pip install opencv-python-headless
COPY server.py /app/server.py
CMD ["python", "/app/server.py"]
