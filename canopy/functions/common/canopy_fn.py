"""Helpers shared by the canopy Nuclio functions (mounted at /opt/canopy_fn, on PYTHONPATH).

The functions are thin CPU adapters between CVAT's lambda contract and:
  - canopy-models (SAM 3 + plate ONNX on the GPU, models/server.py), and
  - vLLM (Qwen / GLM-OCR, OpenAI API), through our own reader code in /canopy/tools (mounted read-only).
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(os.environ.get("CANOPY_TOOLS", "/canopy/tools"))
for p in (TOOLS, TOOLS / "autolabel", TOOLS / "plates"):
    if str(p) not in sys.path:
        sys.path.append(str(p))

MODELS_URL = os.environ.get("MODELS_URL", "http://canopy_models:8090")
VLM_URL = os.environ.get("VLM_URL", "http://vllm:8010/v1")
VLM_NAME = os.environ.get("VLM_NAME", "qwen")
OCR_URL = os.environ.get("OCR_URL", "http://vllm_ocr:8010/v1")
OCR_NAME = os.environ.get("OCR_NAME", "glm")


def models(path, body, timeout=300):
    req = urllib.request.Request(MODELS_URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def served(url):
    """Names of the models the vLLM server at `url` is serving right now ([] if it is down)."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=3) as r:
            return [m["id"] for m in json.loads(r.read())["data"]]
    except Exception:
        return []


def decode_image(b64):
    """CVAT sends the frame as base64 (JPEG/PNG) -> BGR uint8, like cv2.imread in the batch tools."""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def pct(p):
    """Model confidence 0..1 -> the integer percent stored in the auto_conf attribute."""
    return int(round(100 * float(p or 0)))


def box(label, bb, conf, attrs=None):
    return {"label": label, "type": "rectangle", "points": [float(v) for v in bb], "confidence": f"{conf:.3f}",
            "attributes": [{"name": k, "value": str(v).lower() if isinstance(v, bool) else str(v)}
                           for k, v in (attrs or {}).items() if v is not None]}


def respond(context, body, status=200):
    return context.Response(body=json.dumps(body), headers={}, content_type="application/json",
                            status_code=status)
