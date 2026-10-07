"""canopy-models: one GPU process that holds SAM 3 and the plate detector for every CVAT function.

Why a server instead of models inside each Nuclio function: SAM 3 would otherwise be loaded once per function, and
the 5090 also has to fit a VLM (Qwen3.8 NVFP4 alone is 21 GB). Models load lazily and POST /unload frees the GPU
so the batch labeler can bring up the big VLM (same swap as tools/autolabel/run.py).

All endpoints take {"image": <base64 jpeg/png>, ...} and return JSON. Coordinates are full-image pixels.
  POST /sam3/visual   points [[x,y]..], labels [1|0..], bbox [x1,y1,x2,y2]  -> {"masks": [{rle, score}]}
  POST /sam3/text     prompts [..], conf, masks (bool)                     -> {"objects": [{bbox, prompt, score, rle?}]}
  POST /propose       tiles (bool), min_score                              -> {"w","h","boxes"}  (tools/autolabel/propose.py)
  POST /plate/detect  conf                                                 -> {"boxes": [{bbox, score}]}
  POST /unload, GET /health
"""
import base64
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

TOOLS = Path(os.environ.get("CANOPY_TOOLS", "/canopy/tools"))
sys.path[:0] = [str(TOOLS), str(TOOLS / "autolabel")]
SAM3 = Path(os.environ.get("SAM3_WEIGHTS", "/models/sam3/sam3.pt"))
DET = Path(os.environ.get("PLATE_DET", "/models/plate_det_v1/plate_rfdetr_nano_canopy_v1.fp32.onnx"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

GPU = threading.Lock()          # one inference at a time: SAM 3 predictors keep per-image state
M = {}                          # lazily loaded models
LAST = {}                       # predictor name -> md5 of the image currently embedded


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def decode(b64):
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def rle(mask):
    """Binary mask -> CVAT mask points: RLE over the bounding-box crop + [left, top, right, bottom]."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    l, t, r, b = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    crop = (mask[t:b + 1, l:r + 1].reshape(-1) != 0).astype(np.uint8)
    changes = np.flatnonzero(crop[1:] != crop[:-1]) + 1
    out = np.diff(np.concatenate(([0], changes, [crop.size]))).tolist()
    if crop[0] == 1:
        out.insert(0, 0)
    return out + [l, t, r, b]


# ---- model loading -------------------------------------------------------------------------------------------------
def semantic():
    if "semantic" not in M:
        import propose
        propose.SAM3_WEIGHTS = SAM3
        # prompts per forward pass: activation memory grows with it (8 in the auto_v1 run); smaller chunks let SAM 3
        # fit next to Qwen. Each text prompt is scored on its own, so the boxes don't depend on the chunking.
        propose.TEXT_CHUNK = int(os.environ.get("PROPOSE_TEXT_CHUNK", propose.TEXT_CHUNK))
        M["semantic"] = propose.load_predictor(0.2, 1008)        # same settings as the auto_v1 proposals
        log("loaded SAM 3 semantic predictor")
    return M["semantic"]


ON = {}                         # SAM 3 predictor name -> "cuda" | "cpu" (where its weights are right now)


def only_on_gpu(name):
    """Keep ONE SAM 3 model on the GPU and park the other in system RAM. With Qwen (72%) and both SAM 3 models
    resident the card sits at 31.6 of 32.6 GB, Windows starts paging GPU memory and Qwen slows to a crawl (vehicle
    clicks ran past CVAT's 60 s timeout). Moving ~2 GB of fp16 weights takes well under a second."""
    for other in ("semantic", "visual"):
        p = M.get(other)
        if p is None or getattr(p, "model", None) is None:          # not loaded / not built yet (lazy first call)
            continue
        dev = "cuda" if other == name else "cpu"
        if ON.get(other, "cuda") != dev:
            if dev == "cpu" and hasattr(p, "features"):
                p.features = None                                   # cached image embedding lives on the GPU
            p.model.to(dev)
            ON[other] = dev
            if dev == "cpu":
                LAST.pop(other, None)
    release()


def visual():
    if "visual" not in M:
        from ultralytics.models.sam import SAM3Predictor
        M["visual"] = SAM3Predictor(overrides=dict(conf=0.0, task="segment", mode="predict", model=str(SAM3),
                                                   half=True, save=False, verbose=False, imgsz=1008))
        log("loaded SAM 3 visual predictor")
    return M["visual"]


def onnx(name, path):
    if name not in M:
        import onnxruntime as ort
        M[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        log(f"loaded {path.name} on {M[name].get_providers()[0]}")
    return M[name]


def set_image(name, pred, img, key):
    if LAST.get(name) != key:       # annotators click many times on one frame: embed it once
        pred.set_image(img)
        LAST[name] = key


# ---- endpoints -----------------------------------------------------------------------------------------------------
def sam3_visual(req, img, key):
    pred = visual()
    only_on_gpu("visual")
    set_image("visual", pred, img, key)
    kw = {}
    pts, labels = req.get("points") or [], req.get("labels") or []
    if pts:
        kw["points"] = [[[float(x), float(y)] for x, y in pts]]
        kw["labels"] = [[int(v) for v in labels]]
    if req.get("bbox"):
        kw["bboxes"] = [[float(v) for v in req["bbox"]]]
    if not kw:
        return {"masks": []}
    r = pred(multimask_output=False, **kw)[0]
    out = []
    if r.masks is not None:
        scores = r.boxes.conf.tolist() if r.boxes is not None else [1.0] * len(r.masks.data)
        for m, s in zip(r.masks.data.cpu().numpy(), scores):
            p = rle(m > 0.5)
            if p:
                out.append({"rle": p, "score": float(s)})
    return {"masks": out}


def sam3_text(req, img, key):
    pred = semantic()
    only_on_gpu("semantic")
    set_image("semantic", pred, img, key)
    prompts = [p for p in req.get("prompts") or [] if p.strip()]
    conf, want_masks = float(req.get("conf", 0.35)), bool(req.get("masks"))
    out = []
    for i in range(0, len(prompts), 8):            # TEXT_CHUNK in propose.py
        chunk = prompts[i:i + 8]
        r = pred(text=chunk)[0]
        if r.boxes is None or not len(r.boxes):
            continue
        masks = r.masks.data.cpu().numpy() if (want_masks and r.masks is not None) else None
        for j, (bb, c, s) in enumerate(zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist())):
            if s < conf:
                continue
            o = {"bbox": [round(v, 1) for v in bb], "prompt": chunk[int(c)], "score": round(float(s), 4)}
            if masks is not None:
                o["rle"] = rle(masks[j] > 0.5)
            out.append(o)
    return {"objects": out}


def propose_ep(req, img, key):
    """tools/autolabel/propose.py:propose_frame unchanged (it reads a path, so the decoded bytes go to a temp file)."""
    import propose
    from prompts import SAM_PROMPTS
    pred = semantic()
    only_on_gpu("semantic")
    LAST.pop("semantic", None)                     # propose_frame calls set_image itself (full frame + tiles)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(cv2.imencode(".png", img)[1].tobytes())
    try:
        w, h, boxes = propose.propose_frame(pred, Path(f.name), [], list(SAM_PROMPTS), bool(req.get("tiles", True)))
    finally:
        os.unlink(f.name)
    ms = float(req.get("min_score", 0.35))         # classify.py --min-score default
    return {"w": w, "h": h, "boxes": [b for b in boxes if b["score"] >= ms]}


def plate_detect(req, img, key):
    sess = onnx("det", DET)
    h, w = img.shape[:2]
    x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = cv2.resize((x - MEAN) / STD, (384, 384), interpolation=cv2.INTER_LINEAR)      # sightline rfdetr_input
    x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32)
    batch = sess.get_inputs()[0].shape[0]
    if isinstance(batch, int) and batch > 1:           # plate_det_v1 is exported with a fixed batch of 4
        x = np.repeat(x, batch, axis=0)
    out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input": x})))
    s = 1 / (1 + np.exp(-np.clip(out["labels"][0][:, 0], -60, 60)))
    conf = float(req.get("conf", 0.4))
    boxes = []
    for (cx, cy, bw, bh), sc in zip(out["dets"][0], s):
        if sc >= conf:
            boxes.append({"bbox": [round(float(max(0, (cx - bw / 2) * w)), 1), round(float(max(0, (cy - bh / 2) * h)), 1),
                                   round(float(min(w, (cx + bw / 2) * w)), 1), round(float(min(h, (cy + bh / 2) * h)), 1)],
                          "score": round(float(sc), 4)})
    # RF-DETR has no NMS and sometimes adds a loose second box around the same plate (0.47 next to 0.95):
    # drop a box that overlaps (IoU >= 0.5) or mostly sits on (>= 0.8 of the smaller box) a better-scored one
    def duplicate(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
        iou = inter / max(1e-6, area_a + area_b - inter)
        return iou >= 0.5 or inter / max(1e-6, min(area_a, area_b)) >= 0.8
    kept = []
    for b in sorted(boxes, key=lambda b: -b["score"]):
        if not any(duplicate(b["bbox"], k["bbox"]) for k in kept):
            kept.append(b)
    return {"boxes": kept}


def release():
    """Hand activation memory back after every request: the torch caching allocator would otherwise keep ~10 GB
    reserved, which is exactly the room Qwen needs to stay loaded next to SAM 3."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def unload(req, img, key):
    import gc
    M.clear()
    LAST.clear()
    ON.clear()
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    log("unloaded all models")
    return {"ok": True}


def warm_up_ep(req, img, key):
    threading.Thread(target=warm_up, kwargs={"wait_for_qwen": bool(req.get("wait_for_qwen"))}, daemon=True).start()
    return {"ok": True, "started": True}


ROUTES = {"/warm_up": warm_up_ep, "/sam3/visual": sam3_visual, "/sam3/text": sam3_text, "/propose": propose_ep,
          "/plate/detect": plate_detect, "/unload": unload}


class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/health"):
            mem = {}
            try:
                import torch
                if torch.cuda.is_available():
                    mem = {"gpu_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
                           "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "gpu_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                           "gpu_now_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}
                    if "reset_peak" in self.path:
                        torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
            return self._send(200, {"ok": True, "loaded": sorted(M), **mem})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        fn = ROUTES.get(self.path)
        if not fn:
            return self._send(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            img = decode(req["image"]) if req.get("image") else None
            key = hashlib.md5(req["image"].encode()).hexdigest() if req.get("image") else None
            t0 = time.time()
            with GPU:
                res = fn(req, img, key)
                release()
            log(f"{self.path} {int(1000 * (time.time() - t0))} ms")
            self._send(200, res)
        except Exception as e:
            log(f"{self.path} failed: {e!r}")
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


def qwen_serving():
    import urllib.request
    try:
        with urllib.request.urlopen(os.environ.get("VLM_URL", "http://vllm:8010/v1") + "/models", timeout=3):
            return True
    except Exception:
        return False


def warm_up(wait_for_qwen=True):
    """Load both SAM 3 models and run one pass each. Cold, the first vehicles click spends ~30 s loading SAM 3
    before Qwen even starts, which runs past CVAT's 60 s gateway timeout (504). Qwen must come up FIRST: vLLM
    refuses to start when it can't get its 72% of the GPU, so at startup this waits (up to 10 min) for Qwen."""
    if wait_for_qwen:
        for _ in range(120):
            if qwen_serving():
                break
            time.sleep(5)
    t0 = time.time()
    img = np.full((720, 1280, 3), 127, np.uint8)
    try:
        with GPU:
            propose_ep({"tiles": True}, img, "warm-up")
            sam3_visual({"points": [[640, 360]], "labels": [1]}, img, "warm-up")
            LAST.clear()
            release()
        log(f"warm-up done in {time.time() - t0:.0f}s: {sorted(M)} loaded")
    except Exception as e:
        log(f"warm-up failed ({e!r}); models load on first use instead")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8090))
    if os.environ.get("WARM_UP", "1") == "1":
        threading.Thread(target=warm_up, daemon=True).start()      # requests queue on the GPU lock meanwhile
    log(f"canopy-models on :{port}")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
