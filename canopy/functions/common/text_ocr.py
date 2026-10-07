"""General-text OCR helpers shared by the text-reader function and labeler ocr-fill."""
import re

import cv2

ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN = re.compile(r"[A-Za-z0-9]")
FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
MIN_W, MIN_H = 24, 12          # px; GLM invents text (a whole paragraph once) for regions smaller than this


def script(t):
    a, l = bool(ARABIC.search(t)), bool(LATIN.search(t))
    return "mixed" if a and l else "arabic" if a else "latin" if l else ("other" if t.strip() else "unknown")


def clamp(bb, w, h):
    return [max(0.0, min(float(w), bb[0])), max(0.0, min(float(h), bb[1])),
            max(0.0, min(float(w), bb[2])), max(0.0, min(float(h), bb[3]))]


def readable_size(bb):
    return bb[2] - bb[0] >= MIN_W and bb[3] - bb[1] >= MIN_H


def crop(img, bb, pad=0.08, min_w=400):
    """Text region + a small margin; small text is upscaled like tools/plates/read.py:plate_views does."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bb
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    c = img[int(max(0, y1 - py)):int(min(h, y2 + py)), int(max(0, x1 - px)):int(min(w, x2 + px))]
    if c.size and c.shape[1] < min_w:
        k = min_w / c.shape[1]
        c = cv2.resize(c, (min_w, max(1, round(c.shape[0] * k))), interpolation=cv2.INTER_CUBIC)
    return c


def clean(t, bb):
    """GLM-OCR output -> transcription. Empty markdown fences mean 'no text'; text far longer than the region can
    hold (chars per line ~ 3x the aspect ratio) is a hallucination and is dropped (plausible=False)."""
    t = FENCE.sub("", (t or "").strip()).strip()
    w, h = bb[2] - bb[0], max(1.0, bb[3] - bb[1])
    lines = max(1, round(h / max(12.0, min(h, w / 2))))      # crude: tall narrow boxes may hold a few lines
    limit = max(12, int(3 * w / h * lines) + 8)
    plausible = len(t.replace("\n", "")) <= limit and t.count("\n") < 6
    return (t if plausible else ""), plausible


def read_text(ocr, img, bb):
    """ocr: tools/plates/read.py OCR client (GLM-OCR, prompt 'Text Recognition:')."""
    from read import b64
    bb = clamp(bb, img.shape[1], img.shape[0])
    if not readable_size(bb):
        return {"text": "", "script": "unknown", "p": None, "plausible": True, "too_small": True}
    c = crop(img, bb)
    if not c.size:
        return {"text": "", "script": "unknown", "p": None, "plausible": True, "too_small": True}
    raw, p = ocr.text(b64(c), max_tokens=128)
    t, ok = clean(raw, bb)
    return {"text": t, "script": script(t), "p": p, "plausible": ok, "too_small": False}
