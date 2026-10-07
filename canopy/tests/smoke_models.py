"""Smoke + parity test for canopy_models (run on Windows: python tests/smoke_models.py).
Vehicle frame: /propose vs the stored SAM 3 proposals for the same frame (curated/autolabel/proposals.jsonl),
/sam3/visual from a click, /sam3/text. Plate items: /plate/detect vs plates_v1 boxes."""
import base64, json, sqlite3, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:9090"
ROOT = Path("D:/canopy-evidence")


def post(path, body):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read()), time.time() - t


def b64(p):
    return base64.b64encode(Path(p).read_bytes()).decode()


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0


con = sqlite3.connect(f"file:{ROOT/'curated/index.sqlite'}?mode=ro", uri=True)
stored = {}
with open(ROOT / "curated/autolabel/proposals.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        if len(d["boxes"]) >= 3:
            stored[d["item"]] = d
        if len(stored) >= int(sys.argv[1] if len(sys.argv) > 1 else 3):
            break
for iid, d in stored.items():
    rel = con.execute("SELECT img_path FROM items WHERE id=?", (iid,)).fetchone()[0]
    img = b64(ROOT / "training" / rel)
    res, dt = post("/propose", {"image": img, "tiles": True, "min_score": 0.0})
    ref = [b for b in d["boxes"] if "legacy" not in b["origins"] or len(b["origins"]) > 1]
    hit = sum(any(iou(r["bbox"], b["bbox"]) >= 0.9 for b in res["boxes"]) for r in ref)
    print(f"{iid}: propose {len(res['boxes'])} boxes in {dt:.1f}s; stored SAM boxes {len(ref)}, matched IoU>=0.9: {hit}")
    bb = ref[0]["bbox"]
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    v, dt = post("/sam3/visual", {"image": img, "points": [[cx, cy]], "labels": [1]})
    m = v["masks"][0] if v["masks"] else None
    print(f"  click at box centre -> {len(v['masks'])} mask, bounds {m['rle'][-4:] if m else None} vs box {[round(x) for x in bb]} ({dt:.2f}s)")
    v, dt = post("/sam3/visual", {"image": img, "bbox": bb})
    print(f"  box prompt -> bounds {v['masks'][0]['rle'][-4:] if v['masks'] else None} ({dt:.2f}s, embedding cached)")
    t, dt = post("/sam3/text", {"image": img, "prompts": ["truck", "car"], "conf": 0.35, "masks": True})
    print(f"  text 'truck','car' -> {len(t['objects'])} objects ({dt:.2f}s)")

rows = con.execute("SELECT l.item_id, i.img_path, l.x1, l.y1, l.x2, l.y2, l.text FROM labels l JOIN items i ON i.id=l.item_id "
                   "WHERE l.source='plates_v1' AND json_extract(l.extra,'$.status')='agreed' AND json_extract(l.extra,'$.target') "
                   "ORDER BY l.item_id LIMIT 20").fetchall()
det_hit = n_extra = 0
for iid, rel, *bb, text in rows:
    img = b64(ROOT / "training" / rel)
    d, _ = post("/plate/detect", {"image": img, "conf": 0.4})
    det_hit += any(iou(bb, x["bbox"]) >= 0.5 for x in d["boxes"])
    n_extra += max(0, len(d["boxes"]) - 1)
print(f"plates: detector found {det_hit}/{len(rows)} plates_v1 boxes; extra boxes on those crops: {n_extra}")
