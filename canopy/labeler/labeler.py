"""Batch side of the CVAT integration (runs in the disposable `tools` container, see run.ps1):

  init-projects                         Canopy Vehicles / Canopy Plates / Canopy Text OCR (labels from functions/specs.py)
  create-tasks --project P [--gate G] [--date YYYY-MM-DD] [--limit N] [--size 200]
                                        tasks from curated/index.sqlite items, frames served from the share (no copies)
  seed --task N --source auto_v1|plates_v1   existing auto labels -> CVAT pre-annotations (nothing re-inferred)
  pre-annotate --task N --mode vehicles|plates
                                        full pipeline over a task: vehicles = SAM 3 then Qwen (models swapped in
                                        stages like tools/autolabel/run.py), plates = detector + Qwen.
                                        Every run ends back in the default Qwen mode
  ocr-fill --task N                     read plates (Qwen) / text (GLM) drawn by hand with no reading yet
  pull --task N [--task M ...]          CVAT annotations -> exports/<source>_<ts>.jsonl; apply on Windows with
                                        python labeler/apply_pull.py exports/<file> (index.sqlite stays host-only)

Credentials: CVAT_USER / CVAT_PASSWORD (from .env, see README).
"""
import argparse
import io
import json
import os
import random
import sqlite3
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(os.environ.get("CANOPY_ROOT", "/canopy"))
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "autolabel"), str(ROOT / "tools" / "plates")]
import specs                                                  # noqa: E402  (/work/functions)

CVAT_URL = os.environ.get("CVAT_URL", "http://cvat-server:8080")
MODELS_URL = os.environ.get("MODELS_URL", "http://canopy_models:8090")
VLM_URL = os.environ.get("VLM_URL", "http://vllm:8010/v1")
OCR_URL = os.environ.get("OCR_URL", "http://vllm_ocr:8010/v1")
DB = ROOT / "curated" / "index.sqlite"
EXPORTS = Path("/work/exports")
PROJECTS = {"vehicles": ("Canopy Vehicles", lambda: specs.project("vehicles")),     # all three share one label
            "plates": ("Canopy Plates", lambda: specs.project("plates")),           # set, so every model maps by
            "text": ("Canopy Text OCR", lambda: specs.project("text"))}             # itself in every task


def pct(p):
    """Model confidence 0..1 -> the integer percent stored in the auto_conf attribute."""
    return int(round(100 * float(p or 0)))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def client():
    from cvat_sdk import make_client
    return make_client(CVAT_URL, credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]))


def db():
    """Read-only snapshot of index.sqlite. The curated/ mount is read-only and the DB is in WAL mode, so SQLite
    cannot create its -shm file there; reading a private copy (with any pending -wal) always works."""
    import shutil
    snap = Path("/tmp/index_snapshot.sqlite")
    for suffix in ("", "-wal"):
        src = Path(str(DB) + suffix)
        dst = Path(str(snap) + suffix)
        if src.exists():
            shutil.copyfile(src, dst)
        elif dst.exists():
            dst.unlink()
    return sqlite3.connect(snap)


def http_json(url, body=None, timeout=600):
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# ---- GPU modes (one big model at a time on the 5090) ----------------------------------------------------------------
def docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


def serve(want):
    """want: set of {"qwen", "glm"} that must be up; the rest of the VLMs are stopped first."""
    for name, cont in (("qwen", "canopy_vllm"), ("glm", "canopy_vllm_ocr")):
        if name not in want:
            docker("stop", cont)
    qwen_was_up = bool(docker("ps", "-q", "-f", "name=^canopy_vllm$").stdout.strip())
    if "qwen" in want and not qwen_was_up:
        # vLLM refuses to start unless its 72% of the GPU is free: SAM 3 out of the way first, back in afterwards
        http_json(MODELS_URL + "/unload", {})
    for name, cont, url in (("glm", "canopy_vllm_ocr", OCR_URL), ("qwen", "canopy_vllm", VLM_URL)):
        if name not in want:
            continue
        r = docker("start", cont)
        if r.returncode:
            sys.exit(f"cannot start {cont}: {r.stderr.strip()} (create it with up.ps1 first)")
        log(f"waiting for {cont} ...")
        for _ in range(180):
            try:
                http_json(url.rstrip("/") + "/models", timeout=3)
                break
            except Exception:
                time.sleep(5)
        else:
            sys.exit(f"{cont} did not come up; see `docker logs {cont}`")
    if "qwen" in want and not qwen_was_up:
        http_json(MODELS_URL + "/warm_up", {})             # SAM 3 back in, now that Qwen has its memory
    log("serving:", ", ".join(sorted(want)) or "nothing")


# ---- CVAT helpers -----------------------------------------------------------------------------------------------------
def frames(task):
    """frame index -> (share-relative path, item id)."""
    out = {}
    for i, f in enumerate(task.get_meta().frames):
        name = f.name
        kind = "p" if name.replace("\\", "/").startswith("plates/") else "v"
        out[i] = (name, f"{kind}:{Path(name).stem}")
    return out


def frame_image(task, i):
    return cv2.imdecode(np.frombuffer(task.get_frame(i, quality="original").read(), np.uint8), cv2.IMREAD_COLOR)


def frame_b64(img):
    import base64
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


class Labels:
    """Task label/attribute ids by name, so shapes can be written with spec ids."""

    def __init__(self, task):
        self.ids, self.attrs = {}, {}
        for l in task.get_labels():
            self.ids[l.name] = l.id
            self.attrs[l.name] = {a.name: (a.id, list(a.values), str(getattr(a.input_type, "value", a.input_type)))
                                 for a in l.attributes}     # input_type is an SDK enum

    def shape(self, frame, label, bb, attrs):
        from cvat_sdk.api_client import models
        vals = []
        for k, v in attrs.items():
            spec = self.attrs[label].get(k)
            if spec is None or v is None:
                continue
            aid, allowed, typ = spec
            v = str(v).lower() if isinstance(v, bool) else str(v)
            if typ in ("select", "radio") and v not in allowed:
                v = "unknown" if "unknown" in allowed else allowed[0]
            vals.append(models.AttributeValRequest(spec_id=aid, value=v))
        return models.LabeledShapeRequest(type="rectangle", frame=frame, label_id=self.ids[label],
                                          points=[float(x) for x in bb], attributes=vals, source="auto",
                                          occluded=False, z_order=0)


def put(task, shapes, replace=False):
    from cvat_sdk.api_client import models
    if replace:
        task.remove_annotations()
    task.update_annotations(models.PatchedLabeledDataRequest(shapes=shapes, tracks=[], tags=[]))
    log(f"task {task.id}: +{len(shapes)} shapes")


# ---- commands ---------------------------------------------------------------------------------------------------------
def cmd_init_projects(a):
    with client() as c:
        have = {p.name: p.id for p in c.projects.list()}
        for key, (name, labels) in PROJECTS.items():
            if name in have:                      # existing project: only ADD labels it lacks (never edits/removes)
                p = c.projects.retrieve(have[name])
                present = {l.name: l for l in p.get_labels()}
                missing = [l for l in labels() if l["name"] not in present]
                grown = []                        # existing select attributes: append new values, keep the old ones
                for want in labels():
                    cur = present.get(want["name"])
                    if cur is None:
                        continue
                    attrs = []
                    for wa in want["attributes"]:
                        ca = next((x for x in cur.attributes if x.name == wa["name"]), None)
                        if ca is not None and str(getattr(ca.input_type, "value", ca.input_type)) in ("select", "radio"):
                            new = [v for v in wa["values"] if v not in ca.values]
                            if new:
                                attrs.append({"id": ca.id, "name": ca.name, "input_type": str(getattr(ca.input_type, "value", ca.input_type)),
                                              "mutable": ca.mutable, "default_value": ca.default_value,
                                              "values": list(ca.values) + new})
                    if attrs:
                        grown.append({"id": cur.id, "name": cur.name, "attributes": attrs})
                if missing or grown:
                    p.update({"labels": missing + grown})
                log(f"{name}: exists (id {have[name]})"
                    + (f", added labels {[l['name'] for l in missing]}" if missing else "")
                    + (f", added attribute values on {[l['name'] for l in grown]}" if grown else ""))
                continue
            p = c.projects.create(spec={"name": name, "labels": labels()})
            log(f"{name}: created (id {p.id})")


def cmd_create_tasks(a):
    from cvat_sdk.core.proxies.tasks import ResourceType
    name = PROJECTS[a.project][0]
    kind = "plate" if a.project == "plates" else "vehicle"
    q, args = "SELECT img_path, gate, substr(datetime(ts,'unixepoch'),1,10) FROM items WHERE kind=? AND img_path IS NOT NULL", [kind]
    if a.gate:
        q += " AND gate=?"
        args.append(a.gate)
    if a.date:
        q += " AND date(ts,'unixepoch')=?"
        args.append(a.date)
    if a.items:
        ids = [x.strip() for x in Path(a.items).read_text().split() if x.strip()]
        q += f" AND id IN ({','.join('?' * len(ids))})"
        args += ids
    q += " ORDER BY ts" + (f" LIMIT {int(a.limit)}" if a.limit else "")
    with db() as con:
        rows = con.execute(q, args).fetchall()
    if not rows:
        sys.exit("no items match")
    with client() as c:
        pid = next((p.id for p in c.projects.list() if p.name == name), None) or sys.exit(f"run init-projects first")
        for k in range(0, len(rows), a.size):
            chunk = rows[k:k + a.size]
            tname = f"{a.project} {chunk[0][1]} {chunk[0][2]}..{chunk[-1][2]} #{k // a.size + 1}"
            t = c.tasks.create_from_data(
                spec={"name": tname, "project_id": pid}, resource_type=ResourceType.SHARE,
                resources=[r[0].replace("\\", "/") for r in chunk],
                data_params={"image_quality": 95, "use_cache": True, "sorting_method": "predefined"})
            log(f"task {t.id}: {tname} ({len(chunk)} frames)")


def cmd_seed(a):
    with client() as c, db() as con:
        task = c.tasks.retrieve(a.task)
        L, fr = Labels(task), frames(task)
        shapes, skipped = [], 0
        for i, (_, iid) in fr.items():
            for cls, score, x1, y1, x2, y2, text, extra in con.execute(
                    "SELECT cls, score, x1, y1, x2, y2, text, extra FROM labels WHERE item_id=? AND source=? "
                    "AND x1 IS NOT NULL ORDER BY obj", (iid, a.source)):
                e = json.loads(extra or "{}")
                if a.source == "plates_v1":
                    if e.get("status") == "no_plate":
                        continue
                    attrs = {k: e.get(k) for k in ("number", "code", "origin", "category", "plate_colour", "layout")}
                    attrs.update(legible=e.get("legible") if e.get("legible") != "no_plate" else "none",
                                 ocr_status=e.get("status"), auto_conf=pct(e.get("p_number")), auto_source="plates_v1")
                    label = "plate"
                else:
                    label = cls
                    attrs = {"box_quality": e.get("box_quality"), "auto_conf": pct(score),
                             "needs_review": bool(e.get("doubtful")), "auto_source": a.source}
                if label not in L.ids:
                    skipped += 1
                    continue
                shapes.append(L.shape(i, label, [x1, y1, x2, y2], attrs))
        put(task, shapes, replace=a.replace)
        if skipped:
            log(f"{skipped} boxes skipped (label not in task)")


def _vehicles(task, L, fr, a):
    """SAM 3 proposals for every frame, then swap to Qwen and classify (tools/autolabel/finalize.py:ask)."""
    import finalize
    import prompts as P
    from classify import VLM
    from common import load_classes
    cache = EXPORTS / f"proposals_task{task.id}.jsonl"
    EXPORTS.mkdir(parents=True, exist_ok=True)
    props = {}
    if cache.exists():
        props = {d["frame"]: d for d in map(json.loads, cache.read_text().splitlines()) if d}
    serve(set())
    with open(cache, "a") as f:
        for i in fr:
            if i in props:
                continue
            img = frame_image(task, i)
            p = http_json(MODELS_URL + "/propose", {"image": frame_b64(img), "tiles": True, "min_score": 0.35})
            props[i] = {"frame": i, "boxes": p["boxes"]}
            f.write(json.dumps(props[i]) + "\n")
            log(f"proposals {len(props)}/{len(fr)}")
    with db() as con:
        background = {r[0] for r in con.execute("SELECT id FROM items WHERE trigger='background'")}
    serve({"qwen"})
    cfg, vlm = load_classes(), VLM(VLM_URL, "qwen")
    shapes = []
    for i, (_, iid) in fr.items():
        if iid in background:                     # pure negatives: zero boxes (classify.py rule)
            continue
        img = frame_image(task, i)
        boxes = props[i]["boxes"]
        with ThreadPoolExecutor(32) as ex:
            answers = list(ex.map(lambda kb: finalize.ask(vlm, cfg, img, kb[1]["bbox"], random.Random(kb[0])),
                                  enumerate(boxes)))
        for b, v in zip(boxes, answers):
            if v.get("box_quality") not in P.KEEP_QUALITY:
                continue
            review = (v.get("p_class") or 0) < 0.7 or (v.get("p_quality") or 0) < 0.6 or v["box_quality"] == "several_objects"
            shapes.append(L.shape(i, v["class"], b["bbox"], {"box_quality": v["box_quality"],
                                                              "auto_conf": pct(v.get("p_class")),
                                                              "needs_review": review, "auto_source": "sam3+qwen"}))
        log(f"classified frame {i + 1}/{len(fr)}")
    return shapes


def _plates(task, L, fr, a):
    """plate_det_v1 boxes -> Qwen3.8 reads everything (code, number, origin, category, colour, layout)."""
    from classify import VLM
    from plate_qwen import attrs
    from read import read_vlm
    serve({"qwen"})
    vlm, shapes = VLM(VLM_URL, "qwen38"), []
    for i in fr:
        img = frame_image(task, i)
        dets = http_json(MODELS_URL + "/plate/detect", {"image": frame_b64(img), "conf": 0.4})["boxes"]
        with ThreadPoolExecutor(8) as ex:
            reads = list(ex.map(lambda d: read_vlm(vlm, img, d["bbox"]), dets))
        for d, v in zip(dets, reads):
            if (at := attrs(v)):
                shapes.append(L.shape(i, "plate", d["bbox"], at))
        log(f"plates frame {i + 1}/{len(fr)}")
    return shapes


def cmd_pre_annotate(a):
    with client() as c:
        task = c.tasks.retrieve(a.task)
        L, fr = Labels(task), frames(task)
        try:
            shapes = (_vehicles if a.mode == "vehicles" else _plates)(task, L, fr, a)
            put(task, shapes, replace=a.replace)
        finally:
            serve({"qwen"})                           # back to the default mode (Qwen + SAM 3 click tool)


def cmd_ocr_fill(a):
    """Shapes drawn by hand have no reading yet: plates -> Qwen3.8 (common/plate_qwen.py), text -> GLM-OCR.
    Runs the plates with Qwen loaded first, then (only if there is text to read) swaps to GLM, and always ends
    back in the default Qwen mode."""
    from cvat_sdk.api_client import models
    from classify import VLM
    from plate_qwen import attrs as plate_attrs
    from read import OCR, read_vlm
    from text_ocr import read_text
    with client() as c:
        task = c.tasks.retrieve(a.task)
        L = Labels(task)
        names = {v: k for k, v in L.ids.items()}
        todo = {"plate": [], "text": []}
        for s in task.get_annotations().shapes:
            label = names.get(s.label_id)
            vals = {v.spec_id: v.value for v in s.attributes}
            attr = {n: vals.get(spec[0], "") for n, spec in L.attrs.get(label, {}).items()}
            if attr.get("ocr_status") == "verified":
                continue
            if label == "plate" and (a.all or not attr.get("number")):
                todo["plate"].append(s)
            elif label in ("text", "sign", "vehicle_marking") and (a.all or not attr.get("transcription")):
                todo["text"].append(s)
        log(f"{len(todo['plate'])} plates and {len(todo['text'])} text boxes to read")

        def read_all(shapes, read_one):
            out, by_frame = [], {}
            for s in shapes:
                by_frame.setdefault(s.frame, []).append(s)
            for fi, ss in sorted(by_frame.items()):
                img = frame_image(task, fi)
                with ThreadPoolExecutor(8) as ex:
                    out += list(zip(ss, ex.map(lambda s: read_one(img, s.points[:4]), ss)))
            return out

        def plate_one(img, bb):
            return plate_attrs(read_vlm(vlm, img, bb)) or {"legible": "none", "ocr_status": "unreadable",
                                                           "auto_source": "qwen"}

        def text_one(img, bb):
            t = read_text(ocr, img, bb)
            if not t["plausible"]:                    # output didn't fit the box: leave it to a human
                return {"ocr_status": "unchecked"}
            return {"transcription": t["text"], "script": t["script"],
                    "legible": "full" if t["text"] else "none", "ocr_status": "auto"}

        results = []
        try:
            if todo["plate"]:
                serve({"qwen"})
                vlm = VLM(VLM_URL, "qwen38")
                results += read_all(todo["plate"], plate_one)
            if todo["text"]:
                serve({"glm"})
                ocr = OCR(OCR_URL, "glm", "Text Recognition:")
                results += read_all(todo["text"], text_one)
        finally:
            if todo["text"]:
                serve({"qwen"})
        updated = []
        for s, new in results:
            label = names[s.label_id]
            keep = {v.spec_id: v.value for v in s.attributes}
            for k, v in new.items():
                spec = L.attrs[label].get(k)
                if spec is not None and v is not None:
                    v = str(v)
                    if spec[2] in ("select", "radio") and v not in spec[1]:
                        v = "unknown" if "unknown" in spec[1] else spec[1][0]
                    keep[spec[0]] = v
            updated.append(models.LabeledShapeRequest(
                id=s.id, type=s.type.value, frame=s.frame, label_id=s.label_id, points=s.points,
                occluded=s.occluded, z_order=s.z_order, rotation=s.rotation, source=s.source, group=s.group,
                attributes=[models.AttributeValRequest(spec_id=k, value=v) for k, v in keep.items()]))
        if updated:
            task.update_annotations(models.PatchedLabeledDataRequest(shapes=updated, tracks=[], tags=[]),
                                    action="update")
        log(f"task {task.id}: {len(updated)} shapes filled")


def cmd_pull(a):
    """Annotations -> JSONL rows in the index.sqlite labels format (apply on the host with apply_pull.py)."""
    EXPORTS.mkdir(parents=True, exist_ok=True)
    out = EXPORTS / f"{a.source}_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    n = 0
    with client() as c, open(out, "w", encoding="utf-8") as f:
        for tid in a.task:
            task = c.tasks.retrieve(tid)
            pname = c.projects.retrieve(task.project_id).name if task.project_id else ""
            project = next((k for k, (n, _) in PROJECTS.items() if n == pname), None) or sys.exit(
                f"task {tid} is not in a canopy project")
            L, fr = Labels(task), frames(task)
            names = {v: k for k, v in L.ids.items()}
            per = {iid: [] for _, iid in fr.values()}       # every frame, so empty frames clear old rows
            for s in task.get_annotations().shapes:
                label = names[s.label_id]
                by_id = {spec[0]: n for n, spec in L.attrs[label].items()}
                attrs = {by_id[v.spec_id]: v.value for v in s.attributes if v.spec_id in by_id}
                x1, y1, x2, y2 = s.points[:4] if s.type.value == "rectangle" else (
                    min(s.points[0::2]), min(s.points[1::2]), max(s.points[0::2]), max(s.points[1::2]))
                per[fr[s.frame][1]].append({"cls": label, "bbox": [round(v, 1) for v in (x1, y1, x2, y2)],
                                            "occluded": s.occluded, "cvat_shape": s.id, **attrs})
            for iid, boxes in per.items():
                f.write(json.dumps({"item": iid, "task": tid, "project": project, "boxes": boxes}) + "\n")
                n += 1
    log(f"{n} frames -> {out}  (apply: python labeler/apply_pull.py {out.relative_to('/work')} --source {a.source})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-projects")
    p = sub.add_parser("create-tasks")
    p.add_argument("--project", required=True, choices=list(PROJECTS))
    p.add_argument("--gate")
    p.add_argument("--date")
    p.add_argument("--items", help="file with item ids, one per line")
    p.add_argument("--limit", type=int)
    p.add_argument("--size", type=int, default=200)
    p = sub.add_parser("seed")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--source", required=True, choices=["auto_v1", "plates_v1"])
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("pre-annotate")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--mode", required=True, choices=["vehicles", "plates"])
    p.add_argument("--replace", action="store_true")
    p = sub.add_parser("ocr-fill")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--all", action="store_true", help="re-read every unverified shape, not only empty ones")
    p = sub.add_parser("pull")
    p.add_argument("--task", type=int, action="append", required=True)
    p.add_argument("--source", default="cvat_v1")
    a = ap.parse_args()
    {"init-projects": cmd_init_projects, "create-tasks": cmd_create_tasks, "seed": cmd_seed,
     "pre-annotate": cmd_pre_annotate, "ocr-fill": cmd_ocr_fill, "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    main()
