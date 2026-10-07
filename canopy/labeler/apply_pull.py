"""Apply a `labeler pull` export to D:\\canopy-evidence\\curated\\index.sqlite (run on Windows, like the other tools,
so the SQLite file is never written through a container bind mount).

    python labeler/apply_pull.py exports/cvat_v1_<ts>.jsonl [--source cvat_v1] [--dry-run]

Rows go to three label sources, matching the existing ones:
  <source>        vehicle classes, via tools/import_labels.py:import_records (same as auto_v1; disagreements vs legacy)
  <source>_plates plate boxes with code/number/origin/... in `extra`, text = code + number (same as plates_v1)
  <source>_text   text / sign / vehicle_marking boxes, text = transcription
All projects share one label set. A task replaces a source for a frame when the source belongs to its project (so
boxes deleted in CVAT disappear) or when the frame has boxes of that kind in the task; other rows are left alone.
"""
import argparse
import json
import sys
import time
from pathlib import Path

TOOLS = Path("D:/canopy-evidence/tools")
sys.path.insert(0, str(TOOLS))
from common import connect, load_classes            # noqa: E402
from import_labels import import_records             # noqa: E402

TEXT = {"text", "sign", "vehicle_marking"}
NUM = {"auto_conf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--source", default="cvat_v1")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    vehicle = {c["name"] for c in load_classes()["classes"]}
    rows = [json.loads(l) for l in Path(a.jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    veh, n_plate, n_text, unknown = [], 0, 0, {}
    con = connect()
    for r in rows:
        boxes, proj, iid = r["boxes"], r["project"], r["item"]
        for b in boxes:
            for k in NUM & b.keys():
                b[k] = float(b[k]) / 100 if b[k] not in ("", None) else None   # CVAT stores percent
            if b["cls"] not in vehicle | TEXT | {"plate"}:
                unknown[b["cls"]] = unknown.get(b["cls"], 0) + 1
        vb = [b for b in boxes if b["cls"] in vehicle]
        pb = [b for b in boxes if b["cls"] == "plate"]
        tb = [b for b in boxes if b["cls"] in TEXT]
        # Every project has every label, and one frame can sit in tasks of several projects. A task REPLACES a
        # source for a frame when the source belongs to its project (so boxes deleted in CVAT disappear) or when it
        # has boxes of that kind; otherwise it leaves the rows another project's task wrote alone.
        if iid.startswith("v:") and (proj == "vehicles" or vb):
            veh.append({"item": iid, "boxes": [{**b, "score": b.get("auto_conf")} for b in vb]})
        elif vb:
            unknown["vehicle box on a plate crop (not stored)"] = unknown.get("vehicle box on a plate crop (not stored)", 0) + len(vb)
        for suffix, bs, owner in (("_plates", pb, "plates"), ("_text", tb, "text")):
            if proj != owner and not bs:
                continue
            if suffix == "_plates":
                n_plate += len(bs)
            else:
                n_text += len(bs)
            if a.dry_run:
                continue
            src = a.source + suffix
            con.execute("DELETE FROM labels WHERE item_id=? AND source=?", (iid, src))
            for k, b in enumerate(bs):
                if b["cls"] == "plate":
                    text, skip = (b.get("code") or "") + (b.get("number") or ""), ("cls", "bbox")
                else:
                    text, skip = b.get("transcription"), ("cls", "bbox", "transcription")
                extra = {x: b[x] for x in b if x not in skip}
                con.execute("INSERT INTO labels VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (iid, src, k, b["cls"], b.get("auto_conf"), *b["bbox"], text, None, json.dumps(extra)))
    if a.dry_run:
        print(f"dry run: {len(veh)} vehicle frames, {n_plate} plates, {n_text} text boxes, unknown labels {unknown}")
        return
    for src, kind, n in ((a.source + "_plates", "plate", n_plate), (a.source + "_text", "text", n_text)):
        if n:
            con.execute("INSERT OR REPLACE INTO sources VALUES (?,?,?,?)", (src, kind, f"from {Path(a.jsonl).name}", time.time()))
    res = import_records(con, a.source, veh, note=f"from {Path(a.jsonl).name}", commit=False) if veh else {}
    con.commit()
    print(f"vehicles {res}  plates {n_plate}  text {n_text}  unknown labels {unknown}")


if __name__ == "__main__":
    main()
