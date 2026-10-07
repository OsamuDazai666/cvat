"""Compare a `labeler pull` export of a vehicles task with auto_v1 in index.sqlite (run on Windows).
    python tests/compare_auto_v1.py exports/cvat_v1_<ts>.jsonl
Greedy one-to-one matching per frame at IoU >= 0.5, then class agreement on the matched pairs."""
import json
import sqlite3
import sys
from collections import Counter


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u else 0


con = sqlite3.connect("file:D:/canopy-evidence/curated/index.sqlite?mode=ro", uri=True)
n_ref = n_new = n_match = n_same = 0
confusion = Counter()
for line in open(sys.argv[1], encoding="utf-8"):
    r = json.loads(line)
    if r["project"] != "vehicles":
        continue
    ref = [(row[0], list(row[1:])) for row in con.execute(
        "SELECT cls, x1, y1, x2, y2 FROM labels WHERE item_id=? AND source='auto_v1' AND x1 IS NOT NULL", (r["item"],))]
    new = [(b["cls"], b["bbox"]) for b in r["boxes"]]
    n_ref += len(ref)
    n_new += len(new)
    pairs = sorted(((iou(a[1], b[1]), i, j) for i, a in enumerate(ref) for j, b in enumerate(new)), reverse=True)
    used_i, used_j = set(), set()
    for v, i, j in pairs:
        if v < 0.5 or i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        n_match += 1
        n_same += ref[i][0] == new[j][0]
        if ref[i][0] != new[j][0]:
            confusion[(ref[i][0], new[j][0])] += 1
print(f"auto_v1 boxes {n_ref}, CVAT boxes {n_new}, matched {n_match} "
      f"({n_match / max(1, n_ref):.0%} of auto_v1), same class {n_same}/{n_match} ({n_same / max(1, n_match):.0%})")
for (a, b), k in confusion.most_common(10):
    print(f"   auto_v1 {a} -> cvat {b}: {k}")
