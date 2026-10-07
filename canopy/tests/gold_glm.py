"""GLM-OCR in the canopy-cvat vLLM container vs the gold plates and vs the WSL bake-off reads (parity).
    python tests/gold_glm.py            (needs canopy_vllm_ocr up on 127.0.0.1:9012)"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
sys.path[:0] = ["D:/canopy-evidence/tools/plates", "D:/canopy-evidence/tools"]
from bakeoff import items, load_gold, GOLD          # noqa: E402
from read import OCR, read_ocr                       # noqa: E402

ocr = OCR("http://127.0.0.1:9012/v1", "glm", "Text Recognition:")
gold, _ = load_gold()
ref = {r["gid"]: r for r in map(json.loads, (GOLD / "bakeoff_glm.jsonl").read_text(encoding="utf-8").splitlines())}
todo = [(g, c) for g, c in items() if c["read_bbox"]]
with ThreadPoolExecutor(16) as ex:
    res = dict(ex.map(lambda gc: (gc[0], read_ocr(ocr, cv2.imread(gc[1]["img"]), gc[1]["read_bbox"])), todo))
same = sum(res[g]["number"] == ref[g]["number"] for g in res if g in ref)
ok = n = 0
for g, r in res.items():
    v = gold.get(g)
    if v and v["legible"] in ("full", "partial") and v["number"] and "number" not in v["unsure"]:
        n += 1
        ok += r["number"] == v["number"]
print(f"container GLM: digits correct {ok}/{n} = {ok / n:.1%} on gold; same number as WSL bake-off {same}/{len(res)}")
Path(__file__).with_name("gold_glm_reads.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
