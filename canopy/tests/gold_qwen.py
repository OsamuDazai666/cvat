"""Qwen3.8 in the canopy-cvat vLLM container on the gold plates: code/origin/category vs gold, parity with the WSL
bake-off reads (bakeoff_qwen38.jsonl), and full-plate accuracy after combine() with the WSL GLM reads.
    python tests/gold_qwen.py      (needs canopy_vllm up on 127.0.0.1:9011)"""
import json, sys
from concurrent.futures import ThreadPoolExecutor
import cv2
sys.path[:0] = ["D:/canopy-evidence/tools/plates", "D:/canopy-evidence/tools"]
from bakeoff import items, load_gold, GOLD     # noqa: E402
from read import VLM, read_vlm                  # noqa: E402
from combine import combine                     # noqa: E402

vlm = VLM("http://127.0.0.1:9011/v1", "qwen38")
gold, _ = load_gold()
load = lambda n: {r["gid"]: r for r in map(json.loads, (GOLD / n).read_text(encoding="utf-8").splitlines())}
ref_v, ref_o = load("bakeoff_qwen38.jsonl"), load("bakeoff_glm.jsonl")
todo = [(g, c) for g, c in items() if c["read_bbox"]]
with ThreadPoolExecutor(16) as ex:
    res = dict(ex.map(lambda gc: (gc[0], read_vlm(vlm, cv2.imread(gc[1]["img"]), gc[1]["read_bbox"])), todo))
for f in ("code", "origin", "category"):
    ok = n = 0
    for g, r in res.items():
        v = gold.get(g)
        if v and v["legible"] in ("full", "partial") and f not in v["unsure"] and (v[f] or f != "code"):
            n += 1
            ok += (r[f] or "") == (v[f] or "")
    same = sum(res[g][f] == ref_v[g][f] for g in res if g in ref_v)
    print(f"{f:9s} correct {ok}/{n} = {ok / n:.1%}   same as WSL read {same}/{len(res)}")
ok = n = 0
for g, r in res.items():
    v = gold.get(g)
    if v and v["legible"] in ("full", "partial") and v["number"] and not ({"code", "number"} & v["unsure"]):
        c = combine(r, ref_o.get(g))
        n += 1
        ok += c["code"] + c["number"] == v["code"] + v["number"]
print(f"full plate (code+number, combine with GLM) {ok}/{n} = {ok / n:.1%}")
