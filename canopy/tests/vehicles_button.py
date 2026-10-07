"""Press "Canopy vehicles" through CVAT's lambda API (same path as the UI button) on every frame of a vehicles task;
time each call and write the shapes as a pull-format JSONL for tests/compare_auto_v1.py.
    run in the tools container with TASK=<id>"""
import json, os, time
from pathlib import Path
from cvat_sdk import make_client
from cvat_sdk.api_client import models

c = make_client(os.environ["CVAT_URL"], credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]))
task = int(os.environ.get("TASK", 4))
t = c.tasks.retrieve(task)
names = {l.id: l.name for l in t.get_labels()}
frames = t.get_meta().frames
out = Path(f"/work/exports/button_task{task}.jsonl")
times = []
with open(out, "w") as f:
    for fr in range(len(frames)):
        t0 = time.time()
        _, r = c.api_client.lambda_api.create_functions("canopy-vehicles", online_function_call_request=models.OnlineFunctionCallRequest(
            task=task, frame=fr, cleanup=False), _parse_response=False)
        dt = time.time() - t0
        times.append(dt)
        shapes = json.loads(r.data)
        shapes = shapes.get("shapes", shapes) if isinstance(shapes, dict) else shapes
        src = {next((a["value"] for a in s["attributes"] if a["value"] in ("sam3", "sam3+qwen")), "?") for s in shapes}
        print(f"frame {fr}: {len(shapes)} boxes in {dt:.1f}s, source {src}", flush=True)
        item = ("v:" if frames[fr].name.startswith("vehicles/") else "p:") + Path(frames[fr].name).stem
        f.write(json.dumps({"item": item, "task": task, "project": "vehicles",
                            "boxes": [{"cls": names[s["label_id"]], "bbox": s["points"][:4]} for s in shapes]}) + "\n")
print(f"mean {sum(times) / len(times):.1f}s per frame (first {times[0]:.1f}s) -> {out}")
