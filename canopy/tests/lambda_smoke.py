"""Call every canopy function through CVAT's lambda API (the path the UI uses) and print what comes back.
    bash tests/lab.sh is for labeler; run this with:  ... run --rm tools python /work/tests/lambda_smoke.py"""
import json, os, time
from cvat_sdk import make_client
from cvat_sdk.api_client import models

c = make_client(os.environ["CVAT_URL"], credentials=(os.environ["CVAT_USER"], os.environ["CVAT_PASSWORD"]))
fns = {f["id"]: f for f in json.loads(c.api_client.lambda_api.list_functions(_parse_response=False)[1].data)}
print("functions in CVAT:", sorted(fns), {k: fns[k]["kind"] for k in fns})


def call(fid, **body):
    t = time.time()
    _, r = c.api_client.lambda_api.create_functions(fid, online_function_call_request=models.OnlineFunctionCallRequest(**body),
                                                   _parse_response=False)
    return json.loads(r.data), time.time() - t


veh_task = int(os.environ.get("VEH_TASK", 3)); plate_task = int(os.environ.get("PLATE_TASK", 2))
meta = c.tasks.retrieve(veh_task).get_meta()
w, h = meta.frames[0].width, meta.frames[0].height
r, dt = call("canopy-sam3-interactor", task=veh_task, frame=0, pos_points=[[w / 2, h * 0.6]], neg_points=[])
r = r.get("shapes", r) if isinstance(r, dict) else r
print(f"interactor: {len(r)} shapes, type {r[0]['type'] if r else None}, {len(r[0]['points']) if r else 0} rle values ({dt:.1f}s)")
for fid, task in (("canopy-vehicles", veh_task), ("canopy-plate-reader", plate_task), ("canopy-text-reader", veh_task)):
    r, dt = call(fid, task=task, frame=0, cleanup=False)
    r = r.get("shapes", r) if isinstance(r, dict) else r
    print(f"{fid}: {len(r)} shapes ({dt:.1f}s)")
    for s in r[:4]:
        print("   ", s.get("label_id"), s["type"], [round(p) for p in s["points"][:4]],
              {a["spec_id"]: a["value"] for a in s.get("attributes", [])})
