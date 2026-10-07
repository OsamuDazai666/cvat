"""CVAT interactor: SAM 3 mask from positive/negative clicks and/or a box, or from text prompts (API only — the
stock CVAT UI has no text box). The model runs in canopy-models; this function only adapts the contract.
"""
import numpy as np

from canopy_fn import models, respond


def init_context(context):
    context.logger.info("sam3-interactor ready")


def _bbox(obj_bbox):
    if not obj_bbox:
        return None
    a = np.asarray(obj_bbox, dtype=float).reshape(-1)
    return [float(min(a[0], a[2])), float(min(a[1], a[3])), float(max(a[0], a[2])), float(max(a[1], a[3]))]


def _mask(rle_pts, score):
    return {"type": "mask", "points": rle_pts, "group": 0, "source": "semi-auto", "occluded": False,
            "rotation": 0, "attributes": [{"spec_id": 0, "value": f"{score:.3f}"}]}


def handler(context, event):
    data = event.body
    text = [t for t in (data.get("text_prompts") or []) if t and t.strip()]
    if text:
        objs = models("/sam3/text", {"image": data["image"], "prompts": text, "conf": 0.35, "masks": True})["objects"]
        return respond(context, {"shapes": [_mask(o["rle"], o["score"]) for o in objs if o.get("rle")]})
    pos, neg = data.get("pos_points") or [], data.get("neg_points") or []
    bb = _bbox(data.get("obj_bbox"))
    if not pos and not bb:
        return respond(context, {"shapes": []})
    res = models("/sam3/visual", {"image": data["image"], "points": pos + neg,
                                  "labels": [1] * len(pos) + [0] * len(neg), "bbox": bb})
    return respond(context, {"shapes": [_mask(m["rle"], m["score"]) for m in res["masks"][:1]]})
