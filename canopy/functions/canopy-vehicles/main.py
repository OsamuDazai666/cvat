"""CVAT detector: SAM 3 proposals (full frame + 2x2 tiles, tools/autolabel/propose.py) -> Qwen closed-set class per
box (tools/autolabel/finalize.py:ask, the prompt that produced auto_v1).

If no Qwen is being served, boxes get the class of their best SPECIFIC SAM 3 prompt instead and are marked
needs_review: fast pre-labels, not auto_v1 quality.
"""
import random
from concurrent.futures import ThreadPoolExecutor

from canopy_fn import VLM_NAME, VLM_URL, box, decode_image, models, pct, respond, served

# SAM 3 concept prompt -> our class, for the no-VLM fallback (prompts.SAM_PROMPTS keys)
PROMPT_CLASS = {
    "car": "car", "suv": "suv", "pickup truck": "pickup", "van": "van", "minibus": "minibus", "bus": "bus",
    "truck": "truck", "water tanker truck": "water_tanker", "dump truck": "dump_truck",
    "flatbed truck": "flatbed_truck", "concrete mixer truck": "concrete_mixer_truck", "tow truck": "truck",
    "semi-trailer truck": "truck", "trailer": "other_vehicle", "motorcycle": "motorcycle",
    "bicycle": "other_vehicle", "vehicle": "other_vehicle", "excavator": "excavator", "wheel loader": "wheel_loader",
    "backhoe loader": "backhoe_loader", "bulldozer": "other_vehicle", "motor grader": "grader",
    "road roller": "other_vehicle", "mobile crane": "crane_truck", "forklift": "other_vehicle",
    "heavy machinery": "other_vehicle", "construction vehicle": "other_vehicle", "person": "worker",
    "construction worker": "worker",
}
CATCH_ALL = {"vehicle", "heavy machinery", "construction vehicle"}


def init_context(context):
    from common import load_classes
    from classify import VLM
    context.user_data.cfg = load_classes()
    context.user_data.vlm = VLM(VLM_URL, VLM_NAME)
    context.logger.info("canopy-vehicles ready")


def handler(context, event):
    import finalize                      # ask(): crop + context frame -> {box_quality, class, p_class, ...}
    import prompts as P
    data = event.body
    img = decode_image(data["image"])
    thr = float(data.get("threshold") or 0.35)
    prop = models("/propose", {"image": data["image"], "tiles": True, "min_score": thr})
    boxes = prop["boxes"]
    use_vlm = VLM_NAME in served(VLM_URL)
    shapes = []
    if use_vlm:
        cfg, vlm = context.user_data.cfg, context.user_data.vlm

        def one(i_b):
            i, b = i_b
            return b, finalize.ask(vlm, cfg, img, b["bbox"], random.Random(i))

        with ThreadPoolExecutor(16) as ex:
            for b, v in ex.map(one, enumerate(boxes)):
                if v.get("box_quality") not in P.KEEP_QUALITY:      # fragments / not_an_object are dropped
                    continue
                review = ((v.get("p_class") or 0) < 0.7 or (v.get("p_quality") or 0) < 0.6
                          or v["box_quality"] == "several_objects")           # finalize.route criteria
                shapes.append(box(v["class"], b["bbox"], v.get("p_class") or 0, {
                    "box_quality": v["box_quality"], "auto_conf": pct(v.get("p_class")),
                    "needs_review": review, "auto_source": "sam3+qwen"}))
    else:
        for b in boxes:
            # the catch-all prompts ("vehicle" ...) win on almost every box; the best SPECIFIC prompt names the type
            specific = [(p, s) for p, s in b["prompts"].items() if p not in CATCH_ALL]
            top, s = max(specific, key=lambda t: t[1]) if specific else ("vehicle", b["score"])
            shapes.append(box(PROMPT_CLASS.get(top, "other_vehicle"), b["bbox"], s, {
                "auto_conf": pct(s), "needs_review": True, "auto_source": "sam3"}))
    context.logger.info(f"{len(boxes)} proposals -> {len(shapes)} shapes (vlm={use_vlm})")
    return respond(context, shapes)
