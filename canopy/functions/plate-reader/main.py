"""CVAT detector: plate boxes (plate_det_v1 RF-DETR) + the full plate read by Qwen3.8 (tools/plates/read.py:read_vlm,
prompt v2): number, code, origin, category, plate colour, layout, legibility. Status rules: common/plate_qwen.py.
Without Qwen being served, only the boxes come back (ocr_status unchecked).
"""
from concurrent.futures import ThreadPoolExecutor

from canopy_fn import VLM_NAME, VLM_URL, box, decode_image, models, respond, served
from plate_qwen import attrs


def init_context(context):
    from classify import VLM
    context.user_data.vlm = VLM(VLM_URL, VLM_NAME)
    context.logger.info("plate-reader ready")


def handler(context, event):
    from read import read_vlm
    data = event.body
    img = decode_image(data["image"])
    thr = float(data.get("threshold") or 0.4)
    dets = models("/plate/detect", {"image": data["image"], "conf": thr})["boxes"]
    if not dets:
        return respond(context, [])
    if VLM_NAME not in served(VLM_URL):
        context.logger.warn("Qwen is not being served: returning plate boxes without readings")
        return respond(context, [box("plate", d["bbox"], d["score"], {"ocr_status": "unchecked", "auto_source": "detector"})
                                 for d in dets])
    with ThreadPoolExecutor(8) as ex:
        reads = list(ex.map(lambda d: read_vlm(context.user_data.vlm, img, d["bbox"]), dets))
    shapes = [box("plate", d["bbox"], d["score"], a) for d, v in zip(dets, reads) if (a := attrs(v))]
    context.logger.info(f"{len(dets)} plates -> {len(shapes)} read by qwen")
    return respond(context, shapes)
