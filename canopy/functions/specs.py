"""Single source of truth for CVAT labels + attributes: used by gen.py (Nuclio `spec` annotations, which CVAT maps
onto task attributes) and by the labeler (project label schemas). Classes come from tools/classes.json; plate value
lists from tools/plates/read.py, so CVAT never drifts from the pipeline.
"""
import json
import os
import re
from pathlib import Path

TOOLS = Path(os.environ.get("CANOPY_TOOLS") or
             (Path("/canopy/tools") if Path("/canopy/tools").exists() else Path("D:/canopy-evidence/tools")))


def _read_py_list(path, name):
    """A module-level list literal from a tools file, without importing it (read.py imports cv2)."""
    src = (TOOLS / path).read_text(encoding="utf-8")
    m = re.search(rf"^{name} = (\[.*?\])", src, re.S | re.M)
    return json.loads(m.group(1).replace("'", '"'))


def attr(name, input_type, values, default=None, mutable=True):
    if input_type == "text":
        values = [default or ""]
    return {"name": name, "input_type": input_type, "mutable": mutable, "values": values,
            "default_value": default if default is not None else values[0]}


BOX_QUALITY = ["complete_object", "occluded_or_cut_object", "several_objects", "fragment"]   # kept ones first
CONF = attr("auto_conf", "number", ["0", "100", "1"], "0")      # percent: CVAT number attributes are integers
REVIEW = attr("needs_review", "checkbox", ["false"], "false")


def vehicle_classes(include_reviewer_only=False):
    cfg = json.loads((TOOLS / "classes.json").read_text(encoding="utf-8"))
    return [c for c in cfg["classes"] if include_reviewer_only or not c.get("reviewer_only")]


def vehicles(include_reviewer_only=False):
    attrs = [attr("box_quality", "select", BOX_QUALITY), CONF, REVIEW,
             attr("auto_source", "select", ["human", "sam3", "sam3+qwen", "auto_v1"], "human")]
    return [{"name": c["name"], "color": c.get("color"), "type": "rectangle", "attributes": attrs}
            for c in vehicle_classes(include_reviewer_only)]


def plates():
    origins = _read_py_list("plates/read.py", "ORIGINS")
    cats = _read_py_list("plates/read.py", "CATEGORIES")
    colours = _read_py_list("plates/read.py", "COLOURS")
    layouts = _read_py_list("plates/read.py", "LAYOUTS")
    return [{"name": "plate", "color": "#ffd400", "type": "rectangle", "attributes": [
        attr("number", "text", [""]), attr("code", "text", [""]),
        attr("origin", "select", ["unknown"] + [o for o in origins if o != "unknown"]),
        attr("category", "select", ["unknown"] + [c for c in cats if c != "unknown"]),
        attr("legible", "select", ["unknown", "full", "partial", "none"]),
        attr("plate_colour", "select", ["unknown"] + [c for c in colours if c != "unknown"]),
        attr("layout", "select", layouts[::-1] if layouts[-1] == "unknown" else layouts),
        # confident / low_confidence: Qwen-only reader (plate-reader). disputed / single / agreed: two-reader
        # statuses kept for plates_v1 seeds and earlier annotations.
        attr("ocr_status", "select", ["unchecked", "low_confidence", "confident", "unreadable", "verified",
                                      "disputed", "single", "agreed"]),
        CONF, attr("auto_source", "select", ["human", "qwen", "detector", "plates_v1", "onnx", "glm+onnx", "glm+qwen"],
                   "human"),
    ]}]


def text():
    attrs = [attr("transcription", "text", [""]),
             attr("script", "select", ["unknown", "latin", "arabic", "mixed", "other"]),
             attr("legible", "select", ["unknown", "full", "partial", "none"]),
             attr("ocr_status", "select", ["unchecked", "unread", "auto", "verified"]), CONF]
    return [{"name": n, "color": c, "type": "rectangle", "attributes": attrs}
            for n, c in (("text", "#00c2ff"), ("sign", "#ff7a00"), ("vehicle_marking", "#b14cff"))]


def project(first):
    """Every canopy project carries the SAME full label set (vehicle classes, plate, text labels), so CVAT pre-fills
    the label/attribute mapping of every canopy model in every task (it pairs equal names) and nobody has to map
    by hand. The project's own labels come first in the label list. first: vehicles | plates | text"""
    groups = {"vehicles": lambda: vehicles(include_reviewer_only=True), "plates": plates, "text": text}
    return [l for key in [first] + [k for k in groups if k != first] for l in groups[key]()]


def nuclio_spec(labels):
    """CVAT reads label + attribute names/values from the function spec to build the mapping dialog."""
    return [{"id": i, "name": l["name"], "type": l["type"],
             "attributes": [{"name": a["name"], "input_type": a["input_type"], "values": a["values"]}
                            for a in l["attributes"]]} for i, l in enumerate(labels)]
