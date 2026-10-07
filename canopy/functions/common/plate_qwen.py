"""Plate reading with Qwen3.8 alone (tools/plates/read.py:read_vlm, prompt v2), shared by plate-reader and the labeler.

ocr_status from Qwen's own token confidence, calibrated on the gold set (128 readable plates, threshold 0.9):
  confident       legible=full, p_number >= 0.9 and (no code or p_code >= 0.9)   97/98 correct
  low_confidence  any other read with >= 3 digits                                15/30 correct -> check these first
  unreadable      Qwen says none, or fewer than 3 digits
"""
SURE = 0.9
FIELDS = ("number", "code", "origin", "category", "plate_colour", "layout", "legible")


def status(v):
    if v.get("legible") in ("none", "no_plate") or len(v.get("number") or "") < 3:
        return "unreadable"
    sure = (v.get("legible") == "full" and (v.get("p_number") or 0) >= SURE
            and (not v.get("code") or (v.get("p_code") or 0) >= SURE))
    return "confident" if sure else "low_confidence"


def attrs(v):
    """read_vlm answer -> plate attribute values (None when Qwen says the box is not a plate)."""
    if v.get("legible") == "no_plate":
        return None
    p = min(v.get("p_number") or 0, (v.get("p_code") or 0) if v.get("code") else 1)
    out = {k: v.get(k) or ("" if k in ("number", "code") else "unknown") for k in FIELDS}
    out.update(ocr_status=status(v), auto_conf=int(round(100 * p)), auto_source="qwen")
    return out
