from typing import Tuple, Optional, List
import json, re
from PIL import Image
from math import isfinite
import logging


class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: Optional[float] = None

    def __init__(self, x1, y1, x2, y2, conf=None):
        self.x1 = int(round(x1))
        self.y1 = int(round(y1))
        self.x2 = int(round(x2))
        self.y2 = int(round(y2))
        self.conf = conf


def validate_box_xyxy(x1, y1, x2, y2, W, H, min_w: int = 2, min_h: int = 2):
    """
    Pure detection: returns (ok: bool, violations: list[str])
    Does NOT modify the inputs.
    """
    violations = []

    # Image sanity
    if not isinstance(W, int) or not isinstance(H, int):
        violations.append("image_size_not_int")
    if W <= 0 or H <= 0:
        violations.append("image_size_non_positive")
        return False, violations
    if W < 2 or H < 2:
        violations.append("image_too_small_for_xyxy")  # need at least 2 pixels along each axis

    # Inputs finite?
    nums = (x1, y1, x2, y2)
    if not all(isinstance(v, (int, float)) and isfinite(v) for v in nums):
        violations.append("non_finite_or_non_numeric")

    # Order
    if x2 <= x1:
        violations.append("non_positive_width")
    if y2 <= y1:
        violations.append("non_positive_height")

    # Bounds (allow real numbers here; integer-ness is an output requirement)
    if x1 < 0: violations.append("x1_lt_0")
    if y1 < 0: violations.append("y1_lt_0")
    if x2 > W - 1: violations.append("x2_gt_W-1")
    if y2 > H - 1: violations.append("y2_gt_H-1")

    # Size
    w = x2 - x1
    h = y2 - y1
    if w < min_w: violations.append("width_lt_min_w")
    if h < min_h: violations.append("height_lt_min_h")

    # Feasibility of min size in the image
    if min_w > max(1, W - 1):
        violations.append("min_w_exceeds_image_capacity")
    if min_h > max(1, H - 1):
        violations.append("min_h_exceeds_image_capacity")

    ok = len(violations) == 0
    return ok, violations


def fit_box_xyxy(x1, y1, x2, y2, W, H, min_w: int = 2, min_h: int = 2):
    """
    Repair to meet: 0 <= x1 < x2 <= W-1, 0 <= y1 < y2 <= H-1,
    integer pixels, and (x2-x1) >= min_w, (y2-y1) >= min_h.

    Returns: (x1i, y1i, x2i, y2i, report_dict)
    report_dict = {"violations": [...], "adjustments": [...], "ok": bool}
    """
    adjustments, violations = [], []

    # Early feasibility checks for the image and min sizes
    if not isinstance(W, int) or not isinstance(H, int) or W <= 0 or H <= 0:
        return 0, 0, 0, 0, {"ok": False, "violations": ["invalid_image_size"], "adjustments": []}
    if W < 2 or H < 2:
        return 0, 0, 0, 0, {"ok": False, "violations": ["image_too_small_for_xyxy"], "adjustments": []}

    # Convert to float and check finiteness
    try:
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
    except Exception:
        return 0, 0, 0, 0, {"ok": False, "violations": ["non_numeric_input"], "adjustments": []}
    if not all(isfinite(v) for v in (x1, y1, x2, y2)):
        violations.append("non_finite_or_non_numeric")
        # Replace NaNs/infs with zeros to proceed safely
        x1 = 0.0 if not isfinite(x1) else x1
        y1 = 0.0 if not isfinite(y1) else y1
        x2 = 1.0 if not isfinite(x2) else x2
        y2 = 1.0 if not isfinite(y2) else y2
        adjustments.append("replaced_non_finite_with_defaults")

    # Ensure order (strict)
    if x2 < x1:
        x1, x2 = x2, x1
        violations.append("swapped_x_due_to_negative_width")
        adjustments.append("swap_x1_x2")
    if y2 < y1:
        y1, y2 = y2, y1
        violations.append("swapped_y_due_to_negative_height")
        adjustments.append("swap_y1_y2")

    # Round intended size to integer pixels, enforce mins and image capacity
    w_intended = int(round(x2 - x1))
    h_intended = int(round(y2 - y1))
    if w_intended < min_w:
        violations.append("width_lt_min_w")
    if h_intended < min_h:
        violations.append("height_lt_min_h")

    # Max feasible widths/heights inside image under xyxy semantics
    max_w = max(1, W - 1)
    max_h = max(1, H - 1)

    if min_w > max_w:
        violations.append("min_w_exceeds_image_capacity")
    if min_h > max_h:
        violations.append("min_h_exceeds_image_capacity")

    w = min(max(min_w, w_intended), max_w)
    h = min(max(min_h, h_intended), max_h)
    if w != w_intended:
        adjustments.append(f"width_adjusted_to_{w}")
    if h != h_intended:
        adjustments.append(f"height_adjusted_to_{h}")

    # Place horizontally: valid x1 in [0, (W-1)-w]
    max_x1 = (W - 1) - w
    if max_x1 < 0:
        # Shouldn’t happen given clamps above, but guard anyway
        max_x1 = 0
        w = max(1, W - 1)
        adjustments.append("width_forced_to_image_capacity")

    x1 = int(round(x1))
    if x1 < 0:
        violations.append("x1_lt_0")
    if x1 > max_x1:
        violations.append("x1_gt_max_allowed")
    x1 = min(max(0, x1), max_x1)
    x2 = x1 + w

    # Place vertically: valid y1 in [0, (H-1)-h]
    max_y1 = (H - 1) - h
    if max_y1 < 0:
        max_y1 = 0
        h = max(1, H - 1)
        adjustments.append("height_forced_to_image_capacity")

    y1 = int(round(y1))
    if y1 < 0:
        violations.append("y1_lt_0")
    if y1 > max_y1:
        violations.append("y1_gt_max_allowed")
    y1 = min(max(0, y1), max_y1)
    y2 = y1 + h

    # Final clamps and strictness fixes
    if x2 > W - 1:
        violations.append("x2_gt_W-1")
    if y2 > H - 1:
        violations.append("y2_gt_H-1")
    x2 = min(x2, W - 1)
    y2 = min(y2, H - 1)

    # Ensure strict inequality (fixes the x1==W-1 bug)
    if x2 <= x1:
        violations.append("non_positive_width_after_clamp")
        x1 = max(0, min(x1, W - 2))
        x2 = x1 + 1
        adjustments.append("forced_min_width_1px")
    if y2 <= y1:
        violations.append("non_positive_height_after_clamp")
        y1 = max(0, min(y1, H - 2))
        y2 = y1 + 1
        adjustments.append("forced_min_height_1px")

    # Enforce minimum sizes again post-clamp if necessary
    if (x2 - x1) < min_w and (W - 1) >= min_w:
        violations.append("post_clamp_width_lt_min_w")
        x1 = max(0, min(x1, (W - 1) - min_w))
        x2 = x1 + min_w
        adjustments.append("expanded_to_min_w")
    if (y2 - y1) < min_h and (H - 1) >= min_h:
        violations.append("post_clamp_height_lt_min_h")
        y1 = max(0, min(y1, (H - 1) - min_h))
        y2 = y1 + min_h
        adjustments.append("expanded_to_min_h")

    ok = len(violations) == 0
    report = {"ok": ok, "violations": violations, "adjustments": adjustments}
    return int(x1), int(y1), int(x2), int(y2), report


def _clamp_xyxy(x1, y1, x2, y2, W, H, min_w: int = 2, min_h: int = 2, logger: logging.Logger = None) -> Tuple[int, int, int, int]:
    # First clamp crudely to image bounds to avoid huge numbers
    x1 = max(-1e9, min(1e9, float(x1)))
    y1 = max(-1e9, min(1e9, float(y1)))
    x2 = max(-1e9, min(1e9, float(x2)))
    y2 = max(-1e9, min(1e9, float(y2)))
    # Then fit a window of at least min size entirely inside the image
    ok, violations = validate_box_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h)
    if logger is not None and len(violations) > 0:
        logger.info(f"violations: {violations}")
    x1, y1, x2, y2, report = fit_box_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h)
    if report["ok"]:
        return x1, y1, x2, y2
    else:
        if logger is not None:
            logger.info(f"Error parsing JSON: {report}")
        return 0, 0, W-1, H-1

def _iou(a, b) -> float:
    xi1, yi1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xi2, yi2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = iw * ih
    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _nms(boxes, iou_thresh: float = 0.5, limit: Optional[int] = None):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda b: (b.conf or 0.0), reverse=True)
    kept = []
    for b in boxes:
        if all(_iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
            if limit and len(kept) >= limit:
                break
    return kept


def _pad_box(b, W: int, H: int, pad_ratio: float = 0.08, min_w: int = 2, min_h: int = 2):
    bw = max(1, b.x2 - b.x1)
    bh = max(1, b.y2 - b.y1)
    pw = int(round(bw * pad_ratio))
    ph = int(round(bh * pad_ratio))
    x1, y1, x2, y2 = _clamp_xyxy(b.x1 - pw, b.y1 - ph, b.x2 + pw, b.y2 + ph, W, H, min_w=min_w, min_h=min_h)
    return BBox(x1, y1, x2, y2, b.conf)


def _maybe_to_pil(img_like) -> Image.Image:
    if isinstance(img_like, Image.Image):
        return img_like
    try:
        import numpy as np
        if hasattr(img_like, "cpu"):  # torch tensor
            arr = img_like.detach().cpu().numpy()
        else:
            arr = np.array(img_like)
        if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
            if arr.shape[2] == 1:
                arr = arr.squeeze(-1).repeat(3, axis=2)
            return Image.fromarray(arr.astype("uint8"))
    except Exception:
        pass
    raise TypeError("Unsupported image type—provide PIL.Image or HWC uint8 array/tensor.")


def _parse_bboxes_json(text: str, W: int, H: int, min_w: int = 2, min_h: int = 2, logger: logging.Logger = None) -> List[BBox]:

    def _scale_to_px(b, units: str):
        x1, y1, x2, y2 = map(float, b[:4])
        u = (units or "").lower()
        if u in ("norm", "ratio"):
            # Use (W-1,H-1) so max ratio 1.0 maps to last pixel index
            X1 = x1 * (W - 1)
            Y1 = y1 * (H - 1)
            X2 = x2 * (W - 1)
            Y2 = y2 * (H - 1)
            return _clamp_xyxy(X1, Y1, X2, Y2, W, H, min_w=min_w, min_h=min_h, logger=logger)
        # default px
        return _clamp_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h, logger=logger)

    # Try to parse JSON object span
    obj = None
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            # logger.info(f"obj: {obj}")
        except Exception:
            obj = None
            logger.info(f"Error parsing JSON: {m}")


    if isinstance(obj, dict) and isinstance(obj.get("bbox"), list):

        raw = obj.get("bbox")
        if not raw or len(raw) < 4: 
            return BBox(0, 0, W-1, H-1, None)
        units = obj.get("units") or ""
        x1, y1, x2, y2 = _scale_to_px(raw, units)
        conf = obj.get("confidence")

        return BBox(x1, y1, x2, y2, float(conf) if conf is not None else None)


    # Fallback: harvest numbers in groups of 4
    # nums = re.findall(r"-?\d+\.?\d*", text)
    # vals = list(map(float, nums))
    # for i in range(0, min(len(vals), 4 * max_k), 4):
    #     raw = vals[i:i + 4]
    #     if len(raw) < 4:
    #         break
    #     if all(0.0 <= v <= 1.0 for v in raw):
    #         x1, y1, x2, y2 = _clamp_xyxy(raw[0] * (W - 1), raw[1] * (H - 1),
    #                                       raw[2] * (W - 1), raw[3] * (H - 1),
    #                                       W, H, min_w=min_w, min_h=min_h)
    #     else:
    #         x1, y1, x2, y2 = _clamp_xyxy(*raw, W=W, H=H, min_w=min_w, min_h=min_h)
    #     bboxes.append(BBox(x1, y1, x2, y2, None))
    # return bboxes[:max_k]
