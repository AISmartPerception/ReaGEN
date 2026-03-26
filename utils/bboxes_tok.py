from typing import Tuple, Optional, List, Dict
import json, re
from PIL import Image
from math import isfinite
import logging
import string
import unicodedata

from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessorList, NoBadWordsLogitsProcessor

ASCII_DIGITS = set("0123456789")
# Treat a broad set of Unicode spaces as whitespace, but keep digits ASCII-only
EXTRA_SPACES = set([
    "\u00A0",  # NBSP
    "\u2000", "\u2001", "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
    "\u2007", "\u2008", "\u2009", "\u200A", "\u202F", "\u205F", "\u3000",
])
ASCII_SPACE = set(string.whitespace) | EXTRA_SPACES

def _make_feasible_min_sizes(W: int, H: int, min_w: int, min_h: int) -> tuple[int, int]:
    # Ensure min sizes are not larger than the image capacity (W-1, H-1)
    cap_w = max(1, W - 1)
    cap_h = max(1, H - 1)
    return max(1, min(min_w, cap_w)), max(1, min(min_h, cap_h))

def _is_ascii_digit(ch: str) -> bool:
    return ch in ASCII_DIGITS

def _is_space_like(ch: str) -> bool:
    return (ch in ASCII_SPACE) or (unicodedata.category(ch).startswith("Z"))

def build_prefix_fn_bounds_order_ascii(tokenizer, base_len: int, W: int, H: int,
                                       min_w: int = 8, min_h: int = 8, min_last_digits: int = 2, logger: logging.Logger | None = None):
    """
    ASCII-only version:
      - Only ASCII digits and ASCII whitespace are allowed between <|box_start|> and <|box_end|>.
      - Enforces bounds while typing: x1 ∈ [0, W-1-min_w], y1 ∈ [0, H-1-min_h],
        x2 ∈ [x1+min_w, W-1], y2 ∈ [y1+min_h, H-1].
      - </box_end|> is only allowed after a valid 4th integer (or while typing it with >= min_last_digits).
    """
    BOX_END_ID = tokenizer.convert_tokens_to_ids("<|box_end|>")

    min_w_eff, min_h_eff = _make_feasible_min_sizes(W, H, min_w, min_h)

    # allow only tokens whose decoded text is ASCII digits/ASCII whitespace
    digitspace_ids = []
    for tid in range(tokenizer.vocab_size):
        s = tokenizer.decode([tid], skip_special_tokens=False)
        if not s:
            continue
        ok = True
        for ch in s:
            if _is_ascii_digit(ch) or _is_space_like(ch):
                continue
            ok = False
            break
        if ok:
            digitspace_ids.append(tid)

    def parse_tail(tail: str):
        """Return (vals: list[int], in_num: bool, cur: str of digits for current coord)."""
        vals, in_num, cur = [], False, ""
        for ch in tail:
            if not in_num:
                if _is_ascii_digit(ch):
                    in_num, cur = True, ch
                elif _is_space_like(ch):
                    pass
                else:
                    # Should not happen, but ignore
                    pass
            else:
                if _is_ascii_digit(ch):
                    cur += ch
                elif _is_space_like(ch):
                    if cur:
                        vals.append(int(cur))
                    in_num, cur = False, ""
                else:
                    if cur:
                        vals.append(int(cur))
                    in_num, cur = False, ""
        return vals, in_num, cur

    def coord_bounds(idx: int, vals: list[int]):
        # 0:x1, 1:y1, 2:x2, 3:y2
        if idx == 0:   # x1
            return 0, max(0, (W - 1) - min_w_eff)
        if idx == 1:   # y1
            return 0, max(0, (H - 1) - min_h_eff)
        if idx == 2:   # x2
            lo = (vals[0] + min_w_eff) if len(vals) >= 1 else 0
            return max(0, lo), (W - 1)
        if idx == 3:   # y2
            lo = (vals[1] + min_h_eff) if len(vals) >= 2 else 0
            return max(0, lo), (H - 1)
        return 0, 0

    def can_append_digit(cur: str, d: str, lo: int, hi: int) -> bool:
        if not _is_ascii_digit(d):
            return False
        # Don't allow more digits than hi's length
        if len(cur) + 1 > len(str(max(0, hi))):
            return False
        newv = int(cur + d) if cur else int(d)
        # We enforce upper bound while typing; lower bound is enforced on finalize/close
        return newv <= hi

    def simulate_token(tail: str, token_text: str, vals: list[int], in_num: bool, cur: str) -> bool:
        sv, sin, scur = vals[:], in_num, cur
        idx = len(sv)
        for ch in token_text:
            if _is_space_like(ch):
                if sin:
                    lo, hi = coord_bounds(idx, sv)
                    val = int(scur) if scur else 0
                    if not (lo <= val <= hi):
                        return False
                    sv.append(val)
                    sin, scur, idx = False, "", len(sv)
                continue
            if _is_ascii_digit(ch):
                if idx > 3:
                    return False
                lo, hi = coord_bounds(idx, sv)
                if not can_append_digit(scur, ch, lo, hi):
                    return False
                sin = True
                scur = scur + ch if scur else ch
                continue
            # Any other char is disallowed
            return False

        if sin:
            # Still typing → ensure prefix ≤ hi
            lo, hi = coord_bounds(idx, sv)
            if scur and int(scur) > hi:
                return False
        return True

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        ids = input_ids.tolist()
        tail_ids = ids[base_len:]  # only the generated region
        tail = tokenizer.decode(tail_ids, skip_special_tokens=False)

        vals, in_num, cur = parse_tail(tail)
        idx = len(vals)

        # If constraints are fundamentally infeasible (e.g., W<2/H<2), allow immediate close
        if (W < 2 or H < 2):
            return [BOX_END_ID]

        # After 4 ints → only allow the closer
        if idx >= 4:
            return [BOX_END_ID]

        allowed = []
        for tid in digitspace_ids:
            s = tokenizer.decode([tid], skip_special_tokens=False)
            if s and simulate_token(tail, s, vals, in_num, cur):
                allowed.append(tid)

        # Allow closing if the 4th is valid, or while typing 4th with enough digits and ≤ hi
        allow_close = False
        if len(vals) >= 4:
            allow_close = True
        elif idx == 3 and in_num and len(cur) >= max(1, int(min_last_digits)):
            lo, hi = coord_bounds(idx, vals)
            if int(cur) <= hi:
                allow_close = True
        if allow_close:
            allowed.append(BOX_END_ID)

        # -------- SAFE FALLBACK --------
        if not allowed:
            # Log and relax to avoid HF error
            if logger:
                logger.warning(
                    f"[prefix] empty allowed set — relaxing (idx={idx}, vals={vals}, in_num={in_num}, cur={cur!r}, "
                    f"W={W}, H={H}, min_w_eff={min_w_eff}, min_h_eff={min_h_eff})"
                )
            return list(range(tokenizer.vocab_size))

        return allowed

    return prefix_allowed_tokens_fn

class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: Optional[float] = None

    def __init__(self, x1, y1, x2, y2, conf=None):
        self.x1 = int(round(x1))
        self.y1 = int(round(y1))

        if self.x1 == -1:
            self.x1 = 0
        elif self.y1 < -1:
            self.y1 = 0

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

def fit_box_xyxy(
    x1, y1, x2, y2, W: int, H: int, min_w: int = 2, min_h: int = 2
) -> Tuple[int, int, int, int, Dict[str, object]]:
    """Clamp/repair (x1,y1,x2,y2) to 0<=x1<x2<=W-1, 0<=y1<y2<=H-1 with mins."""
    viol: List[str] = []
    adj:  List[str] = []

    def bad_img(): 
        return 0,0,0,0, {"ok": False, "violations": viol or ["invalid_image_size"], "adjustments": []}
    def clamp(v,a,b): 
        return a if v<a else b if v>b else v

    # Image sanity
    if not isinstance(W,int) or not isinstance(H,int) or W<=0 or H<=0 or W<2 or H<2:
        return bad_img()

    # Cast + non-finite
    try:
        x1,y1,x2,y2 = map(float, (x1,y1,x2,y2))
    except Exception:
        viol.append("non_numeric_input"); return bad_img()
    if not all(isfinite(v) for v in (x1,y1,x2,y2)):
        viol.append("non_finite_or_non_numeric")
        x1 = 0.0 if not isfinite(x1) else x1
        y1 = 0.0 if not isfinite(y1) else y1
        x2 = 1.0 if not isfinite(x2) else x2
        y2 = 1.0 if not isfinite(y2) else y2
        adj.append("replaced_non_finite_with_defaults")

    # Order
    if x2 < x1: x1,x2 = x2,x1; viol.append("swapped_x"); adj.append("swap_x1_x2")
    if y2 < y1: y1,y2 = y2,y1; viol.append("swapped_y"); adj.append("swap_y1_y2")

    # Intended sizes
    w_int = int(round(x2-x1)); h_int = int(round(y2-y1))
    if w_int < min_w: viol.append("width_lt_min_w")
    if h_int < min_h: viol.append("height_lt_min_h")

    # Feasible caps (xyxy => max width/height is W-1/H-1)
    max_w = max(1, W-1); max_h = max(1, H-1)
    if min_w > max_w: viol.append("min_w_exceeds_image"); 
    if min_h > max_h: viol.append("min_h_exceeds_image")

    w = clamp(w_int, min_w, max_w)
    h = clamp(h_int, min_h, max_h)
    if w != w_int: adj.append(f"width_adjusted_to_{w}")
    if h != h_int: adj.append(f"height_adjusted_to_{h}")

    # Place x
    max_x1 = max(0, (W-1) - w)
    xi = clamp(int(round(x1)), 0, max_x1)
    if xi != int(round(x1)):
        if int(round(x1)) < 0: viol.append("x1_lt_0")
        if int(round(x1)) > max_x1: viol.append("x1_gt_max")
    x1i, x2i = xi, xi + w

    # Place y
    max_y1 = max(0, (H-1) - h)
    yi = clamp(int(round(y1)), 0, max_y1)
    if yi != int(round(y1)):
        if int(round(y1)) < 0: viol.append("y1_lt_0")
        if int(round(y1)) > max_y1: viol.append("y1_gt_max")
    y1i, y2i = yi, yi + h

    # Final edge clamps
    if x2i > W-1: viol.append("x2_gt_W-1"); x2i = W-1
    if y2i > H-1: viol.append("y2_gt_H-1"); y2i = H-1

    # Enforce strict > and mins post-clamp
    if x2i <= x1i:
        viol.append("non_positive_width"); x1i = clamp(x1i, 0, max(0, W-2)); x2i = x1i + 1; adj.append("forced_min_width_1px")
    if y2i <= y1i:
        viol.append("non_positive_height"); y1i = clamp(y1i, 0, max(0, H-2)); y2i = y1i + 1; adj.append("forced_min_height_1px")

    if (x2i - x1i) < min_w and (W-1) >= min_w:
        viol.append("post_lt_min_w"); x1i = clamp(x1i, 0, (W-1)-min_w); x2i = x1i + min_w; adj.append("expanded_to_min_w")
    if (y2i - y1i) < min_h and (H-1) >= min_h:
        viol.append("post_lt_min_h"); y1i = clamp(y1i, 0, (H-1)-min_h); y2i = y1i + min_h; adj.append("expanded_to_min_h")

    report = {"ok": not viol, "violations": viol, "adjustments": adj}
    return int(x1i), int(y1i), int(x2i), int(y2i), report


def _clamp_xyxy(x1, y1, x2, y2, W, H, min_w: int = 2, min_h: int = 2, logger: logging.Logger = None) -> Tuple[int, int, int, int]:
    # First clamp crudely to image bounds to avoid huge numbers
    x1 = max(-1e9, min(1e9, float(x1)))
    y1 = max(-1e9, min(1e9, float(y1)))
    x2 = max(-1e9, min(1e9, float(x2)))
    y2 = max(-1e9, min(1e9, float(y2)))
    # logger.info(f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")
    ok = False
    while not ok:
        # ok, violations = validate_box_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h)
        x1, y1, x2, y2, report = fit_box_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h)
        ok = report["ok"]
    
    # if report["ok"]:
    return x1, y1, x2, y2
    # else:
    #     if logger is not None:
    #         logger.info(f"report: {report}")
    #     return 0, 0, W-1, H-1

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


def _pad_box(b, W: int, H: int, pad_ratio: float = 0.08, min_w: int = 2, min_h: int = 2, logger: Optional["logging.Logger"] = None):
    bw = max(1, b.x2 - b.x1)
    bh = max(1, b.y2 - b.y1)
    pw = int(round(bw * pad_ratio))
    ph = int(round(bh * pad_ratio))
    x1, y1, x2, y2 = _clamp_xyxy(b.x1 - pw, b.y1 - ph, b.x2 + pw, b.y2 + ph, W, H, min_w=min_w, min_h=min_h, logger=logger)
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


def _parse_bboxes_tokens(
    text: str,
    W: int,
    H: int,
    min_w: int = 2,
    min_h: int = 2,
    logger: Optional["logging.Logger"] = None,
) -> BBox:

    END = "<|box_end|>"

    if logger:
        logger.info(f"text: {text!r}")

    s = 0
    e = text.find(END, s)
    if e == -1:
        if logger: logger.info("No <|box_end|> marker found after <|box_start|>.")
        return BBox(0, 0, W - 1, H - 1, None)

    payload = text[s:e].strip()
    # if logger:
    #     logger.info(f"box payload: {payload!r}")

    def _extract_first_four_ints(s: str) -> list[int]:
        vals, i, n = [], 0, len(s)
        while i < n and len(vals) < 4:
            ch = s[i]
            # skip until sign or digit
            if not (ch.isdigit() or ch in "+-"):
                i += 1
                continue
            # optional sign
            sign = 1
            if ch in "+-":
                sign = -1 if ch == "-" else 1
                i += 1
                if i >= n or not s[i].isdigit():
                    # lone sign; skip
                    continue
            # read digits
            v, had = 0, False
            while i < n and s[i].isdigit():
                v = v * 10 + (ord(s[i]) - 48)
                i += 1
                had = True
            if had:
                vals.append(sign * v)
        return vals

    # explicit 'none' support: only treat as empty if there are NO integers
    ints = _extract_first_four_ints(payload)
    if ("none" in payload.lower()) and not ints:
        if logger: logger.info("Model returned 'none' (no integers present).")
        return BBox(0, 0, W - 1, H - 1, None)

    if len(ints) < 4:
        if logger: logger.info(f"Fewer than 4 integers found in payload: {ints}")
        return BBox(0, 0, W - 1, H - 1, None)

    x1, y1, x2, y2 = ints[:4]

    x1, y1, x2, y2 = _clamp_xyxy(x1, y1, x2, y2, W, H, min_w=min_w, min_h=min_h, logger=logger)
    # if logger:
    #     logger.info(f"bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

    return BBox(int(x1), int(y1), int(x2), int(y2), None)

def _iou(a, b, eps: float = 1e-6) -> float:
    # assumes half-open boxes: [x1, x2) × [y1, y2)
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2

    # invalid boxes -> IoU 0
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0

    xi1, yi1 = max(ax1, bx1), max(ay1, by1)
    xi2, yi2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, xi2 - xi1), max(0, yi2 - yi1)
    inter = float(iw * ih)

    area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
    area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
    union = max(area_a + area_b - inter, 0.0) + eps  # keep non-negative, avoid /0

    return inter / union
