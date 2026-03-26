from typing import Iterable, Tuple, Optional, Union, List
from PIL import Image, ImageDraw, ImageColor, ImageFont

Box = Union[Tuple[float, float, float, float], Tuple[float, float, float, float, float]]
# ^ (x1,y1,x2,y2) or (x1,y1,x2,y2,conf)

def _clamp_order_xyxy(x1, y1, x2, y2, W, H):
    # order
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    # clamp
    x1 = max(0, min(W - 1, int(round(x1))))
    y1 = max(0, min(H - 1, int(round(y1))))
    x2 = max(0, min(W - 1, int(round(x2))))
    y2 = max(0, min(H - 1, int(round(y2))))
    # avoid zero-area by nudging outward when needed
    if x2 <= x1: x2 = min(W - 1, x1 + 1)
    if y2 <= y1: y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2

def _rgba(color, alpha=255):
    rgb = ImageColor.getrgb(color)
    return (rgb[0], rgb[1], rgb[2], alpha)

def _auto_width(W, H):
    # scale with image size; tweak factor to taste
    return max(2, int(round(min(W, H) * 0.0035)))

def _draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int,int], text: str, font, fill_fg=(255,255,255,255), fill_bg=(0,0,0,160), pad=3):
    x, y = xy
    # textbbox((x,y), text, font=font) returns (l,t,r,b)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    w, h = r - l, b - t
    draw.rectangle([x, y, x + w + 2*pad, y + h + 2*pad], fill=fill_bg)
    draw.text((x + pad, y + pad), text, fill=fill_fg, font=font)

def _normalize_pred_boxes(pred_bboxes: Optional[Iterable], W, H) -> List[Tuple[int,int,int,int,Optional[float]]]:
    """Accepts:
      - list of (x1,y1,x2,y2)
      - list of (x1,y1,x2,y2,conf)
      - list of dicts like {"bbox":[...], "confidence":0.87}
    Returns a list of (x1,y1,x2,y2,conf_or_None) in pixels.
    """
    out = []
    if not pred_bboxes:
        return out
    for item in pred_bboxes:
        conf = None
        if isinstance(item, dict):
            raw = item.get("bbox", [])
            if len(raw) < 4: continue
            x1, y1, x2, y2 = raw[:4]
            conf = item.get("confidence", None)
        else:
            # tuple/list
            if len(item) < 4: continue
            x1, y1, x2, y2 = item[:4]
            if len(item) >= 5:
                conf = item[4]
        x1, y1, x2, y2 = _clamp_order_xyxy(x1, y1, x2, y2, W, H)
        out.append((x1, y1, x2, y2, conf))
    return out

def draw_bboxes(
    sample,
    gt_key: str = "bboxs",                    # key for GT in sample
    gt_color: str = "lime",
    pred_bboxes: Optional[Iterable] = None,   # see _normalize_pred_boxes
    pred_color: str = "#ff8c00",
    width: Optional[int] = None,
    fill_alpha: float = 0.20,
    gt_labels: Optional[List[str]] = None,    # optional per-box text
    pred_labels: Optional[List[str]] = None,  # optional per-box text
    show_conf: bool = True,                   # append conf to pred labels if available
    font: Optional[ImageFont.ImageFont] = None
):
    """
    Draws ground-truth boxes from sample[gt_key] and optional predicted boxes.
    - px-only; clamps to bounds; prevents zero-area boxes
    - separate colors/fill for GT vs predictions
    """
    img = sample["image"]
    if img.mode != "RGBA":
        base = img.convert("RGBA")
    else:
        base = img.copy()

    W, H = base.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None  # PIL will fallback

    # compute widths & fills
    w = width if width is not None else _auto_width(W, H)
    gt_edge = _rgba(gt_color, 255)
    pred_edge = _rgba(pred_color, 220)
    gt_fill = _rgba(gt_color, int(255 * fill_alpha)) if fill_alpha and fill_alpha > 0 else None
    pred_fill = _rgba(pred_color, int(255 * (fill_alpha * 0.7))) if fill_alpha and fill_alpha > 0 else None

    # ----- draw GT -----
    gt_boxes_raw = sample.get(gt_key, []) or []
    gt_boxes = _normalize_pred_boxes(gt_boxes_raw, W, H)  # (x1,y1,x2,y2,conf)
    for i, (x1, y1, x2, y2, _) in enumerate(gt_boxes):
        box = [x1, y1, x2, y2]
        
        if gt_fill is not None:
            draw.rectangle(box, fill=gt_fill)
        draw.rectangle(box, outline=gt_edge, width=w)

        # if gt_labels:
        #     text = gt_labels[i] if i < len(gt_labels) else str(i)
        #     _draw_label(draw, (x1, max(0, y1 - 16 - 4)), text, font)

    # ----- draw Predictions -----
    preds = _normalize_pred_boxes(pred_bboxes, W, H)
    for i, (x1, y1, x2, y2, conf) in enumerate(preds):
        box = [x1, y1, x2, y2]
        print(box)
        # if pred_fill is not None:
        #     draw.rectangle(box, fill=pred_fill)
        draw.rectangle(box, outline=pred_edge, width=w)

        # tag = None
        # if pred_labels and i < len(pred_labels):
        #     tag = pred_labels[i]
        # if show_conf and conf is not None:
        #     tag = f"{tag} | conf={conf:.2f}" if tag else f"conf={conf:.2f}"
        # if tag:
        #     _draw_label(draw, (x1, min(H - 20, y2 + 4)), tag, font)

    return Image.alpha_composite(base, overlay).convert("RGB")