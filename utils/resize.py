from PIL import Image
import math

def round_to_multiple(x, m):
    return max(m, int(x // m) * m)  # floor to multiple m (avoid 0)

def compute_new_size_keep_ratio(w, h, max_side=None, max_area=None, patch_multiple=None):
    """
    Returns (new_w, new_h) that:
      - keeps aspect ratio
      - does not exceed max_side (longest side) if given
      - does not exceed max_area (pixels) if given
      - does not upscale
      - rounds to multiples of patch_multiple if given (e.g., 14 or 16)
    """
    scale = 1.0
    if max_side is not None:
        scale = min(scale, max_side / max(w, h))
    if max_area is not None:
        scale = min(scale, math.sqrt(max_area / (w * h)))
    # avoid upscaling
    scale = min(scale, 1.0)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    if patch_multiple:
        new_w = round_to_multiple(new_w, patch_multiple)
        new_h = round_to_multiple(new_h, patch_multiple)

    # Ensure at least 1 px after rounding
    new_w = max(1, new_w)
    new_h = max(1, new_h)
    return new_w, new_h

def resize_keep_ratio(img, max_side=1024, max_megapixels=None, patch_multiple=16):
    """
    Resize PIL image with preserved aspect ratio.
      - max_side: cap on the longest side (e.g., 768 or 1024)
      - max_megapixels: cap on total pixels (e.g., 1.0 -> ~1MP). Use None to disable
      - patch_multiple: round both sides to multiple of ViT patch size (14 or 16). Use None to disable
    """
    w, h = img.size
    max_area = None if max_megapixels is None else int(max_megapixels * 1_000_000)

    new_w, new_h = compute_new_size_keep_ratio(
        w, h, max_side=max_side, max_area=max_area, patch_multiple=patch_multiple
    )

    if (new_w, new_h) == (w, h):
        return img, (w, h)  # no resize

    # Use high-quality downsampling
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS), (new_w, new_h)

def apply_resize(example,
                 max_side=2048,         # keep aspect; cap longest side
                 max_megapixels=1.0,    # also cap total pixels (~1MP). Set None to disable
                 patch_multiple=16):    # round to ViT patch multiple; set 16 or None as needed
    img = example["image"].convert("RGB")
    resized, new_size = resize_keep_ratio(
        img,
        max_side=max_side,
        patch_multiple=patch_multiple
    )
    example["orig_size"] = list(img.size)
    example["new_size"]  = list(new_size)
    example["image"] = resized
    return example

