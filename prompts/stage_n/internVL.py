import torch
from utils.bboxes_tok import _maybe_to_pil

from dataset.mmmu_pro import format_mmmu_pro_sample
from dataset.mmmu import format_mmmu_sample
from dataset.mmstar import format_mmstar_sample
from dataset.vstar import format_vstar_sample
from dataset.mathvision import format_mathvision_sample
from dataset.mathverse import format_mathverse_sample

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from transformers import GenerationConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _build_transform(sz=448):
    return T.Compose([
        T.Lambda(lambda im: im.convert("RGB") if im.mode != "RGB" else im),
        T.Resize((sz, sz), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def _dynamic_preprocess(img: Image.Image, image_size=448, min_num=1, max_num=12, use_thumbnail=True):
    # tiling logic from InternVL card (closest aspect grid, crop tiles, optional thumbnail)
    W, H = img.size
    aspect = W / H
    # candidate grids
    grids = sorted({(i,j) for n in range(min_num, max_num+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max_num},
                   key=lambda x: x[0]*x[1])
    def pick_grid():
        best = (1,1); best_diff = float("inf")
        for (a,b) in grids:
            diff = abs(aspect - a/b)
            if diff < best_diff:
                best_diff, best = diff, (a,b)
        return best
    a,b = pick_grid()
    tgtW, tgtH = a*image_size, b*image_size
    tiles = []
    resized = img.resize((tgtW, tgtH), Image.BICUBIC)
    for r in range(b):
        for c in range(a):
            box = (c*image_size, r*image_size, (c+1)*image_size, (r+1)*image_size)
            tiles.append(resized.crop(box))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(img.resize((image_size, image_size), Image.BICUBIC))
    return tiles  # list[PIL]

def pil_list_to_internvl(images, device, image_size=448, max_num=12):
    tfm = _build_transform(image_size)
    all_tiles = []
    num_patches_list = []
    for im in images:
        tiles = _dynamic_preprocess(im, image_size=image_size, max_num=max_num, use_thumbnail=True)
        num_patches_list.append(len(tiles))
        all_tiles.extend(tiles)
    pixel_values = torch.stack([tfm(t) for t in all_tiles]).to(device)  # [total_tiles, 3, 448, 448]
    return pixel_values, num_patches_list



def get_user_text(sample: dict, W: int, H: int, dataset_name: str):
    """Build a unified user prompt for any reasoning stage."""
    
    # --- (1) Base question text (formatted for MMMU-Pro) ---
    if dataset_name == "MMMU_Pro":
        full_prompt, labels, gt = format_mmmu_pro_sample(sample)  # e.g., question + A)-J) options
    elif dataset_name == "MMStar":
        full_prompt, labels, gt = format_mmstar_sample(sample)  # e.g., question + A)-J) options
    elif dataset_name == "vstar_bench":
        full_prompt, labels, gt = format_vstar_sample(sample)  # e.g., question + A)-J) options
    elif dataset_name == "MathVision":
        full_prompt, labels, gt = format_mathvision_sample(sample)
    elif dataset_name == "MathVerse":
        full_prompt, labels, gt = format_mathverse_sample(sample)
    elif dataset_name == "MMMU":
        full_prompt, labels, gt = format_mmmu_sample(sample)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    full_prompt = full_prompt.strip()

    # --- (2) Context section: image info and question ---
    image_info = f"Image info: W={W}, H={H}."
    context_block = f"[CONTEXT]\n{full_prompt}\n{image_info}\n"

    # --- (3) Output formatting rules ---
    ofmt = (
        "Output: ONE JSON object only; lowercase keys; no prose/markdown/code fences.\n"
        "Do not include explanations outside JSON.\n"
    )

    # --- (4) Inject into each reasoning stage dynamically ---
    wh = f"{context_block}\n{ofmt}"
    
    full_prompt = (full_prompt, gt)

    return (
        f"{wh}"
        f'Schema: {{"answer":"<str>"}}\n'
        f'Task: Directly answer the question using the visible image(s) and options only. '
        f'Do not repeat the question or options.'
        f'If the question is a multiple choice question, output the chosen option letter <A|B|C|D|...>.'
        f'If the question is not a multiple choice question, output the final answer <str>.'
    ), full_prompt


def run_internVL(model, tokenizer, sample, return_debug=True, config=None, system_prompt=None, logger=None):
    dataset_name = config["dataset"]["data_id"].split("/")[-1]
    if dataset_name == "MMMU_Pro" or dataset_name == "MMMU":
        if "vision" in sample['config_name']:
            original_image = _maybe_to_pil(sample["image"])
            current_images = [original_image]
        else:
            num_images = 8
            original_images = []
            for i in range(num_images-1):
                if sample[f"image_{i+1}"]:
                    original_images.append(_maybe_to_pil(sample[f"image_{i+1}"]))
            current_images = original_images
    else:
        original_image = _maybe_to_pil(sample["image"])
        current_images = [original_image]
        
    images = current_images
    
    W, H = images[0].size
    user_text, full_prompt = get_user_text(sample, W, H, dataset_name)
    
    # parts = []

    # if system_prompt:
    #     parts.append(system_prompt.strip())

    # # Optionally reference images in text if multiple / MathVision
    # if len(images) > 1 or dataset_name == "MathVision":
    #     for i in range(len(images)):
    #         parts.append(f"IMG {i} = image {i+1}.")

    # parts.append(user_text.strip())
    # question = "\n".join(parts)

    parts = []
    if system_prompt:
        parts.append(system_prompt.strip())
    if len(images) > 1 or dataset_name == "MathVision":
        for i in range(len(images)):
            parts.append(f"Image-{i+1}: <image>")
    else:
        parts.append("<image>")
    parts.append(user_text.strip())
    question = "\n".join(parts)

    # 2) Convert PIL images -> pixel_values tensor + num_patches_list
    pixel_values, num_patches_list = pil_list_to_internvl(images, device=model.device, image_size=448, max_num=12)

    
    # gen_cfg = GenerationConfig(
    #     max_new_tokens=256,
    #     do_sample=False,
    #     temperature=0.0,
    #     top_p=1.0,
    # )
    gen_cfg = {
        "max_new_tokens": 256,
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 1.0,
        # helpful to include these so InternVL doesn't try to set them and fail
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
    }

    pixel_values = pixel_values.to(model.device, dtype=next(model.parameters()).dtype)

    with torch.no_grad():
        if len(images) == 1:
            # Single image: num_patches_list is optional, but fine to pass as [num_tiles]
            response, _ = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=gen_cfg,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )
        else:
            # Multi-image: REQUI``RED to pass num_patches_list
            response, _ = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question,
                generation_config=gen_cfg,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True,
            )


    answer = response  # raw text from model

    parsed_json = extract_json_from_text(answer)
    rendered_output = render_output(parsed_json, W=W, H=H, logger=logger)
    
    out = {
        "direct_answer_raw": answer,
        "direct_answer_rendered": rendered_output,
        "rendered_answer": rendered_output,
        "gt": full_prompt[1],
    }
    return out
    
    
from typing import Optional, Dict, Any
import json
import re
    
def _fix_incomplete_json(json_str: str) -> str:
    """Attempt to fix incomplete or malformed JSON strings."""
    json_str = json_str.strip()
    
    # Handle incomplete arrays
    if json_str.startswith('[') and not json_str.endswith(']'):
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        json_str += ']' * (open_brackets - close_brackets)
    
    # Handle incomplete objects
    if json_str.startswith('{') and not json_str.endswith('}'):
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        json_str += '}' * (open_braces - close_braces)
    
    # Fix trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    # Handle incomplete string values
    if json_str.count('"') % 2 == 1:
        json_str += '"'
    
    return json_str    

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from raw model output text that may contain markdown code blocks.
    
    Args:
        text: Raw text output from the model
        
    Returns:
        Parsed JSON object if found and valid, None otherwise
    """
    if not text or not text.strip():
        return None
    
    # First, try to find JSON in markdown code blocks
    json_pattern = r'```(?:json)?\s*(.*?)\s*```'
    matches = re.findall(json_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        cleaned_json = match.strip()
        if cleaned_json:
            try:
                return json.loads(cleaned_json)
            except json.JSONDecodeError:
                # Try to fix common issues
                fixed_json = _fix_incomplete_json(cleaned_json)
                try:
                    return json.loads(fixed_json)
                except json.JSONDecodeError:
                    return "Invalid JSON format"
                    continue
    
    # If no markdown blocks found, try to find JSON directly in text
    # Look for patterns that start with { or [
    json_candidates = re.findall(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
    
    for candidate in json_candidates:
        try:
            return json.loads(candidate.strip())
        except json.JSONDecodeError:
            # Try to fix and parse
            fixed = _fix_incomplete_json(candidate.strip())
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                continue
    
    return None


def render_output(parsed_json: dict, W: int = None, H: int = None, logger = None):
    """
    Render the raw text output from each stage into a formatted string for the blackboard.
    """
    try:
        answer = parsed_json.get("answer", "unknown")
        return answer
        
    except Exception as e:
        logger.warning(f"Answer parsing failed: {e}")
        return f"Unknown answer: {parsed_json}"
