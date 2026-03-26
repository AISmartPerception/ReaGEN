import os 
import json
from datasets import load_dataset, load_from_disk, Features, Value, Sequence
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import concatenate_datasets, get_dataset_config_names
from datasets import Dataset
import ast
import re
import string
from PIL import Image

# subjects = ["History", "Art", "Design", "Literature"]
# difficulties = ["Medium", "Hard"]

def format_mmmu_sample(sample):
    # Extract core fields
    options = ast.literal_eval(sample["options"])  # convert string list -> Python list
    
    # question_clean = re.sub(r"<image\s*\d+>", "", question).strip()

    
    # Format options as A), B), C) ...
    option_labels = list(string.ascii_uppercase)
    formatted_options = "\n".join(
        [f"{option_labels[i]}) {opt}" for i, opt in enumerate(options)]
    )

    labels = ",".join([option_labels[i] for i, _ in enumerate(options)])
    # if 'vision' in sample['config_name']:
    #     full_prompt = f"Options:\n{formatted_options}"
    # else:
    #     # Build the full text prompt
    question = sample["question"]
    full_prompt = f"{question.strip()}\n\nOptions:\n{formatted_options}"
    
    if sample["answer"] in option_labels:
        if len(options) == 0:
            gt = (sample["answer"], sample["answer"])
        else:
            gt = (sample["answer"], options[option_labels.index(sample["answer"])])
    else:
        gt = (sample["answer"], sample["answer"])
    
    return full_prompt, labels, gt


from utils.resize import resize_keep_ratio


IMAGE_KEYS = [f"image_{i}" for i in range(1, 8)]  # image_1 ... image_7

def add_multi_image_size(example):
    total_area = 0
    max_side = 0

    for key in IMAGE_KEYS:
        img = example.get(key, None)
        if img is None:
            continue
        # img should be a PIL.Image.Image
        w, h = img.size
        area = w * h
        total_area += area
        max_side = max(max_side, w, h)

    example["img_total_area"] = total_area
    example["img_max_side"] = max_side
    return example
         # older Pillow


def resize_keep_ratio(
    img: Image.Image,
    max_side: int = 2048,
    patch_multiple: int | None = 16,
):
    w, h = img.size
    if w == 0 or h == 0:
        return img, (w, h)

    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img, (w, h)

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    if patch_multiple is not None and patch_multiple > 1:
        new_w = max(patch_multiple, (new_w // patch_multiple) * patch_multiple)
        new_h = max(patch_multiple, (new_h // patch_multiple) * patch_multiple)

    if new_w <= 0 or new_h <= 0:
        return img, (w, h)

    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, (new_w, new_h)


def apply_resize_conditional(
    example,
    max_side=1864,
    area_threshold=2_500_000,  # resize only if total pixels > 4MP
    patch_multiple=16,
):
    # assume you've already computed img_total_area;
    # if not, quickly compute it here:


    if example["img_total_area"] <= area_threshold:
        return example  # small enough, skip resizing

    for image_id in range(1, 8):
        image_key = f"image_{image_id}"
        img = example.get(image_key, None)
        if img is not None:
            img = img.convert("RGB")
            resized, new_size = resize_keep_ratio(
                img,
                max_side=max_side,
                patch_multiple=patch_multiple,
            )
            # example[f"orig_size_{image_id}"] = list(img.size)
            # example[f"new_size_{image_id}"] = list(new_size)
            example[image_key] = resized

    return example


def mmmu_ds(data_id, local_dir, mod, config, args):
    # option_str = ""
    # for option in config["dataset"]["options"]:
    #     option_str += "_" + option.split(' ')[0]
    
    
    ds_dir = local_dir + f"/mmmu.dataset"
    
    if os.path.exists(ds_dir):
    # if False:
        ds = load_from_disk(ds_dir)
        

        print(f"Loaded data {data_id} from {ds_dir}")
    

    else:

        target_subsets = get_dataset_config_names(data_id)

        sub_datasets = []
        for subset_name in target_subsets:
            ds_all = load_dataset(data_id, subset_name, cache_dir=local_dir)
            ds_dev  = ds_all["validation"]
            ds_test = ds_all["dev"]

            ds_sub = concatenate_datasets([ds_dev, ds_test])
            ds_sub = ds_sub.add_column("config_name", [subset_name] * len(ds_sub))

            sub_datasets.append(ds_sub)

        ds = concatenate_datasets(sub_datasets)

        ds = ds.map(
            add_multi_image_size,
            desc="Compute multi-image sizes",
            load_from_cache_file=False,
        )

        # sort from largest total area to smallest
        ds = ds.sort("img_total_area", reverse=True)


        # resize images as before
        # ds = ds.map(
        #     apply_resize,
        #     fn_kwargs={"max_side": 1240, "max_megapixels": 1.0, "patch_multiple": 16},
        #     desc="Resize images with preserved aspect ratio",
        #     load_from_cache_file=False,
        # )

        ds.save_to_disk(ds_dir)
        print(f"Saved data {data_id} to {ds_dir}")
    
    
    # ds = ds.select(range(400))

    # if args.partition == 0:
        
    #     ds = ds.select(range(400))
        
    ds = ds.map(
        apply_resize_conditional,
        fn_kwargs={"max_side": 1024, "area_threshold": 2_000_000, "patch_multiple": 16},
        desc="Conditionally resize large multi-image samples",
        load_from_cache_file=False,
    )
    # ds = ds.select(range(len(ds)-1, -1, -1))
    #     # ds = ds.select(range(200))
    #     # ds = ds.select(range(200, 300))
    # elif args.partition == 1:
    #     ds = ds.select(range(400, 800))
    # elif args.partition == 2:
    #     ds = ds.select(range(800, len(ds)))
    
    
    return ds