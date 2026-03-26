import os 
import json
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import concatenate_datasets
from datasets import Dataset
import ast
import re
import string

subjects = ["History", "Art", "Design", "Literature"]
difficulties = ["Medium", "Hard"]

def format_mmmu_pro_sample(sample):
    # Extract core fields
    options = ast.literal_eval(sample["options"])  # convert string list -> Python list
    
    # question_clean = re.sub(r"<image\s*\d+>", "", question).strip()

    
    # Format options as A), B), C) ...
    option_labels = list(string.ascii_uppercase)
    formatted_options = "\n".join(
        [f"{option_labels[i]}) {opt}" for i, opt in enumerate(options)]
    )

    labels = ",".join([option_labels[i] for i, _ in enumerate(options)])
    if 'vision' in sample['config_name']:
        # full_prompt = f"Options:\n{formatted_options}"
        full_prompt =f"Answer the question shown in the image."
    else:
        # Build the full text prompt
        question = sample["question"]
        full_prompt = f"{question.strip()}\n\nOptions:\n{formatted_options}"
    
    gt = (sample["answer"], options[option_labels.index(sample["answer"])])
    
    return full_prompt, labels, gt


from utils.resize import resize_keep_ratio

def apply_resize(example,
                 max_side=2048,         # keep aspect; cap longest side
                 max_megapixels=1.0,    # also cap total pixels (~1MP). Set None to disable
                 patch_multiple=16):    # round to ViT patch multiple; set 16 or None as needed
    # Process images 1 to 7 (not all rows have all 7 images)
    for image_id in range(1, 8):
        image_key = f"image_{image_id}"
        if image_key in example and example[image_key] is not None:
            img = example[image_key].convert("RGB")
            resized, new_size = resize_keep_ratio(
                img,
                max_side=max_side,
                patch_multiple=patch_multiple
            )
            example[f"orig_size_{image_id}"] = list(img.size)
            example[f"new_size_{image_id}"] = list(new_size)
            example[image_key] = resized
    return example


def mmmu_pro_ds(data_id, local_dir, mod, config, args):
    option_str = ""
    for option in config["dataset"]["options"]:
        if str(10) in option:
            option_str += "_standard_10_options"
        elif str(4) in option:
            option_str += "_standard_4_options"
        elif "vision" in option:
            option_str += "_vision"
    
    ds_dir = local_dir + f"/mmmu_pro{option_str}.dataset"

    if os.path.exists(ds_dir):
    # if False:
        ds = load_from_disk(ds_dir)
        
        # reverse the order of the ds
        # ds = ds.select(range(len(ds)-1, -1, -1))
        if config["gen_training"]["token_cost"]:
            test_samples = []
            token_cost_samples_dir = config["paths"]["token_cost_samples_dir"]
            if "standard_10_options" in option_str:
                with open(os.path.join(token_cost_samples_dir, "mmmu_pro_standard_10_100.jsonl"), "r") as f:
                    token_cost_samples = [json.loads(line) for line in f]
            elif "standard_4_options" in option_str:
                with open(os.path.join(token_cost_samples_dir, "mmmu_pro_standard_4_100.jsonl"), "r") as f:
                    token_cost_samples = [json.loads(line) for line in f]

            selected_ids = [sample["orig_id"] for sample in token_cost_samples]
            for sample in ds:
                if sample['id'] in selected_ids:
                    test_samples.append(sample)
            test_ds = Dataset.from_list(test_samples)
            return test_ds
        print(f"Loaded data {data_id} from {ds_dir}")
    
    else:
        ds = load_dataset(data_id, config["dataset"]["options"][0], cache_dir=local_dir)['test']
        ds = ds.add_column("config_name", [config["dataset"]["options"][0]]*len(ds))
        
        if "vision" in option_str:
            print("Resizing images to 856")
            ds = ds.map(
                apply_resize,
                fn_kwargs={"max_side": 725, "max_megapixels": 1.0, "patch_multiple": 16},
                desc="Resize images with preserved aspect ratio",
                load_from_cache_file=False,
            )
        else:
            print("Resizing images to 1248")
            ds = ds.map(
                apply_resize,
                fn_kwargs={"max_side": 1248, "max_megapixels": 1.0, "patch_multiple": 16},
                desc="Resize images with preserved aspect ratio",
                load_from_cache_file=False,
            )
        
        ds.save_to_disk(ds_dir)
        print(f"Saved data {data_id} to {ds_dir}")
    
    
    # ds = ds.select(range(400))
    
    return ds