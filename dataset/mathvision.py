import os 
import json
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
import ast
import re
import string

from utils.resize import apply_resize

def format_mathvision_sample(sample):
    # Multiple choice question
    if len(sample["options"]) > 0:
        options = sample["options"] #cnvert string list -> Python list

        option_labels = list(string.ascii_uppercase)
        # option_labels = sample["options"]
        formatted_options = "\n".join(
            [f"{option_labels[i]}) {opt}" for i, opt in enumerate(options)]
        )

        labels = ",".join([option_labels[i] for i, _ in enumerate(options)])

        full_prompt = f"{sample['question'].strip()}\n\nOptions:\n{formatted_options}"
        
        gt = (sample["answer"], options[option_labels.index(sample["answer"])])
    else: # Not a multiple choice question
        labels = None

        full_prompt = sample["question"]
        
        gt = (sample["answer"], sample["answer"])
    
    return full_prompt, labels, gt


import random
from datasets import Dataset

def mathvision_ds(data_id, local_dir, mod, config, args):
    ds_dir = local_dir + f"/mathvision.dataset"
    if os.path.exists(ds_dir):
    # if False:
        ds = load_from_disk(ds_dir)
        ds = ds.select(random.sample(range(len(ds)), 100))

        print(f"Loaded data {data_id} from {ds_dir} with length {len(ds)}")
    else:
        ds = load_dataset(data_id, cache_dir=local_dir)['test']
        if args:
            ds = ds.shuffle(seed=args.seed)
        else:
            ds = ds.shuffle(seed=1)
        ds = ds.remove_columns('image')
        
        ds = ds.rename_column("decoded_image", "image")
        # ds = ds.map(
        #     apply_resize,
        #     fn_kwargs={"max_side": 1664, "max_megapixels": 1.0, "patch_multiple": 16},
        #     desc="Resize images with preserved aspect ratio",
        #     load_from_cache_file=False,
        # )
        # ds = ds.map(
        #     apply_resize,
        #     fn_kwargs={"max_side": 1664, "max_megapixels": 1.0, "patch_multiple": 16},
        #     desc="Resize images with preserved aspect ratio",
        #     load_from_cache_file=False,
        # )

        # ds = ds.shuffle(seed=args.seed)
        if args:
            ds = ds.shuffle(seed=args.seed)
        else:
            ds = ds.shuffle(seed=1)
        # ds.save_to_disk(ds_dir)
        # print(f"Saved data {data_id} to {ds_dir}")


    # if args.partition == 0:
    #     # subset = ds.select(range(750))
    #     subset = ds.select(range(1500, len(ds)))
    # elif args.partition == 1:
    #     subset = ds.select(range(750, 1500))
    # elif args.partition == 2:
    #     subset = ds.select(range(750))
    # elif args.partition == 3:
    #     subset = ds.select(range(2600, len(ds)))
    
    # subset = ds.select(random.sample(range(len(ds)), 500))
    
    subset = ds

    # if config["gen_training"]["token_cost"]:
    #     test_samples = []
    #     token_cost_samples_dir = config["paths"]["token_cost_samples_dir"]
    #     with open(os.path.join(token_cost_samples_dir, "MathVision_100.jsonl"), "r") as f:
    #         token_cost_samples = [json.loads(line) for line in f]

    #     selected_ids = [sample["id"] for sample in token_cost_samples]
    #     for sample in subset:
    #         if sample['id'] in selected_ids:
    #             test_samples.append(sample)
    #     subset = Dataset.from_list(test_samples)
    #     return subset
    
    # subset = ds.select(range(1600, 2100))
    # subset = ds.select(range(2100, 2600))
    # subset = ds.select(range(2600, len(ds)))
    return subset