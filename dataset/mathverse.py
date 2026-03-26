import os 
import json
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image, concatenate_datasets
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download
import ast
import re
import string

from utils.resize import apply_resize

def format_mathverse_sample(sample):
    # Extract core fields
    # full_prompt = sample["question"]
    # # labels = 'A,B,C,D'
    # gt = (sample["answer"], sample["answer"])
    if sample["problem_version"] == "Vision Only":
        full_prompt =f"Answer the question shown in the image."
        gt = (sample["answer"], sample["answer"])
        return full_prompt, None, gt

    if sample["question_type"] == "multi-choice":
        full_prompt = sample["question"]
        labels = 'A,B,C,D'

        # matches = re.findall(r'([A-Z]):\s*([^,]+)', full_prompt)
        
        # pattern = r'(?:\(?([A-Z])\)?)[\:\.\)]\s*([^\n,]+)'
        # pattern = re.compile(
        #     r'^\s*\(?([A-Z])\)?\s*[:\.\)]?\s*(.+?)\s*$',
        #     flags=re.MULTILINE
        # )
        
        # pattern = re.compile(
        #     r'^\s*\(?([A-Z])\)?\s*[:\.\)]\s*(.+?)\s*$',
        #     flags=re.MULTILINE,
        # )
        # pattern = re.compile(
        #     r'^\s*\(?([A-Z])\)?\s*[:\.\)]\s*(.*)$',
        #     flags=re.MULTILINE,
        # )
        
        pattern = re.compile(
            r'^\s*\(?([A-Z])\)?[ \t]*[:\.\)][ \t]*(.*)$',
            flags=re.MULTILINE,
        )
        matches = pattern.findall(full_prompt)
        # matches = re.findall(pattern, full_prompt)
        # matches = re.findall(r'\(([A-Z])\)\s*(.+)', full_prompt)
        options_dict = {letter: option.strip() for letter, option in matches}
        
        raw_ans = str(sample["answer"]).strip()
        if raw_ans == "True" or raw_ans == "False":
            for letter, option in options_dict.items():
                if option.strip().lower() == raw_ans.lower():
                    ans_letter = letter
                    break
            # ans_letter = raw_ans
        else:
            if len(raw_ans) < 5:
                m = re.search(r"[A-F]", raw_ans)
                if m is None:
                    # True, False
                    for letter, option in options_dict.items():
                        if option.strip().lower() == raw_ans.lower():
                            ans_letter = letter
                            break
                else:
                    ans_letter = m.group(0) 
            else:
                for letter, option in options_dict.items():
                    if option.strip().lower() == raw_ans.lower():
                        ans_letter = letter
                        break

   
        gt = (ans_letter, options_dict[ans_letter])
    elif sample["question_type"] == "free-form":
        full_prompt = sample["question"]
        # labels = 'A,B,C,D'
        gt = (sample["answer"], sample["answer"])
    
    return full_prompt, None, gt


def add_image_size(example):
    img = example["image"]              # this is a PIL.Image.Image
    w, h = img.size                     # (width, height)
    example["img_area"] = w * h
    example["img_max_side"] = max(w, h)
    return example

target_n = 1500

def mathverse_ds(data_id, local_dir, mod, config, args):
    if config["dataset"]["vison_only"] or config["training_data"]["vison_only"]:
        ds_dir = local_dir + f"/mathverse_vision.dataset"
    else:
        ds_dir = local_dir + f"/mathverse.dataset"

    
    # if os.path.exists(ds_dir):
    if False:
        ds = load_from_disk(ds_dir)
        print(f"Loaded data {data_id} from {ds_dir} with length {len(ds)}")

    else:
        ds = load_dataset(data_id, 'testmini', cache_dir=local_dir)['testmini']
        ds = ds.shuffle(seed=args.seed)
        ds = ds.rename_column("sample_index", "id")
        if config["dataset"]["vison_only"]:
            ds = ds.filter(lambda ex: ex["problem_version"] == "Vision Only")
            print(f"Vision Only: {len(ds)}")
        else:

            non_vision = ds.filter(lambda ex: ex["problem_version"] != "Vision Only")
            print(f"Non-vision: {len(non_vision)}")

            versions = sorted(set(non_vision["problem_version"]))
            num_versions = len(versions)
            target_per_version = target_n // num_versions 
            target_per_type = target_per_version // 2 # half for each type

            buckets = []

            for v in versions:
                ds_v = non_vision.filter(lambda ex, v=v: ex["problem_version"] == v)

                mc_v = ds_v.filter(lambda ex: ex["question_type"] == "multi-choice")
                ff_v = ds_v.filter(lambda ex: ex["question_type"] == "free-form")

                n_mc = min(target_per_type, len(mc_v))
                n_ff = min(target_per_type, len(ff_v))

                mc_sample = mc_v.shuffle(seed=args.seed).select(range(n_mc))
                ff_sample = ff_v.shuffle(seed=args.seed).select(range(n_ff))

                buckets.extend([mc_sample, ff_sample])


            ds = concatenate_datasets(buckets).shuffle(seed=args.seed)


        ds = ds.map(
            apply_resize,
            fn_kwargs={"max_side": 1450, "max_megapixels": 1.0, "patch_multiple": 16},
            desc="Resize images with preserved aspect ratio",
            load_from_cache_file=False,
        )

        # subset = ds.select(range(100))
        
        # ds.save_to_disk(ds_dir)
        # print(f"Saved data {data_id} to {ds_dir} with length {len(ds)}")



    ds = ds.map(add_image_size, desc="Compute image sizes", load_from_cache_file=False)
    ds = ds.sort("img_area", reverse=True)

    # if args.partition == 0:
    #     ds = ds.select(range(500))
    #     # ds = ds.select(range(500))
    #     ds = ds.select(range(len(ds)-1, -1, -1))
        
    # elif args.partition == 1:
    #     ds = ds.select(range(500, 1000))
    #     # ds = ds.select(range(len(ds), -1, -1))
    #     # ds = ds.select(range(len(ds)-1, -1, -1))
        
    # elif args.partition == 2:
    #     ds = ds.select(range(1000, len(ds)))

    return ds

