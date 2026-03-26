import os
import json
from huggingface_hub import snapshot_download
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset

from utils.resize import apply_resize

def _stem(x):
    # Handles plain strings and (later) Image dicts
    if isinstance(x, str):
        return Path(x).stem
    if isinstance(x, dict) and "path" in x:
        return Path(x["path"]).stem
    return str(x)

def add_bbox(example):
    # get image path string (relative, e.g. direct_attributes/sa_4690.jpg)
    img_path = example["image"].filename  # works after cast_column
    base, _ = os.path.splitext(img_path)
    json_path = base + ".json"   # same stem, .json extension
    
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        example["target_object"] = data.get("target_object", [])
        example["bbox"] = data.get("bbox", [])
    else:
        example["target_object"] = []
        example["bbox"] = []
    
    return example

def to_abs(ex, local_dir):
    ex["image"] = os.path.join(local_dir, ex["image"])
    
    # Extract image_id from filename more robustly
    filename = ex["image"].split("/")[-1]  # Get filename
    filename_no_ext = filename.split(".")[0]  # Remove extension
    ex["image_id"] = int(filename_no_ext.split("_")[-1])

    return ex


import re
def format_vstar_sample(sample):
    full_prompt = sample["question"]
    labels = re.findall(r'\(([A-Z])\)', full_prompt)
    labels = ",".join(labels)

    # matches = re.findall(r'([A-Z]):\s*([^,]+)', full_prompt)
    matches = re.findall(r'\(([A-Z])\)\s*(.+)', full_prompt)
    options_dict = {letter: option.strip() for letter, option in matches}
    # labels = 'A,B,C,D'

    gt = (sample["answer"], options_dict[sample["answer"]])
    return full_prompt, labels, gt




def vstar_ds(data_id, local_dir, mod, config, args):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=data_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )

    ds_dir = local_dir + f"/vstar_bench.dataset"
        
    if os.path.exists(ds_dir):
    # if False:
        ds = load_from_disk(ds_dir)
        print(f"Loaded data {data_id} from {ds_dir}")

        if config["gen_training"]["token_cost"]:
            test_samples = []
            token_cost_samples_dir = config["paths"]["token_cost_samples_dir"]
            with open(os.path.join(token_cost_samples_dir, "VStar_100.jsonl"), "r") as f:
                token_cost_samples = [json.loads(line) for line in f]

            # ds = ds.rename_column("question_id", "id")
            selected_ids = [sample["question_id"] for sample in token_cost_samples]
            for sample in ds:
                if sample['id'] in selected_ids:
                    test_samples.append(sample)
            test_ds = Dataset.from_list(test_samples)
            return test_ds

        return ds
        
    
    # ds = load_dataset("json", data_files=f"{local_dir}/test_questions.jsonl")['train']
    ds = load_dataset("json", data_files=f"{local_dir}/test_questions_updated.jsonl")['train']

    ds = ds.map(to_abs, 
        fn_kwargs={"local_dir": local_dir}, 
        desc="Attach VSTAR image paths",
        load_from_cache_file=False
    )
    # ds = ds.cast_column("image", Image())
    ds = ds.cast_column("image", HFImage())

    # ds = ds.map(
    #     apply_resize,
    #     fn_kwargs={"max_side": 1664, "max_megapixels": 1.0, "patch_multiple": 16},
    #     desc="Resize images with preserved aspect ratio",
    #     load_from_cache_file=False,
    # )

    # size_count = {}
    # for ex in ds:
    #     size = ex["image"].size
    #     if size not in size_count:
    #         size_count[size] = 0
    #     size_count[size] += 1
    

    ds = ds.rename_column("text", "question")
    ds = ds.rename_column("question_id", "id")
    ds = ds.rename_column("label", "answer")

    ds = ds.shuffle(seed=args.seed)
    # ds = ds.select(range(100))

    
    os.makedirs(ds_dir, exist_ok=True)
    ds.save_to_disk(ds_dir)
    print(f"Saved data {data_id} to {ds_dir}")


    return ds