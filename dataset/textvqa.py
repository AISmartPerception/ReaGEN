import os
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image
import json
from pathlib import Path
from tqdm import tqdm

def _stem(x):
    # Handles plain strings and (later) Image dicts
    if isinstance(x, str):
        return Path(x).stem
    if isinstance(x, dict) and "path" in x:
        return Path(x["path"]).stem
    return str(x)

FEATURES = Features({
    "question": Value("string"),
    "answer": Value("string"),
    "full_answer": Value("string"),
    "possible_answers": Sequence(Value("string")),
    "image": Value("string"),
    "width": Value("float64"),
    "height": Value("float64"),
    "bboxs": Sequence(Sequence(Value("float64"))),
    "dataset": Value("string"),
    "split": Value("string"),
    "reasoning": Sequence({
        "operation": Value("string"),
        "dependencies": Sequence(Value("int64")),
        "argument": Value("string"),
    }),
    "thought": Value("string"),
})

def to_abs(ex, data_dir):
    ex["image"] = os.path.join(f"{data_dir}/textvqa/train_images/train_images", ex["image"])
    
    return ex

def add_possible(batch, QA_pairs):
    imgs = [_stem(p) for p in batch["image"]]
    qs = batch["question"]
    out = [QA_pairs.get((im, q), []) for im, q in zip(imgs, qs)]
    return {"possible_answers": out, "image_id": imgs}


def textvqa_ds(data_id, local_dir, mod, config, args):

    RAW_URL = [f"https://huggingface.co/datasets/deepcs233/Visual-CoT/resolve/main/metadata/textvqa_cot_{split}.jsonl" for split in mod]

    if os.path.exists(local_dir):
        try:
            ds = load_from_disk(local_dir)
            print(f"Loaded data {data_id} from {local_dir}")
            return ds
        except Exception as e:
            print(f"Cache at {local_dir} is invalid; rebuilding… ({e})")

    # ds = load_dataset("json", data_files={split: RAW_URL}, features=FEATURES)
    data_files = {split: url for split, url in zip(mod, RAW_URL)}
    ds = load_dataset("json", data_files=data_files, features=FEATURES)
    
    for split in mod:
        with open(f"{config['paths']['data_dir']}/textvqa/TextVQA_0.5.1_{split}.json", "r") as f:
            data_json = json.load(f)["data"]

        QA_pairs = {
            (ex["image_id"], ex["question"]): ex["answers"]
            for ex in data_json
        }

        new_features = ds[split].features.copy()
        new_features["possible_answers"] = Sequence(Value("string"))
        new_features["image_id"] = Value("string")

        ds[split] = ds[split].map(
            add_possible,
            batched=True,
            fn_kwargs={"QA_pairs": QA_pairs},
            features=new_features,
            desc="Attach possible answers and image IDs",
        )

        poss = ds[split]["possible_answers"]
        count = sum(bool(a) for a in poss)
        if count != len(ds[split]):
            print(f"Warning: matched {count}/{len(ds[split])} examples.")
            # Show a few unmatched examples for debugging
            bad_idxs = [i for i, a in enumerate(poss) if not a][:3]
            for i in bad_idxs:
                print(ds[split][i])

    
        ds[split] = ds[split].map(to_abs, fn_kwargs={"data_dir": config["paths"]["data_dir"]}, desc="Attach TextVQA image paths", load_from_cache_file=False)
    
        ds[split] = ds[split].cast_column("image", Image())

    os.makedirs(local_dir, exist_ok=True)
    # print(local_dir)
    ds.save_to_disk(local_dir)
    print(f"Downloaded data {data_id} to {local_dir} with length {len(ds[split])}")

    return ds