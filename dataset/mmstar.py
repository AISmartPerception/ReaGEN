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

def format_mmstar_sample(sample):
    # Extract core fields
    full_prompt = sample["question"]
    labels = 'A,B,C,D'

    # matches = re.findall(r'([A-Z]):\s*([^,]+)', full_prompt)
    
    pattern = r'(?:\(?([A-Z])\)?)[\:\)]\s*([^\n,]+)'
    matches = re.findall(pattern, full_prompt)
    # matches = re.findall(r'\(([A-Z])\)\s*(.+)', full_prompt)
    options_dict = {letter: option.strip() for letter, option in matches}

    gt = (sample["answer"], options_dict[sample["answer"]])
    
    return full_prompt, labels, gt


from collections import Counter
from datasets import Dataset

def mstar_ds(data_id, local_dir, mod, config, args):
    ds_dir = local_dir + f"/mmstar.dataset" 
    # if not config["gen_training"]["test_data"] else local_dir + f"/mmstar_test.dataset"
    # if os.path.exists(ds_dir):
    if False:
        ds = load_from_disk(ds_dir)
        #     print(f"Loaded data {data_id} from {ds_dir}")
        #     if args.test:
        #         return ds
        #     else:
        #         ds = ds.select(range(100, len(ds)))
        #         return ds
        ds = ds.class_encode_column("category")
        label_feat = ds.features["category"] 
        id2name = label_feat.int2str
        name2id = label_feat.str2int
        
        # print("All classes (ids):", sorted(set(ds["category"])))
        # print("All classes (names):", sorted({id2name(i) for i in set(ds["category"])}))
        
        def show_counts(tag, dset):
            counts = Counter(dset["category"])
            as_names = {id2name(i): n for i, n in sorted(counts.items())}
            print(f"{tag} counts (id):", dict(sorted(counts.items())))
            print(f"{tag} counts (name):", as_names)

        indices = [
            i for i, ex in enumerate(ds)
            if os.path.exists(f"/data00/XXX/evo_pairs/MMStar_cot_init_attn/cot_init_attn_{ex['id']}.pkl")
        ]
        new_ds = ds.select(indices) 
    
        split = new_ds.train_test_split(
            test_size=0.1,
            stratify_by_column="category",
            seed=args.seed,
        )

        train_ds = split["train"]
        test_ds  = split["test"]

        
        if config["gen_training"]["test_data"]:
            if config["gen_training"]["token_cost"]:
                test_samples = []
                token_cost_samples_dir = config["paths"]["token_cost_samples_dir"]
                with open(os.path.join(token_cost_samples_dir, "MMStar_100.jsonl"), "r") as f:
                    token_cost_samples = [json.loads(line) for line in f]

                selected_ids = [sample["id"] for sample in token_cost_samples]
                for sample in new_ds:
                    if sample['id'] in selected_ids:
                        test_samples.append(sample)
                test_ds = Dataset.from_list(test_samples)
                return test_ds
            else:
                return new_ds
        else:
            return train_ds
        
        # return train_ds, test_ds
        
        
        

    ds = load_dataset(data_id, cache_dir=local_dir)['val']

    ds = ds.shuffle(seed=args.seed)
    # test_ids = ds.select(range(100)).column_names['id']
    ds = ds.rename_column("index", "id")

    # ds = ds.map(
    #     apply_resize,
    #     desc="Resize images with preserved aspect ratio",
    #     load_from_cache_file=False,
    # )
    
    # if config["gen_training"]["test_data"]:
    #     subset = ds.select(range(100))
    #     subset.save_to_disk(ds_dir)
    #     print(f"Saved data {data_id} to {ds_dir}")
    #     return subset
    # else:
    #     ds.save_to_disk(ds_dir)
    #     print(f"Saved data {data_id} to {ds_dir}")
    #     return ds
    # subset = ds.select(range(100))
    # subset.save_to_disk(ds_dir)
    # print(f"Saved data {data_id} to {ds_dir}")
    # return subset

    # ds.save_to_disk(ds_dir)
    # print(f"Saved data {data_id} to {ds_dir}")
    
    return ds


    