import os 
import json
from datasets import load_dataset, load_from_disk, Features, Value, Sequence, Image
from tqdm import tqdm
from pathlib import Path
from huggingface_hub import snapshot_download

def blink_ds(data_id, local_dir, mod, config, args):
    task_names = ['Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence', 'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization', 'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence', 'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity']
    for task_name in task_names:
        ds = load_dataset(data_id, task_name, cache_dir=local_dir)

        ds = ds.rename_column("question", "question")
        
    # ds = load_dataset(data_id, task_names, cache_dir=local_dir)
    
    return ds
