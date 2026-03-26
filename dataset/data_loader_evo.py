import os
import json

from dataset.textvqa import textvqa_ds
from dataset.vstar import vstar_ds
from dataset.mmstar import mstar_ds
from dataset.mmmu import mmmu_ds
from dataset.blink import blink_ds
from dataset.mathvision import mathvision_ds
from dataset.mathverse import mathverse_ds
from dataset.mmmu_pro import mmmu_pro_ds


def load_data(data_id, local_dir, config, args=None):
    mod = ['train']
    
    if data_id.split('/')[1] == "Visual-CoT": 
        ds = textvqa_ds(data_id, local_dir, mod, config, args)
        
    elif data_id.split('/')[1] == "vstar_bench":
        ds = vstar_ds(data_id, local_dir, mod, config, args)

    elif data_id.split('/')[1] == "MMStar":
        ds = mstar_ds(data_id, local_dir, mod, config, args)

    elif data_id.split('/')[1] == "MMMU_Pro":
        ds = mmmu_pro_ds(data_id, local_dir, mod, config, args)

    elif data_id.split('/')[1] == "MathVision":
        ds = mathvision_ds(data_id, local_dir, mod, config, args)

    elif data_id.split('/')[1] == "MMMU":
        ds = mmmu_ds(data_id, local_dir, mod, config, args)

    elif data_id.split('/')[1] == "MathVerse":
        ds = mathverse_ds(data_id, local_dir, mod, config, args)
    

    return ds



# Thinking about the MMMU-Pro, MMStar, and Vstar data type and attributes:
# I want to split the direct answer into several stages. What are the stage could be helpful for these benchmarks. 
# 
# Rules: 
# - Each stage should be generalizble to all the data from these benchmarks. 
# - Each stage contains stage-specific prompts to the same VLM model to generate the stage-specific output.
# - Each stage should provide output that is valuable reasoning to the final answer generation. What are the options?
# - Each stage should be able to be used in a pipeline and does not rely on external tools or information. 
# 
# Insights
# - Try to think about what reasoning steps that a human would take to answer the question from these benchmarks.
# - Then without using external tools or information, how can we design a pipeline to answer the question.