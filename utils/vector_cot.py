import os
import torch


def encode_stages(stages, stage_pool = None):
    if stage_pool is None:
        stage_pool = {stage: i for i, stage in enumerate(stages)}
    
    max_length = len(stage_pool)
    cot = [stage_pool[stages[i]] if i < len(stages) else max_length for i in range(max_length)]
    return torch.tensor(cot), stage_pool


def decode_stages(cot, stage_pool, pred_length=None):
    # stages = [stage_pool[c.item()] if c.item() < len(stage_pool) else None for c in cot]
    cot = cot.detach().cpu().tolist()
    stages = []
    for i, stage_id in enumerate(cot):
        # if pred_length is not None:
        #     if i == pred_length:
        #         return stages
        if stage_id < len(stage_pool):
            for stage_nm in stage_pool:
                if stage_pool[stage_nm] == stage_id:
                    stages.append(stage_nm)
                    break
        
    return stages