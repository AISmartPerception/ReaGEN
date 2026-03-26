from sys_prompt import sys_prompts
import yaml
from numpy import zeros

    

def get_CoT_vector(stages: list, max_length: int, stage_pool: dict):
    order_vector = zeros(max_length)-1
    temperature_vector = zeros(max_length)-1
    max_length_vector = zeros(max_length)-1
    stage_wise_prompt = {}
    for i, stage in enumerate(stages):
        if stage not in stage_pool:
            raise ValueError(f"Stage {stage} not found in stage_pool")
        else:
            order_vector[i] = stage_pool[stage]["id"]
            max_length_vector[i] = stage_pool[stage]["max_length"]
            temperature_vector[i] = stage_pool[stage]["temperature"]
            stage_wise_prompt[stage] = stage_pool[stage]["prompt"]
    
    return order_vector, temperature_vector, max_length_vector, stage_wise_prompt

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    stage_pool = {}
    for i, stage in enumerate(sys_prompts.keys()):
        stage_pool[stage] = {}
        stage_pool[stage]["id"] = i
        stage_pool[stage]["prompt"] = sys_prompts[stage]
        stage_pool[stage]["max_length"] = config["inference"][stage]["max_new_tokens"]
        stage_pool[stage]["temperature"] = config["inference"][stage]["temperature"]


    stages = config["inference"]["stages"]
    order_vector, temperature_vector, max_length_vector, stage_wise_prompt = get_CoT_vector(stages, max_length=len(stage_pool)-1, stage_pool=stage_pool)
    with open("CoT_vector.txt", "w") as f:
        f.write(f"order_vector: {order_vector}\n")
        f.write(f"temperature_vector: {temperature_vector}\n")
        f.write(f"max_length_vector: {max_length_vector}\n")
        f.write(f"stage_wise_prompt: {stage_wise_prompt}\n")

    for stage in stage_wise_prompt:
        # print(stage)
        print(stage_wise_prompt[stage])
   
