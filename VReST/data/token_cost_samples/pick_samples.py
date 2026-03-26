import json
import random

input_file = "/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/data/MMMU_Pro/export/mmmu_pro_standard_10.jsonl"
output_file = "/home/mohan/InternVL2-main/prompt_evolution_direction/Baselines/VReST/data/token_cost_samples/mmmu_pro_standard_10_100.jsonl"
sample_size = 100

# Read all lines
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Randomly sample 100 lines
sampled_lines = random.sample(lines, min(sample_size, len(lines)))

# Write to new JSONL file
with open(output_file, "w", encoding="utf-8") as f:
    for line in sampled_lines:
        f.write(line)

print(f"✅ Saved {len(sampled_lines)} samples to {output_file}")
