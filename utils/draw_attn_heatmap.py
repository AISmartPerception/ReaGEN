import numpy as np
import matplotlib.pyplot as plt
import os


def draw_attn_heatmap(per_layer_mass, out_png, mem_names = None):
    layers = sorted(per_layer_mass.keys())
    if mem_names is None:
        stages = list(dict.fromkeys([k for d in per_layer_mass.values() for k in d.keys()]))
    else:
        stages = list(mem_names)
    # stages = {k for d in per_layer_mass.values() for k in d.keys()}

    M = np.array([[per_layer_mass[l].get(s, 0.0) for s in stages] for l in layers])  # shape (L, S)


    fig = plt.figure(figsize=(max(6, len(stages)*0.7), max(4, len(layers)*0.5)))
    im = plt.imshow(M, aspect='auto')
    plt.colorbar(im, label='fraction of prompt attention')
    plt.yticks(ticks=np.arange(len(layers)), labels=layers)
    plt.xticks(ticks=np.arange(len(stages)), labels=stages, rotation=60, ha='right')
    plt.xlabel('stage')
    plt.ylabel('layer')
    plt.title('Per-layer attention mass by stage')
    plt.tight_layout()

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def draw_attn_heatmap_dataset(atten_mass_dataset):
    stages = atten_mass_dataset[0].keys()
    stages_name = "_".join(stages)
    n_layers = len(atten_mass_dataset[0]['FINAL']['mean_attention_mass'].keys())
    # stages = atten_mass_dataset[0].keys()
    for istage, stage_name in enumerate(stages):
        mem_names = atten_mass_dataset[0][stage_name]['mem_names']
        if len(mem_names) == 0:
            continue

        sum_per_layer_mass = {l: {mem_name: 0.0 for mem_name in mem_names} for l in range(n_layers)}
        for sample_id, stage_results in enumerate(atten_mass_dataset):
            # sum_per_layer_mass += stage_results[stage_name]['mean_attention_mass']
            for l in range(n_layers):
                for k in mem_names:
                    sum_per_layer_mass[l][k] += stage_results[stage_name]['mean_attention_mass'][l][k]


        sum_per_layer_mass = {l: {k: sum_per_layer_mass[l][k] / len(atten_mass_dataset) for k in mem_names} for l in range(n_layers)}
        os.makedirs(f"imgs/{stages_name}", exist_ok=True)
        draw_attn_heatmap(sum_per_layer_mass, f"imgs/{stages_name}/stage{istage}_{stage_name}.png", mem_names)
        # draw_attn_heatmap(stage, f"imgs/attn_mass_per_layer_{stage}.png")
        