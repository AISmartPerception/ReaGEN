# draw_heatmap.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import torch

# ---- span mapping (same logic as in your module) --------------------------
def token_spans_for_lines(chat_str: str, tokenizer, lines: List[str]) -> List[List[int]]:
    try:
        enc = tokenizer(
            chat_str,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        if enc.offset_mapping is None:
            return [[] for _ in lines]
        offs = enc.offset_mapping[0].tolist()
    except Exception:
        return [[] for _ in lines]

    spans: List[List[int]] = []
    for ln in lines:
        i0 = chat_str.find(ln)
        if i0 < 0:
            spans.append([])
            continue
        i1 = i0 + len(ln)
        idxs = [ti for ti, (a, b) in enumerate(offs) if a >= i0 and b <= i1]
        spans.append(idxs)
    return spans


def group_mem_lines_by_stage(mem_lines: List[str]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {"DESCRIBE": [], "BBOX": [], "COLOR": []}
    for i, line in enumerate(mem_lines):
        if line.startswith("[IMG 1] CROP") or "from BBOX" in line:
            groups["BBOX"].append(i)
        elif "from DESCRIBE" in line:
            groups["DESCRIBE"].append(i)
        elif "from COLOR" in line:
            groups["COLOR"].append(i)
    # drop empties
    return {k: v for k, v in groups.items() if v}

# ---- build token-by-stage attention ---------------------------------------
def build_token_by_stage_attention(
    gen_attentions,                    # list[T] of tuples[L] of tensors [B,H,1,KV]
    stage2spans: Dict[str, List[List[int]]],
    avg_layers: bool = True,
    avg_heads: bool = True,
    row_normalize: bool = True,
    input_len: Optional[int] = None,   # clip to input tokens (exclude generated)
) -> Tuple[np.ndarray, List[str]]:
    stages = list(stage2spans.keys())
    S = len(stages)
    T = len(gen_attentions)
    H = np.zeros((T, S), dtype=np.float32)
    # print("input_len", input_len)
    # union token sets per stage
    stage_token_sets: List[List[int]] = []
    for s in stages:
        union = set()
        for idxs in stage2spans[s]:
            union.update(idxs)
        stage_token_sets.append(sorted(list(union)))

    for t, step in enumerate(gen_attentions):
        if t == 0:
            continue

        # step: tuple/list over layers, each [B,H,1,KV]
        if avg_layers and avg_heads:
            A = torch.stack([Al[0, :, 0, :] for Al in step], 0).mean(0).mean(0)  # [KV]
        else:
            A = torch.stack([Al[0, :, 0, :].mean(0) for Al in step], 0).mean(0)  # [KV]

        # cast to float32 to avoid bfloat16 numpy error
        A = A.detach().to(torch.float32).cpu().numpy()  # [KV]
        kv_len = A.shape[0]

        for s_idx, token_idxs in enumerate(stage_token_sets):
            if not token_idxs:
                H[t, s_idx] = 0.0
                continue
            if input_len is not None:
                valid = [i for i in token_idxs if i < kv_len and i < input_len]
            else:
                valid = [i for i in token_idxs if i < kv_len]
            H[t, s_idx] = float(A[valid].sum()) if valid else 0.0

        if row_normalize:
            rs = H[t].sum()
            if rs > 0:
                H[t] /= rs

    return H, stages

# ---- plotting -------------------------------------------------------------
def save_stage_output_heatmap(
    chat: str,
    tokenizer,
    gen_attentions,
    mem_lines: List[str],
    out_png: str,
    gen_token_ids: Optional[List[int]] = None,
    title: str = "Final-token → Stage-outputs attention",
    also_save_csv: Optional[str] = None,
    input_len: Optional[int] = None,  # pass inputs["input_ids"].shape[1]
):
    # 1) span indices per MEM line
    spans_all = token_spans_for_lines(chat, tokenizer, mem_lines)

    # 2) group lines by stage and gather spans per stage
    groups = group_mem_lines_by_stage(mem_lines)
    # print(f"mem_lines: {mem_lines}")
    # print(f"groups: {groups}")
    stage2spans = {stage: [spans_all[i] for i in idxs] for stage, idxs in groups.items()}
    if not stage2spans:
        # nothing to plot
        plt.figure()
        plt.title(title + " (no MEM lines matched)")
        plt.savefig(out_png, dpi=180)
        plt.close()
        return

    # 3) matrix [T,S]
    H, stages = build_token_by_stage_attention(
        gen_attentions, stage2spans, input_len=input_len
    )

    # 4) y-axis labels (optional pretty tokens)
    y_labels = None
    if gen_token_ids:
        toks = tokenizer.convert_ids_to_tokens(gen_token_ids)
        y_labels = [t.replace("▁", " ").replace("Ġ", " ") for t in toks]

    # 5) plot
    plt.figure()
    plt.imshow(H, aspect="auto")
    plt.title(title)
    plt.xlabel("Stage outputs")
    plt.ylabel("Generated token step")
    plt.xticks(range(len(stages)), stages, rotation=45, ha="right")
    if y_labels and len(y_labels) == H.shape[0]:
        step = max(1, len(y_labels) // 50)
        yticks = list(range(0, len(y_labels), step))
        plt.yticks(yticks, [y_labels[i] for i in yticks])
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    # 6) optional CSV dump
    if also_save_csv:
        import csv
        with open(also_save_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["token_step"] + stages)
            for t in range(H.shape[0]):
                w.writerow([t] + list(H[t]))
