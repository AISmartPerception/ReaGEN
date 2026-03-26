import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from torchview import draw_graph
from model.model_loader_gen import GENModel
import torch
import torch.nn as nn
import math


def first_linear_in_features(module: nn.Module) -> int:
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    raise ValueError("No Linear() found to infer in_features")

def infer_tri_N(m: int) -> int:
    disc = 1 + 4*m
    N = int((-1 + math.isqrt(disc)) // 2)
    if N*(N+1) != m:
        raise ValueError(f"{m} is not N*(N+1)")
    return N

def make_dummy_batch(model, B=2, Ti=64, Tq=32, L=None, device="cpu"):
    Din = first_linear_in_features(model.proj)              # e.g., 3584
    M_A = first_linear_in_features(model.A_proj)            # e.g., 56
    N = infer_tri_N(M_A)                                    # e.g., 7
    num_classes = getattr(model, "num_classes", 8)
    max_L = getattr(model.pos_emb, "num_embeddings", 7)
    L = max_L if L is None else min(L, max_L)              # ensure L <= max_cot_len
    dtype_head = getattr(model.stage_head.weight, "dtype", torch.float32)

    image_emb    = torch.randn(B, Ti, Din, device=device)
    question_emb = torch.randn(B, Tq, Din, device=device)
    image_mask    = torch.ones(B, Ti, dtype=torch.bool, device=device)
    question_mask = torch.ones(B, Tq, dtype=torch.bool, device=device)
    cot_input     = torch.randint(0, num_classes, (B, L), device=device)
    A = torch.randn(B, N, N+1, device=device, dtype=torch.float32).to(dtype=dtype_head)

    return dict(
        image_emb=image_emb,
        image_mask=image_mask,
        question_emb=question_emb,
        question_mask=question_mask,
        cot_input=cot_input,
        A=A,
    )
    
device = "cpu"
model = GENModel(device=device).to(device)
model.eval()
batch = make_dummy_batch(model, device=device)

graph = draw_graph(
    model,
    input_data=batch,          # use input_data (not input_size) for multi-input + dtypes
    depth=3,                   # tweak as you like
    expand_nested=True,
    device=device
)
graph.visual_graph.render("model_architecture", format="png")