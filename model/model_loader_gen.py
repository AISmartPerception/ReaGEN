import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")

# # Naive Implementation
# class GENModel_naive(nn.Module):
#     def __init__(
#         self,
#         emb_dim=3584,
#         hidden_dim=512,
#         num_stages=7,
#         num_heads=8,
#         topk_img=256,
#         max_cot_len=7,
#         device=None,
#         use_sdpa=False,         # not used anymore; kept for API compat
#         config = None,
#     ):
#         super().__init__()
#         self.device = device
#         self.num_stages = num_stages
#         self.num_classes = num_stages + 1   # last = EOS/NO_STAGE
#         self.topk_img = topk_img
#         self.attn = config['gen_training']['attn']

#         # 1) Projections (shared for image & question)
#         self.proj = nn.Sequential(
#             nn.Linear(emb_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#         # 2) Tiny question encoder
#         self.q_enc = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads,
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=2
#         )

#         # 3) Stage embeddings
#         self.stage_emb = nn.Embedding(self.num_classes, hidden_dim)

#         # 4) A summary
#         self.A_proj = nn.Sequential(
#             nn.Linear(num_stages * (num_stages + 1), hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#         # Pre-fuse norms (help balance scales)
#         self.pre_fuse_ln_q = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_i = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_a = nn.LayerNorm(hidden_dim)

#         # 5) Joint fusion for (img_feat || q_feat) → hidden_dim
#         self.joint_fuse = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )

#         # 6) Final fuse for (joint_feat || A_feat) → hidden_dim
#         self.fuse = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim)
#         )

#         # 7) Decoder: 2-layer TransformerDecoder (causal)
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
#         self.pos_emb = nn.Embedding(max_cot_len, hidden_dim)

#         # 8) Transition bias from previous token (row-wise prior)
#         self.trans_bias = nn.Sequential(
#             nn.Linear(self.num_classes, 128), nn.ReLU(),
#             nn.Linear(128, self.num_classes)
#         )

#         # 9) Heads
#         self.stage_head  = nn.Linear(hidden_dim, self.num_classes)
#         self.length_head = nn.Linear(hidden_dim, self.num_stages)  # length bucket 0..num_stages-1

#         # ---- cached buffers to avoid reallocations ----
#         self.register_buffer("_cached_pos_idx", torch.arange(max_cot_len), persistent=False)
#         self._cached_causal_L = 0
#         self.register_buffer("_cached_causal", torch.empty(0, 0, dtype=torch.bool), persistent=False)

#     # ---------- utilities ----------

#     def _get_pos(self, L, device):
#         # slice cached arange to avoid new tensor every forward
#         return self._cached_pos_idx[:L].to(device)

#     def _get_causal_mask(self, L, device):
#         # cache the largest we’ve built and slice
#         if L > self._cached_causal_L:
#             causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
#             self._cached_causal = causal.to(device)
#             self._cached_causal_L = L
#         return self._cached_causal[:L, :L]

#     def mean_pool(self, x, mask):
#         """
#         x:    [B, T, D]
#         mask: [B, T] (1=valid, 0=pad)
#         returns [B, D] with safe handling of all-zero mask
#         """
#         m = mask.float()
#         denom = m.sum(1, keepdim=True).clamp_min(1e-6)    # avoid /0
#         pooled = (x * m.unsqueeze(-1)).sum(1) / denom
#         # if a row has all zeros, pooled becomes 0 by construction (good default)
#         return pooled

#     # ---------- forward ----------

#     def forward(self, image_emb, image_mask, question_emb, question_mask, cot_input, A, teacher_next=None):
#         # types (AMP-friendly)
#         A = A.to(dtype=self.stage_head.weight.dtype)
#         cot_input = cot_input.long()

#         B, L = cot_input.size()
#         device = cot_input.device

#         # --- Question encode
#         q = self.proj(question_emb) # [B,Tq,D]
#         q = self.q_enc(q, src_key_padding_mask=~question_mask.to(torch.bool))
#         q_feat = self.mean_pool(q, question_mask.to(torch.bool))           # [B,D]
#         q_feat = self.pre_fuse_ln_q(q_feat)

#         # --- Image proj + simple masked mean pooling
#         img = self.proj(image_emb)                                         # [B,Ti,D]
#         img_feat = self.mean_pool(img, image_mask.to(torch.bool))          # [B,D]
#         img_feat = self.pre_fuse_ln_i(img_feat)

        
        

#         if self.attn:
#             # --- A summary
#             A_vec  = A.flatten(start_dim=1).contiguous()                        # [B, N*(N+1)]
#             A_feat = self.A_proj(A_vec)                                         # [B,D]
#             A_feat = self.pre_fuse_ln_a(A_feat)
#             # --- Joint fusion (img + question) → hidden_dim
#             joint_feat = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1)) # [B,D]

#             # --- Fused memory for decoder cross-attn: (joint || A_feat)
#             memory = self.fuse(torch.cat([joint_feat, A_feat], dim=-1)).unsqueeze(1)  # [B,1,D]
#         else:
#             memory = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1)).unsqueeze(1)

#         # --- Decoder inputs (teacher forcing)
#         pos_idx = self._get_pos(L, device)
#         pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)
#         dec_in = self.stage_emb(cot_input) + pos  # [B,L,D]

#         # --- Causal mask (cached) for self-attn
#         causal = self._get_causal_mask(L, device)

#         dec_out = self.decoder(
#             dec_in,
#             memory,
#             tgt_mask=causal,
#             memory_key_padding_mask=None,   # memory len=1
#         )  # [B,L,D]

#         logits = self.stage_head(dec_out)  # [B,L,num_classes]

#         # --- Transition bias using previous step (gold prev during teacher forcing)
#         prev = cot_input.clamp(min=0, max=self.num_classes - 1)  # [B,L]
#         prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)  # [B,L,C]
#         bias = self.trans_bias(prev_oh)       # [B,L,num_classes]
#         logits = logits + bias

#         length_logits = self.length_head(memory.squeeze(1))  # [B, num_stages]
#         return logits, length_logits



# class GENModel_1(nn.Module):
#     def __init__(
#         self,
#         emb_dim=3584,
#         hidden_dim=512,
#         num_stages=7,
#         num_heads=8,
#         topk_img=256,
#         max_cot_len=7,
#         device=None,
#         use_sdpa=False,         # not used anymore; kept for API compat
#         config = None,
#     ):
#         super().__init__()
#         self.device = device
#         self.num_stages = num_stages
#         self.num_classes = num_stages + 1   # last = EOS/NO_STAGE
#         self.topk_img = topk_img
#         self.attn = config['gen_training']['attn']

#         # 1) Projections (shared for image & question)
#         self.proj = nn.Sequential(
#             nn.Linear(emb_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#         # 2) Tiny question encoder
#         self.q_enc = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads,
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=2
#         )

#         # 3) Stage embeddings
#         self.stage_emb = nn.Embedding(self.num_classes, hidden_dim)

#         # 4) A summary
#         self.A_proj = nn.Sequential(
#             nn.Linear(num_stages * (num_stages + 1), hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#         # Pre-fuse norms (help balance scales)
#         self.pre_fuse_ln_q = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_i = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_a = nn.LayerNorm(hidden_dim)

#         # 5) Joint fusion for (img_feat || q_feat) → hidden_dim
#         self.joint_fuse = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )

#         # # 6) Final fuse for (joint_feat || A_feat) → hidden_dim
#         # self.fuse = nn.Sequential(
#         #     nn.Linear(hidden_dim * 2, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim)
#         # )

#         # 7) Decoder: 2-layer TransformerDecoder (causal)
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
#         self.pos_emb = nn.Embedding(max_cot_len, hidden_dim)

#         # 8) Transition bias from previous token (row-wise prior)
#         self.trans_bias = nn.Sequential(
#             nn.Linear(self.num_classes, 128), nn.ReLU(),
#             nn.Linear(128, self.num_classes)
#         )

#         # 9) Heads
#         self.stage_head  = nn.Linear(hidden_dim, self.num_classes)
#         self.length_head = nn.Linear(hidden_dim, self.num_stages)  # length bucket 0..num_stages-1

#         # ---- cached buffers to avoid reallocations ----
#         self.register_buffer("_cached_pos_idx", torch.arange(max_cot_len), persistent=False)
#         self._cached_causal_L = 0
#         self.register_buffer("_cached_causal", torch.empty(0, 0, dtype=torch.bool), persistent=False)

#     # ---------- utilities ----------

#     def _get_pos(self, L, device):
#         # slice cached arange to avoid new tensor every forward
#         return self._cached_pos_idx[:L].to(device)

#     def _get_causal_mask(self, L, device):
#         # cache the largest we’ve built and slice
#         if L > self._cached_causal_L:
#             causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
#             self._cached_causal = causal.to(device)
#             self._cached_causal_L = L
#         return self._cached_causal[:L, :L]

#     def mean_pool(self, x, mask):
#         """
#         x:    [B, T, D]
#         mask: [B, T] (1=valid, 0=pad)
#         returns [B, D] with safe handling of all-zero mask
#         """
#         m = mask.float()
#         denom = m.sum(1, keepdim=True).clamp_min(1e-6)    # avoid /0
#         pooled = (x * m.unsqueeze(-1)).sum(1) / denom
#         # if a row has all zeros, pooled becomes 0 by construction (good default)
#         return pooled

#     # ---------- forward ----------

#     def forward(self, image_emb, image_mask, question_emb, question_mask, cot_input, A, teacher_next=None):
#         # types (AMP-friendly)
#         A = A.to(dtype=self.stage_head.weight.dtype)
#         cot_input = cot_input.long()

#         B, L = cot_input.size()
#         device = cot_input.device

#         # --- Question encode
#         q = self.proj(question_emb) # [B,Tq,D]
#         q = self.q_enc(q, src_key_padding_mask=~question_mask.to(torch.bool))
#         q_feat = self.mean_pool(q, question_mask.to(torch.bool))           # [B,D]
#         q_feat = self.pre_fuse_ln_q(q_feat)

#         # --- Image proj + simple masked mean pooling
#         img = self.proj(image_emb)                                         # [B,Ti,D]
#         img_feat = self.mean_pool(img, image_mask.to(torch.bool))          # [B,D]
#         img_feat = self.pre_fuse_ln_i(img_feat)
        
#         memory = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1)) # [B,D]

#         # Positional encoding
#         pos_idx = self._get_pos(L, device)
#         pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)

#         # Stage embeddings
#         stage_tok = self.stage_emb(cot_input)  # [B, L, D]

#         # -----------------------------------------
#         # === Incorporate structure matrix A ===
#         # -----------------------------------------

#         # Compute valid (unpadded) reasoning lengths per sample
#         valid_len = (cot_input != self.num_classes-1).sum(dim=1)  # [B]

#         A_guided = torch.zeros_like(stage_tok)  # [B, L, D]
#         A_to_final = torch.zeros(B, L, device=device)

#         for b in range(B):
#             l_real = valid_len[b].item()

#             A_real = A[b, :l_real, :l_real]  # [l_real, l_real]
#             # A_norm = A_real / (A_real.sum(-1, keepdim=True).clamp_min(1e-6))

#             # (1) Structural influence within reasoning chain
#             # Each target slot j receives weighted input from earlier slots i
#             # A_trunc = A_norm[:, :-1]  # exclude final-output column
#             A_ctx = torch.matmul(A_real, stage_tok[b, :l_real, :])  # [l_real, D]
#             A_guided[b, :l_real, :] = A_ctx

#             # (2) Final-output contribution (true last non-zero column)
#             A_to_final[b, :l_real] = A[b, :l_real, -1] # [l_real]

#         # Fuse structural context into stage embeddings
#         stage_tok = stage_tok + 0.5 * A_guided

#         # Optional: compute reasoning summary for final stage guidance
#         A_to_final = A_to_final / (A_to_final.sum(-1, keepdim=True).clamp_min(1e-6))
#         final_ctx = torch.bmm(A_to_final.unsqueeze(1), stage_tok).squeeze(1)  # [B, D]
#         memory = memory + 0.5 * final_ctx

#         # Add positional encoding
#         dec_in = stage_tok + pos  # [B, L, D]

#         # -----------------------------------------
#         # === Decoder ===
#         # -----------------------------------------
#         causal = self._get_causal_mask(L, device)

#         dec_out = self.decoder(
#             dec_in,
#             memory.unsqueeze(1),
#             tgt_mask=causal,
#             memory_key_padding_mask=None,
#         )  # [B, L, D]

#         logits = self.stage_head(dec_out)  # [B, L, num_classes]

#         # --- Transition bias ---
#         prev = cot_input.clamp(min=0, max=self.num_classes - 1)  # [B,L]
#         prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)  # [B,L,C]
#         bias = self.trans_bias(prev_oh)
#         logits = logits + bias
                
        


#         # if self.attn:
#         #     # --- A summary
#         #     A_vec  = A.flatten(start_dim=1).contiguous()                        # [B, N*(N+1)]
#         #     A_feat = self.A_proj(A_vec)                                         # [B,D]
#         #     A_feat = self.pre_fuse_ln_a(A_feat)
#         #     # --- Joint fusion (img + question) → hidden_dim
#         #     joint_feat = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1)) # [B,D]

#         #     # --- Fused memory for decoder cross-attn: (joint || A_feat)
#         #     memory = self.fuse(torch.cat([joint_feat, A_feat], dim=-1)).unsqueeze(1)  # [B,1,D]
#         # else:
#         #     memory = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1)).unsqueeze(1)

#         # # --- Decoder inputs (teacher forcing)
#         # pos_idx = self._get_pos(L, device)
#         # pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)
#         # dec_in = self.stage_emb(cot_input) + pos  # [B,L,D]

#         # # --- Causal mask (cached) for self-attn
#         # causal = self._get_causal_mask(L, device)

#         # dec_out = self.decoder(
#         #     dec_in,
#         #     memory,
#         #     tgt_mask=causal,
#         #     memory_key_padding_mask=None,   # memory len=1
#         # )  # [B,L,D]

#         # logits = self.stage_head(dec_out)  # [B,L,num_classes]

#         # # --- Transition bias using previous step (gold prev during teacher forcing)
#         # prev = cot_input.clamp(min=0, max=self.num_classes - 1)  # [B,L]
#         # prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)  # [B,L,C]
#         # bias = self.trans_bias(prev_oh)       # [B,L,num_classes]
#         # logits = logits + bias

#         length_logits = self.length_head(memory.squeeze(1))  # [B, num_stages]
#         return logits, length_logits



# class GENModel_2(nn.Module):
#     def __init__(
#         self,
#         emb_dim=3584,
#         hidden_dim=512,
#         num_stages=7,
#         num_heads=8,
#         topk_img=256,
#         max_cot_len=7,
#         device=None,
#         use_sdpa=False,         # not used anymore; kept for API compat
#         config = None,
#     ):
#         super().__init__()
#         self.device = device
#         self.num_stages = num_stages
#         self.num_classes = num_stages + 1   # last = EOS/NO_STAGE
#         self.topk_img = topk_img
#         self.attn = config['gen_training']['attn']

#         # 1) Projections (shared for image & question)
#         self.proj = nn.Sequential(
#             nn.Linear(emb_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )

#         # 2) Tiny question encoder
#         self.q_enc = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(
#                 d_model=hidden_dim,
#                 nhead=num_heads,
#                 batch_first=True,
#                 norm_first=True
#             ),
#             num_layers=2
#         )

#         # 3) Stage embeddings
#         self.stage_emb = nn.Embedding(self.num_classes, hidden_dim)

#         # 4) A summary
#         self.A_reason_proj = nn.Sequential(
#             nn.Linear(num_stages * (num_stages + 1), hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU()
#         )
        
#         # self.cot_proj = nn.Sequential(
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.GELU()
#         # )
#         # self.A_reason_proj = nn.Sequential(
#         #     nn.Linear(num_stages * (num_stages + 1), hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.GELU()
#         # )

#         # Pre-fuse norms (help balance scales)
#         self.pre_fuse_ln_q = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_i = nn.LayerNorm(hidden_dim)
#         self.pre_fuse_ln_a = nn.LayerNorm(hidden_dim)

#         # 5) Joint fusion for (img_feat || q_feat) → hidden_dim
#         self.joint_fuse = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.GELU(),
#             nn.Dropout(0.1)
#         )

#         # # 6) Final fuse for (joint_feat || A_feat) → hidden_dim
#         # self.fuse = nn.Sequential(
#         #     nn.Linear(hidden_dim * 2, hidden_dim),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.1),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         #     nn.LayerNorm(hidden_dim)
#         # )

#         # 7) Decoder: 2-layer TransformerDecoder (causal)
#         dec_layer = nn.TransformerDecoderLayer(
#             d_model=hidden_dim,
#             nhead=num_heads,
#             batch_first=True,
#             norm_first=True
#         )
#         self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
#         self.pos_emb = nn.Embedding(max_cot_len, hidden_dim)

#         # 8) Transition bias from previous token (row-wise prior)
#         self.trans_bias = nn.Sequential(
#             nn.Linear(self.num_classes, 128), nn.ReLU(),
#             nn.Linear(128, self.num_classes)
#         )

#         # 9) Heads
#         self.stage_head  = nn.Linear(hidden_dim, self.num_classes)
#         self.length_head = nn.Linear(hidden_dim, self.num_stages)  # length bucket 0..num_stages-1

#         # ---- cached buffers to avoid reallocations ----
#         self.register_buffer("_cached_pos_idx", torch.arange(max_cot_len), persistent=False)
#         self._cached_causal_L = 0
#         self.register_buffer("_cached_causal", torch.empty(0, 0, dtype=torch.bool), persistent=False)

#     # ---------- utilities ----------

#     def _get_pos(self, L, device):
#         # slice cached arange to avoid new tensor every forward
#         return self._cached_pos_idx[:L].to(device)

#     def _get_causal_mask(self, L, device):
#         # cache the largest we’ve built and slice
#         if L > self._cached_causal_L:
#             causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
#             self._cached_causal = causal.to(device)
#             self._cached_causal_L = L
#         return self._cached_causal[:L, :L]

#     def mean_pool(self, x, mask):
#         """
#         x:    [B, T, D]
#         mask: [B, T] (1=valid, 0=pad)
#         returns [B, D] with safe handling of all-zero mask
#         """
#         m = mask.float()
#         denom = m.sum(1, keepdim=True).clamp_min(1e-6)    # avoid /0
#         pooled = (x * m.unsqueeze(-1)).sum(1) / denom
#         # if a row has all zeros, pooled becomes 0 by construction (good default)
#         return pooled


#     def forward(self, image_emb, image_mask, question_emb, question_mask, cot_input, A, teacher_next=None):
#         """
#         image_emb:     [B, Ti, emb_dim]
#         image_mask:    [B, Ti]
#         question_emb:  [B, Tq, emb_dim]
#         question_mask: [B, Tq]
#         cot_input:     [B, L]          (stage IDs, including EOS)
#         A:             [B, N, N+1] or [B, N, N] structure matrices
#         """
#         # ---------- setup ----------
#         A = A.to(dtype=self.stage_head.weight.dtype)
#         cot_input = cot_input.long()
#         B, L = cot_input.size()
#         device = cot_input.device

#         # ---------- 1. Encode question ----------
#         q = self.proj(question_emb)  # [B, Tq, D]
#         q = self.q_enc(q, src_key_padding_mask=~question_mask.bool())
#         q_feat = self.mean_pool(q, question_mask.bool())  # [B, D]
#         q_feat = self.pre_fuse_ln_q(q_feat)

#         # ---------- 2. Encode image ----------
#         img = self.proj(image_emb)  # [B, Ti, D]
#         img_feat = self.mean_pool(img, image_mask.bool())  # [B, D]
#         img_feat = self.pre_fuse_ln_i(img_feat)

#         # ---------- 3. Fuse multimodal features ----------
#         joint_feat = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1))  # [B, D]
#         joint_feat = F.layer_norm(joint_feat, (joint_feat.size(-1),))

#         # ---------- 4. Stage embeddings ----------
#         stage_tok = self.stage_emb(cot_input)  # [B, L, D]

#         # ---------- 5. Compute structure-guided embeddings from A ----------
#         valid_len = (cot_input != self.num_classes - 1).sum(dim=1)
#         A_guided = torch.zeros_like(stage_tok)
#         A_to_final = torch.zeros(B, L, device=device)

#         for b in range(B):
#             l_real = valid_len[b].item()
#             A_real = A[b, :l_real, :l_real]  # [l_real, l_real]
#             A_ctx = torch.matmul(A_real, stage_tok[b, :l_real, :])  # [l_real, D]
#             A_guided[b, :l_real, :] = A_ctx
#             A_to_final[b, :l_real] = A[b, :l_real, -1] if A.shape[-1] > l_real else 0.0
            

#         # Fuse A-guided reasoning context into stage tokens
#         stage_tok = stage_tok + 0.5 * A_guided

#         # ---------- 6. Build A_reason (flattened reasoning vector) ----------
#         A_vec = A.flatten(start_dim=1)  # [B, N*(N+1)]
#         A_reason = self.A_reason_proj(A_vec)  # [B, D]
#         A_reason = self.pre_fuse_ln_a(A_reason)

#         # ---------- 7. Construct 2-token memory ----------
#         # Token 0: joint multimodal embedding (perception)
#         # Token 1: reasoning embedding (structure)
#         memory = torch.stack([joint_feat, A_reason], dim=1)  # [B, 2, D]
#         memory = F.layer_norm(memory, (memory.size(-1),))

#         # ---------- 8. Add positional encoding for decoder ----------
#         pos_idx = self._get_pos(L, device)
#         pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)
#         dec_in = stage_tok + pos  # [B, L, D]

#         # ---------- 9. Decode ----------
#         causal = self._get_causal_mask(L, device)
#         dec_out = self.decoder(
#             dec_in,
#             memory,  # [B, 2, D] cross-attended by decoder
#             tgt_mask=causal,
#             memory_key_padding_mask=None
#         )  # [B, L, D]

#         # ---------- 10. Predict next-stage logits ----------
#         logits = self.stage_head(dec_out)  # [B, L, num_classes]

#         # Transition bias (token-wise)
#         prev = cot_input.clamp(min=0, max=self.num_classes - 1)
#         prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)
#         bias = self.trans_bias(prev_oh)
#         logits = logits + bias

#         # ---------- 11. Length prediction ----------
#         length_logits = self.length_head(memory.mean(dim=1))  # [B, num_stages]

#         return logits, length_logits
    
    


class GENModel_old(nn.Module):
    def __init__(
        self,
        emb_dim=3584,
        hidden_dim=512,
        num_stages=14,
        num_heads=8,
        max_cot_len=5,
        device=None,
        config = None,
    ):
        super().__init__()
        self.device = device
        self.num_stages = num_stages
        self.num_classes = num_stages + 1   # last = EOS/NO_STAGE
        self.attn = config['gen_training']['attn']

        # 1) Projections (shared for image & question)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # 2) Tiny question encoder
        self.q_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )

        # 3) Stage embeddings
        self.stage_emb = nn.Embedding(self.num_classes, hidden_dim)

        # 4) A summary
        self.A_reason_proj = nn.Sequential(
            nn.Linear(num_stages * (num_stages + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # self.cot_proj = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU()
        # )
        # self.A_reason_proj = nn.Sequential(
        #     nn.Linear(num_stages * (num_stages + 1), hidden_dim),
        #     nn.LayerNorm(hidden_dim),
        #     nn.GELU()
        # )

        # Pre-fuse norms (help balance scales)
        self.pre_fuse_ln_q = nn.LayerNorm(hidden_dim)
        self.pre_fuse_ln_i = nn.LayerNorm(hidden_dim)
        self.pre_fuse_ln_a = nn.LayerNorm(hidden_dim)

        # 5) Joint fusion for (img_feat || q_feat) → hidden_dim
        self.joint_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # # 6) Final fuse for (joint_feat || A_feat) → hidden_dim
        # self.fuse = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.LayerNorm(hidden_dim)
        # )

        # 7) Decoder: 2-layer TransformerDecoder (causal)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.pos_emb = nn.Embedding(max_cot_len, hidden_dim)

        # 8) Transition bias from previous token (row-wise prior)
        self.trans_bias = nn.Sequential(
            nn.Linear(self.num_classes, 128), nn.ReLU(),
            nn.Linear(128, self.num_classes)
        )

        # 9) Heads
        self.stage_head  = nn.Linear(hidden_dim, self.num_classes)
        self.length_head = nn.Linear(hidden_dim, self.num_stages)  # length bucket 0..num_stages-1

        # ---- cached buffers to avoid reallocations ----
        self.register_buffer("_cached_pos_idx", torch.arange(max_cot_len), persistent=False)
        self._cached_causal_L = 0
        self.register_buffer("_cached_causal", torch.empty(0, 0, dtype=torch.bool), persistent=False)

    # ---------- utilities ----------

    def _get_pos(self, L, device):
        # slice cached arange to avoid new tensor every forward
        return self._cached_pos_idx[:L].to(device)

    def _get_causal_mask(self, L, device):
        # cache the largest we’ve built and slice
        if L > self._cached_causal_L:
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            self._cached_causal = causal.to(device)
            self._cached_causal_L = L
        return self._cached_causal[:L, :L]

    def mean_pool(self, x, mask):
        """
        x:    [B, T, D]
        mask: [B, T] (1=valid, 0=pad)
        returns [B, D] with safe handling of all-zero mask
        """
        m = mask.float()
        denom = m.sum(1, keepdim=True).clamp_min(1e-6)    # avoid /0
        pooled = (x * m.unsqueeze(-1)).sum(1) / denom
        # if a row has all zeros, pooled becomes 0 by construction (good default)
        return pooled

    
    def forward(self, image_emb, image_mask, question_emb, question_mask, cot_input, A, teacher_next=None):
        """
        image_emb:     [B, Ti, emb_dim]
        image_mask:    [B, Ti]
        question_emb:  [B, Tq, emb_dim]
        question_mask: [B, Tq]
        cot_input:     [B, L]
        A:             [B, N, N+1] or [B, N, N]
        """
        # ---------- setup ----------
        A = A.to(dtype=self.stage_head.weight.dtype)
        cot_input = cot_input.long()
        B, L = cot_input.size()
        device = cot_input.device

        # ---------- 1. Encode question ----------
        q = self.proj(question_emb)
        q = self.q_enc(q, src_key_padding_mask=~question_mask.bool())
        q_feat = self.mean_pool(q, question_mask.bool())  # [B, D]
        q_feat = self.pre_fuse_ln_q(q_feat)

        # ---------- 2. Encode image ----------
        img = self.proj(image_emb)
        img_feat = self.mean_pool(img, image_mask.bool())  # [B, D]
        img_feat = self.pre_fuse_ln_i(img_feat)

        # ---------- 3. Fuse multimodal context ----------
        joint_feat = self.joint_fuse(torch.cat([img_feat, q_feat], dim=-1))  # [B, D]
        joint_feat = F.layer_norm(joint_feat, (joint_feat.size(-1),))

        # ---------- 4. Stage embeddings ----------
        stage_tok = self.stage_emb(cot_input)  # [B, L, D]

        # ---------- 5. Structure-guided adjustment (A) ----------
        valid_len = (cot_input != self.num_classes - 1).sum(dim=1)
        A_guided = torch.zeros_like(stage_tok)

        for b in range(B):
            l_real = valid_len[b].item()
            if l_real > 0:
                A_real = A[b, :l_real, :l_real]
                A_ctx = torch.matmul(A_real, stage_tok[b, :l_real, :])  # [l_real, D]
                A_guided[b, :l_real, :] = A_ctx

        stage_tok = stage_tok + 0.5 * A_guided  # reasoning-aware tokens

        # ---------- 6. Global reasoning representation ----------
        A_vec = A.flatten(start_dim=1)
        A_reason = self.A_reason_proj(A_vec)  # [B, D]
        A_reason = self.pre_fuse_ln_a(A_reason)

        # ---------- 7. CoT summary representation ----------
        cot_mask = (cot_input != self.num_classes - 1).float()  # [B, L]
        cot_mean = self.mean_pool(stage_tok, cot_mask.bool())  # [B, D]
        cot_mean = F.layer_norm(cot_mean, (cot_mean.size(-1),))

        # ---------- 8. Combine all three sources ----------
        memory = torch.stack([joint_feat, A_reason, cot_mean], dim=1)  # [B, 3, D]
        memory = F.layer_norm(memory, (memory.size(-1),))

        # ---------- 9. Decoder input ----------
        pos_idx = self._get_pos(L, device)
        pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)
        dec_in = stage_tok + pos  # [B, L, D]

        # ---------- 10. Decoder ----------
        causal = self._get_causal_mask(L, device)
        dec_out = self.decoder(
            dec_in,
            memory,  # [B, 3, D]
            tgt_mask=causal,
            memory_key_padding_mask=None
        )  # [B, L, D]

        # ---------- 11. Predictions ----------
        logits = self.stage_head(dec_out)  # [B, L, num_classes]

        prev = cot_input.clamp(min=0, max=self.num_classes - 1)
        prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)
        bias = self.trans_bias(prev_oh)
        logits = logits + bias

        length_logits = self.length_head(memory.mean(dim=1))  # [B, num_stages]

        return logits, length_logits


class TokenPool(nn.Module):
    """Learnable pooling to reduce tokens into fixed count (e.g. 24)."""
    def __init__(self, n_tokens, dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_tokens, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, x, mask=None):
        B = x.size(0)
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.attn(q, x, x, key_padding_mask=(~mask.bool()) if mask is not None else None)
        return self.ln(pooled)


class CrossModalEncoder(nn.Module):
    """Cross-attention: question attends to image tokens."""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)

    def forward(self, q, img, img_mask=None):
        attn_out, _ = self.cross_attn(q, img, img, key_padding_mask=(~img_mask.bool()) if img_mask is not None else None)
        return self.ln(q + attn_out)


class GENModel_1(nn.Module):
    def __init__(
        self,
        emb_dim=3584,
        hidden_dim=512,
        num_stages=14,
        num_heads=8,
        max_cot_len=14,
        mean_tok_len=2,
        # mean_tok_len=8,
        device=None,
        config = None,
    ):
        super().__init__()
        self.device = device
        self.max_cot_len = max_cot_len
        self.num_stages = num_stages
        self.num_classes = num_stages + 1   # last = EOS/NO_STAGE
        self.attn = config['gen_training']['attn']
        self.i_q = config['gen_training']['i_q']

        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # self.q_enc = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=hidden_dim,
        #         nhead=num_heads,
        #         batch_first=True,
        #         norm_first=True
        #     ),
        #     num_layers=2
        # )
        self.q_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                batch_first=True,
                norm_first=True
            ),
            num_layers=2
        )

        self.cross_modal_enc = CrossModalEncoder(hidden_dim, num_heads)

        self.token_pool = TokenPool(mean_tok_len, hidden_dim, num_heads)

        self.stage_emb = nn.Embedding(self.num_classes, hidden_dim)

        self.A_reason_proj = nn.Sequential(
            nn.Linear(num_stages * (num_stages + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.dec_pos_emb = nn.Embedding(max_cot_len, hidden_dim)
        self.mem_pos_emb = nn.Embedding(mean_tok_len + 2, hidden_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True
        )
        # self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=2)


        self.stage_head = nn.Linear(hidden_dim, self.num_classes)
        self.length_head = nn.Linear(hidden_dim, self.max_cot_len)
        self.trans_bias = nn.Sequential(
            nn.Linear(self.num_classes, 128), nn.ReLU(), nn.Linear(128, self.num_classes)
        )

        # ---- cached buffers to avoid reallocations ----
        self.register_buffer("_cached_pos_idx", torch.arange(max_cot_len), persistent=False)
        self._cached_causal_L = 0
        self.register_buffer("_cached_causal", torch.empty(0, 0, dtype=torch.bool), persistent=False)


    def _get_causal_mask(self, L, device):
        if L > self._cached_causal_L:
            self._cached_causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1).to(device)
            self._cached_causal_L = L
        return self._cached_causal[:L, :L]

    def mean_pool(self, x, mask):
        m = mask.float()
        denom = m.sum(1, keepdim=True).clamp_min(1e-6)
        return (x * m.unsqueeze(-1)).sum(1) / denom

    
    def forward(self, image_emb, image_mask, question_emb, question_mask, cot_input, A, teacher_next=None):
        """
        image_emb:     [B, Ti, emb_dim]
        image_mask:    [B, Ti]
        question_emb:  [B, Tq, emb_dim]
        question_mask: [B, Tq]
        cot_input:     [B, L]
        A:             [B, N, N+1] or [B, N, N]
        """
        
        # ---------- setup ----------
        A = A.to(dtype=self.stage_head.weight.dtype)
        cot_input = cot_input.long()
        B, L = cot_input.size()
        device = cot_input.device

        # 1. Encode image 
        img = self.proj(image_emb)

        # 2. Encode question + cross-modal conditioning 
        q = self.proj(question_emb)
        q = self.q_enc(q, src_key_padding_mask=~question_mask.bool())
        q = self.cross_modal_enc(q, img, img_mask=image_mask)

        # 3. Fuse multimodal context
        joint_seq = torch.cat([img, q], dim=1)
        joint_mask = torch.cat([image_mask, question_mask], dim=1)

        # 4. Token pooling: multi-token mem about Image and question
        mem_seq = self.token_pool(joint_seq, joint_mask)  # [B, mem_tokens, D]
        # mem_seq = self.mean_pool(joint_seq, joint_mask)

        # 5. Positional embeddings to memory
        # mem_pos = self.mem_pos_emb(torch.arange(mem_seq.size(1), device=device))
        # mem_seq = mem_seq + mem_pos.unsqueeze(0)
        


        # 6. Attn info: Structure-guided adjustment (A)
        A_vec = A.flatten(start_dim=1)
        A_reason = self.A_reason_proj(A_vec).unsqueeze(1)  # [B,1,D]
        # cot_mask = (cot_input != self.num_classes - 1).float()
        # cot_mean = self.mean_pool(self.stage_emb(cot_input), cot_mask.bool()).unsqueeze(1)
        
        # 7. Reasoning-aware bias based on A
        stage_tok = self.stage_emb(cot_input)
        valid_len = (cot_input != self.num_classes - 1).sum(dim=1)
        A_guided = torch.zeros_like(stage_tok)

        for b in range(B):
            l_real = valid_len[b].item()
            if l_real > 0:
                A_real = A[b, :l_real, :l_real]
                A_ctx = torch.matmul(A_real, stage_tok[b, :l_real, :])  # [l_real, D]
                A_guided[b, :l_real, :] = A_ctx

        if self.attn:
            stage_tok = stage_tok + A_guided  # reasoning-aware tokens
        else:
            stage_tok = stage_tok


        # 7. Final memory = [joint memory + A_reason + cot_mean]
        if self.i_q and self.attn:
            memory = torch.cat([mem_seq, A_reason, A_guided], dim=1)  # [B, M+2, D]
        elif self.i_q and not self.attn:
            memory = torch.cat([mem_seq], dim=1)  # [B, M+1, D]
        elif not self.i_q and self.attn:
            memory = torch.cat([A_reason, A_guided], dim=1)  # [B, M+1, D]
      
        # memory = torch.cat([mem_seq, A_reason, A_guided], dim=1)  # [B, M+2, D]

        # 8. Decoder input with positional embeddings
        pos_idx = self._cached_pos_idx[:self.max_cot_len].to(device)
        dec_in = stage_tok[:, :self.max_cot_len, :] + self.dec_pos_emb(pos_idx).unsqueeze(0)
        causal = self._get_causal_mask(self.max_cot_len, device)

        # Decode
        dec_out = self.decoder(dec_in, memory, tgt_mask=causal)

        # Predict per CoT stage
        logits = self.stage_head(dec_out)
        # prev_oh = F.one_hot(cot_input.clamp(0, self.num_classes - 1), num_classes=self.num_classes).float()
        # logits += self.trans_bias(prev_oh)
        length_logits = self.length_head(memory.mean(dim=1))

        return logits, length_logits
        
        # TODO: Add bias based on attn info


        # ---------- 5. Structure-guided adjustment (A) ----------
        valid_len = (cot_input != self.num_classes - 1).sum(dim=1)
        A_guided = torch.zeros_like(stage_tok)

        for b in range(B):
            l_real = valid_len[b].item()
            if l_real > 0:
                A_real = A[b, :l_real, :l_real]
                A_ctx = torch.matmul(A_real, stage_tok[b, :l_real, :])  # [l_real, D]
                A_guided[b, :l_real, :] = A_ctx

        stage_tok = stage_tok + 0.5 * A_guided  # reasoning-aware tokens

        # ---------- 6. Global reasoning representation ----------
        A_vec = A.flatten(start_dim=1)
        A_reason = self.A_reason_proj(A_vec)  # [B, D]
        A_reason = self.pre_fuse_ln_a(A_reason)

        # ---------- 7. CoT summary representation ----------
        cot_mask = (cot_input != self.num_classes - 1).float()  # [B, L]
        cot_mean = self.mean_pool(stage_tok, cot_mask.bool())  # [B, D]
        cot_mean = F.layer_norm(cot_mean, (cot_mean.size(-1),))

        # ---------- 8. Combine all three sources ----------
        memory = torch.stack([joint_feat, A_reason, cot_mean], dim=1)  # [B, 3, D]
        memory = F.layer_norm(memory, (memory.size(-1),))

        # ---------- 9. Decoder input ----------
        pos_idx = self._get_pos(L, device)
        pos = self.pos_emb(pos_idx).unsqueeze(0).expand(B, L, -1)
        dec_in = stage_tok + pos  # [B, L, D]

        # ---------- 10. Decoder ----------
        causal = self._get_causal_mask(L, device)
        dec_out = self.decoder(
            dec_in,
            memory,  # [B, 3, D]
            tgt_mask=causal,
            memory_key_padding_mask=None
        )  # [B, L, D]

        # ---------- 11. Predictions ----------
        logits = self.stage_head(dec_out)  # [B, L, num_classes]

        prev = cot_input.clamp(min=0, max=self.num_classes - 1)
        prev_oh = F.one_hot(prev, num_classes=self.num_classes).to(logits.dtype)
        bias = self.trans_bias(prev_oh)
        logits = logits + bias

        length_logits = self.length_head(memory.mean(dim=1))  # [B, num_stages]

        return logits, length_logits