# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import time
import tiktoken
import torch
import torch.nn as nn


################# Notes from Sebastian's script ##################
# # DSA combines two components:
#   1. A Lightning Indexer that scores all past tokens for each query
#      using a lightweight gated sum of ReLU(q · k) dot products.
#   2. A Token Selector that picks the top-K highest-scoring past tokens
#      and masks out the rest.
#
# This teaching implementation applies a dense mask to standard attention.
# It reproduces the DSA selection logic but does not include a fused sparse
# attention kernel that would reduce attention compute from O(L^2) to O(L*k).
#
# Reference implementation inspired by:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py


class LightningIndexer(nn.Module):
    """Lightweight module that scores every past token for each incoming query.

    For each query token t and each candidate past token s, the score is a
    gated sum of per-head dot products:
        I_{t,s} = sum_j [ (w_{t,j} / sqrt(H_I)) * ReLU((q_{t,j} · k_s) / sqrt(d_I)) ]

    where w_{t,j} is a learned per-head scalar weight derived from the input,
    H_I is the number of index heads, d_I is the index head dimension, and j
    indexes over the index heads.

    Args:
        d_model:       model dimension (same as emb_dim).
        index_n_heads: number of lightweight index heads (H_I in the paper).
        index_head_dim: dimension of each index head.
    """
    
    def __init__(self, d_model: int, index_n_heads: int, index_head_dim: int):
        super().__init__()
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        
        # Project input to indexer query vectors: (d_model -> index_n_heads * index_head_dim)
        self.W_q_index = nn.Linear(d_model, index_n_heads, * index_head_dim, bias=False)
        # Project input to shared key vectors: (d_model -> index_head_dim)
        self.W_k_index = nn.Linear(d_model, index_head_dim, bias=False)
        # Learn a per-head weight scalar: (d_model -> index_n_heads) 
        self.W_weights = nn.Linear(d_model, index_n_heads, bias=False) # based on DS paper
        
        self.scale = index_head_dim ** -0.5
        
    def forward(
        self, 
        x: torch.Tensor,      # (b, T, d_model) current tokens(s)
        x_ctx: torch.Tensor,  # (b, S, d_model) all past + current tokens
        topk: int,
        causal_mask: torch.Tensor | None = None, # (T, S) float mask
    ) -> torch.Tensor:
        """Return top-K token indices shape (b, T, topk)."""
        b, T, _ = x.shape
        _, S, _ = x_ctx.shape
        
        # Indexer queries: (b, T, H_I, head_dim)
        q = self.W_q_index(x).view(b, T, self.index_n_heads, self.index_head_dim)
        # Index keys: (b, S, head_dim)
        k = self.W_k_index(x_ctx) # (b, s, head_dim)
        
        # ReLU(q . k^T) for each head: (b, T, H_I, S)
        # k: (b, S, head_dim) -> (b, 1, S, head_dim) for broadcast
        raw = torch.einsum("bthd,bsd->bths", q, k) * self.scale # (b, T, H_I, S)
        raw = torch.relu(raw)
        
        # Per-head learned gates: (b, T, H_I)
        # Reference implementations use raw learned gates scaled by sqrt(H_I),
        # not a probability distribution over index heads.
        w = self.W_weights(x) # (b, T, H_I)
        w = w * (self.index_n_heads ** -0.5)
        
        # Weighted sum over heads -> index scores (b, T, S)
        index_scores = torch.einsum("bth,bths->bts", w, raw) # (b, T, S)
        
        if causal_mask is not None:
            index_scores = index_scores + causal_mask # broadcast over batch
            
        # Select top-k positions. topk is capped at available context length S.
        k_val = min(topk, S)
        topk_indices = index_scores.topk(k_val, dim=-1).indices # (b, T, k)
        return topk_indices