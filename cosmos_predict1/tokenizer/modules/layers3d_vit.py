# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The model definition for 3D layers

Adapted from: https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/magvit2_pytorch/magvit2_pytorch.py#L889

[MIT License Copyright (c) 2023 Phil Wang]
https://github.com/lucidrains/magvit2-pytorch/blob/
9f49074179c912736e617d61b32be367eb5f993a/LICENSE
"""
import math
from typing import Tuple, Union, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict1.tokenizer.modules.patching import Patcher3DArbitrary, UnPatcher3DArbitrary


########################################################
# Causal ViT-based Tokenizer Layers
########################################################

# ------------------------------------------------------
# RMSNorm
# ------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x) * self.weight

# ------------------------------------------------------
# 1D RoPE
# ------------------------------------------------------

def precompute_freqs_cis(dim: int, max_position_embeddings: int, theta: float = 10000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Create the precomputed cos/sin for rotary embeddings (dim must be even).
    Returns a [max_position_embeddings, dim/2, 2] tensor with cos/sin.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype) / dim))
    t = torch.arange(max_position_embeddings, dtype=dtype)
    freqs = torch.einsum('i,j->ij', t, freqs)  # [max_position_embeddings, dim/2]
    sin, cos = freqs.sin(), freqs.cos()
    # Combine cos/sin into last dimension
    return torch.stack([cos, sin], dim=-1)  # [max_pos, dim/2, 2]


def precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float) -> torch.Tensor:
    """
    Copied from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py

    freqs_cis: 2D complex tensor of shape (height, width, dim // 2) to be indexed by
        (height, width) position tuples
    """
    # (dim / 2) frequency bases
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    h = torch.arange(height, device=freqs.device)
    w = torch.arange(width, device=freqs.device)

    freqs_h = torch.outer(h, freqs[::2]).float()
    freqs_w = torch.outer(w, freqs[1::2]).float()
    freqs_2d = torch.cat(
        [
            freqs_h[:, None, :].repeat(1, width, 1),
            freqs_w[None, :, :].repeat(height, 1, 1),
        ],
        dim=-1,
    )

    # (height, width, dim // 2)
    freqs_2d = torch.polar(torch.ones_like(freqs_2d), freqs_2d) # (height, width, dim // 2)
    cos, sin = freqs_2d.real, freqs_2d.imag   
    # Combine cos/sin into last dimension
    return torch.stack([cos, sin], dim=-1)  # [height, width, dim //2, 2]


def apply_rotary_emb_2d(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Copied from https://github.com/mistralai/mistral-inference/blob/main/src/mistral_inference/rope.py

    q, k: [B, h, w, n_heads, head_dim], head_dim must be even
    freqs_cis: [max_seq_len, head_dim/2, 2]
    """
    bsz, h, w, n_heads, head_dim = q.shape
    # slice out the needed positions
    assert freqs_cis.shape[0] >= h, f"freqs_cis.shape[0] = {freqs_cis.shape[0]} must be >= h = {h}"
    assert freqs_cis.shape[1] >= w, f"freqs_cis.shape[1] = {freqs_cis.shape[1]} must be >= w = {w}"
    freqs_cis = freqs_cis[:h, :w]  # shape [h, w, head_dim//2, 2]

    # Expand to shape [1, h, w, 1, head_dim//2, 2]
    freqs_cis = freqs_cis.unsqueeze(2).unsqueeze(0)

    # reshape Q/K to complex
    q_reshaped = q.view(bsz, h, w, n_heads, head_dim // 2, 2)
    k_reshaped = k.view(bsz, h, w, n_heads, head_dim // 2, 2)

    # This convert is to ensure view_as_complex is supported
    q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))

    # Properly expand freqs_cis to match the batch and head dimensions
    # [1, h, w, 1, head_dim//2, 2] -> [bsz, h, w, n_heads, head_dim//2, 2]
    freqs_cis = freqs_cis.expand(bsz, h, w, n_heads, head_dim // 2, 2)
    freqs_complex = torch.view_as_complex(freqs_cis.to(torch.float32))

    q_out = torch.view_as_real(q_complex * freqs_complex).to(q.dtype)
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype)

    q_out = q_out.view(bsz, h, w, n_heads, head_dim)
    k_out = k_out.view(bsz, h, w, n_heads, head_dim)
    return q_out, k_out


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, S, n_heads, head_dim], head_dim must be even
    freqs_cis: [max_seq_len, head_dim/2, 2]
    """
    bsz, seq_len, n_heads, head_dim = q.shape
    # slice out the needed positions
    assert freqs_cis.shape[0] >= seq_len, f"freqs_cis.shape[0] = {freqs_cis.shape[0]} must be >= seq_len = {seq_len}"
    freqs_cis = freqs_cis[:seq_len]  # shape [seq_len, head_dim//2, 2]
    # Expand to shape [1, seq_len, 1, head_dim//2, 2]
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)

    # reshape Q/K to complex
    q_reshaped = q.view(bsz, seq_len, n_heads, head_dim // 2, 2)
    k_reshaped = k.view(bsz, seq_len, n_heads, head_dim // 2, 2)

    # This convert is to ensure view_as_complex is supported
    q_complex = torch.view_as_complex(q_reshaped.to(torch.float32))
    k_complex = torch.view_as_complex(k_reshaped.to(torch.float32))

    # Properly expand freqs_cis to match the batch and head dimensions
    # [1, seq_len, 1, head_dim//2, 2] -> [bsz, seq_len, n_heads, head_dim//2, 2]
    freqs_cis = freqs_cis.expand(bsz, seq_len, n_heads, head_dim // 2, 2)
    freqs_complex = torch.view_as_complex(freqs_cis.to(torch.float32))

    q_out = torch.view_as_real(q_complex * freqs_complex).to(q.dtype)
    k_out = torch.view_as_real(k_complex * freqs_complex).to(k.dtype)

    q_out = q_out.view(bsz, seq_len, n_heads, head_dim)
    k_out = k_out.view(bsz, seq_len, n_heads, head_dim)
    return q_out, k_out

# ------------------------------------------------------
# Attention with RoPE
# ------------------------------------------------------

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, config: Dict, use_1d_rotary: bool = False):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.head_dim = embed_dim // num_heads

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.use_1d_rotary = use_1d_rotary

        if self.use_1d_rotary or self.config.concat_decode_2d: # If not use_1d_rotary but concat, freqs_cis is required as well
            # Build a buffer for rotary
            self.register_buffer(
                'freqs_cis',
                precompute_freqs_cis(self.head_dim, config.max_sequence_length_1d, config.theta),
                persistent=False
            )
        
        if not self.use_1d_rotary:
            # Build a buffer for 2D rotary
            self.register_buffer(
                'freqs_cis_2d',
                precompute_freqs_cis_2d(self.head_dim, config.max_sequence_length, config.max_sequence_length, config.theta),
                persistent=False
            )

        # Initialize parameters as in the JAX code
        nn.init.normal_(self.mha.in_proj_weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.mha.in_proj_bias)
        nn.init.normal_(self.mha.out_proj.weight, mean=0.0, std=config.initializer_range)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        encoding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        hidden_states: [B, S, E]
        encoding_mask: [B, S], 1 indicates the token is masked, 0 indicates the token is visible, used for adaptive tokenization
        attention_mask: [S, S] where 1=masked, 0=keep, or None
        position_ids: [B, S], optional
        cache: optional dict for incremental decode
        """
        B, S, E = hidden_states.shape

        # MHA does Q/K/V creation inside. We'll do it manually so we can do rotary on Q/K
        # Extract the projection parameters
        in_proj_weight = self.mha.in_proj_weight
        in_proj_bias = self.mha.in_proj_bias

        # Project Q, K, V in one go
        qkv = F.linear(hidden_states, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, S, E]

        # Now reshape Q/K for rotary
        n_heads = self.mha.num_heads
        head_dim = E // n_heads
        T, H, W = position_ids

        if self.use_1d_rotary:
            q = q.view(B * T, -1, n_heads, head_dim)
            k = k.view(B * T, -1, n_heads, head_dim)
            # For advanced usage, you'd gather exact freq for each position, but here we
            # assume sequences are uniform (like Llama).
            q, k = apply_rotary_emb(q, k, self.freqs_cis)
            q = q.view(B, -1, n_heads, head_dim)
            k = k.view(B, -1, n_heads, head_dim)
        
        elif self.config.concat_decode_2d and S > T*H*W: # use_2d_rotary but x is concated as [x_1d, x_2d] in sequence
            q = q.view(B, S, n_heads, head_dim)
            k = k.view(B, S, n_heads, head_dim)
            
            # Apply 1d rotary embedding on 1d tokens
            q_1d = q[:, :-T*H*W, :, :].view(B * T, -1, n_heads, head_dim)
            k_1d = k[:, :-T*H*W, :, :].view(B * T, -1, n_heads, head_dim)
            q_1d, k_1d = apply_rotary_emb(q_1d, k_1d, self.freqs_cis)
            q_1d = q_1d.view(B, -1, n_heads, head_dim)
            k_1d = k_1d.view(B, -1, n_heads, head_dim)
            
            # Apply 2d rotary embedding on 2d tokens
            q_2d = q[:, -T*H*W:, :, :].view(B, T, H, W, n_heads, head_dim)
            k_2d = k[:, -T*H*W:, :, :].view(B, T, H, W, n_heads, head_dim)
            # merge T to batch dimension to apply rotary embedding
            q_2d = rearrange(q_2d, "b t h w n d -> (b t) h w n d")
            k_2d = rearrange(k_2d, "b t h w n d -> (b t) h w n d")
            # apply rotary embedding
            q_2d, k_2d = apply_rotary_emb_2d(q_2d, k_2d, self.freqs_cis_2d)
            # reshape back to [B, T, H, W, n_heads, head_dim]
            q_2d = rearrange(q_2d, "(b t) h w n d -> b (t h w) n d", b=B, t=T)
            k_2d = rearrange(k_2d, "(b t) h w n d -> b (t h w) n d", b=B, t=T)
            
            # Merge 2d tokens back with 1d tokens
            q = torch.cat([q_1d, q_2d], dim=1)
            k = torch.cat([k_1d, k_2d], dim=1)
            
        else:
            # apply 3D rotary embedding
            q = q.view(B, T, H, W, n_heads, head_dim)
            k = k.view(B, T, H, W, n_heads, head_dim)

            # merge T to batch dimension to apply rotary embedding
            q = rearrange(q, "b t h w n d -> (b t) h w n d")
            k = rearrange(k, "b t h w n d -> (b t) h w n d")
            # apply rotary embedding
            q, k = apply_rotary_emb_2d(q, k, self.freqs_cis_2d)
            # reshape back to [B, T, H, W, n_heads, head_dim]
            q = rearrange(q, "(b t) h w n d -> b t h w n d", b=B, t=T)
            k = rearrange(k, "(b t) h w n d -> b t h w n d", b=B, t=T)


        # Reshape back to [B, S, E]
        q = q.view(B, S, E)
        k = k.view(B, S, E)

        # Optional caching for incremental decode
        # if cache is not None:
        #   handle extending k, v, etc.
        #   omitted here for brevity

        out, _ = self.mha(
            q, k, v,
            attn_mask=attention_mask,
            key_padding_mask=encoding_mask,
            need_weights=False
        )
        return out
    
# ------------------------------------------------------------------
# MLP
# ------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)

        nn.init.normal_(self.w1.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.w2.weight, mean=0.0, std=config.initializer_range)
        nn.init.normal_(self.w3.weight, mean=0.0, std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# ------------------------------------------------------------------
# TransformerBlock
# ------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, config: Dict, layer_loc: float):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.use_1d_rotary = config.switch_rotary_to_1d <= layer_loc # For accessment from DecoderViT
        self.attention = RotaryMultiheadAttention(config, use_1d_rotary=self.use_1d_rotary)
        self.mlp = MLP(config)

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        encoding_mask: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None, 
        cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        # Attention
        attn_in = self.attention_norm(hidden_states)
        attn_out = self.attention(attn_in, encoding_mask=encoding_mask, attention_mask=attention_mask, position_ids=position_ids, cache=cache)
        hidden_states = hidden_states + attn_out

        # MLP
        ffn_in = self.ffn_norm(hidden_states)
        ffn_out = self.mlp(ffn_in)
        hidden_states = hidden_states + ffn_out
        return hidden_states

# ------------------------------------------------------------------
# EncoderViT
# ------------------------------------------------------------------

class EncoderViT(nn.Module):
    """Vision Transformer Encoder for 3D Video"""
    def __init__(
        self, 
        in_channels: int = 3, 
        z_channels: int = 512,
        spatial_compression: int = 8,
        temporal_compression: int = 16,
        **kwargs
    ):
        super().__init__()

        # ---------- Config for ElasticTok ----------
        # Notice that some of parameters are redundancy to keep the same interface with the original codebase
        # Apart from settings of Patcher & Quantizer, other parameters are directly inherited from ElasticTokConfig
        self.config = kwargs.get('vit_config')

        # ---------- Patchify ----------
        # self.patch_size = kwargs.get('patch_size', 8)
        self.patch_method = kwargs.get('patch_method', "rearrange")
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression
        
        self.patcher = Patcher3DArbitrary(
            spatial_patch_size=self.spatial_compression,
            temporal_patch_size=self.temporal_compression,
            patch_method=self.patch_method,
        )

        # TODO: Add 1D latent tokens & Learnable Positional Embedding
        # ---------- 1D Latent Tokens & Its Positional Embedding ----------
        
        # ---------- Temporal Embedding ----------
        self.max_num_video_frames = kwargs.get('max_num_video_frames', 121)
        self.temporal_embed = nn.Parameter(torch.empty(self.max_num_video_frames, self.config.hidden_size))
        nn.init.kaiming_normal_(self.temporal_embed)

        # ---------- Input Projection ----------
        patch_dims = in_channels * self.temporal_compression * self.spatial_compression ** 2

        self.patch_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.patch_proj = nn.Linear(patch_dims, self.config.hidden_size, bias=False)
        nn.init.normal_(self.patch_proj.weight, mean=0.0, std=self.config.initializer_range)

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.config, 'num_encoder_layers')
        self.all_blocks = nn.ModuleList([TransformerBlock(self.config, layer_loc=i/num_layers) for i in range(num_layers)])
        print("1D block across encoder:", [w.attention.use_1d_rotary for w in self.all_blocks])

        
        # ---------- Output Projection ----------
        # self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.output_proj = nn.Linear(self.config.hidden_size, z_channels, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, 
                x: torch.Tensor,         # [B, S, hidden_size]
                encoding_mask: torch.Tensor = None,  # [B, S] bool
                attention_mask: torch.Tensor = None, # [B, S] float
                position_ids: Optional[torch.Tensor] = None,
                cache: Optional[Dict[str, torch.Tensor]] = None,
                training: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # x: (B, C, T, H, W)

        # ---------- 1.Patchify ----------
        x = self.patcher(x)  # (B, C', T', H', W')
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, -1)  # (B, C', T'*H'*W')
        x = x.permute(0, 2, 1)  # (B, T'*H'*W', C)
        # ---------- 2. Input Projection ----------
        x = self.patch_proj(x)  # (B, T'*H'*W', D)

        # apply learnable 
        x = x.reshape(B, T, H * W, -1) + self.temporal_embed[None, :T, None]
        x = x.reshape(B, T * H * W, -1)

        # ---------- 3. Add Positional Embedding ----------
        # ---------- 4. Transformer Forward ----------
        for blk in self.all_blocks:
            x = blk(
                x,
                attention_mask=attention_mask,
                position_ids=(T, H, W),
                cache=cache
            )
        
        x = self.patch_norm(x)  # (B, T'*H'*W', D)
        x = self.output_proj(x)  # (B, T * H * W, z_channels)
        
        # ---------- 6. Reshape to 3D ----------
        ### To match original bodebase format
        x = x.permute(0, 2, 1)  # (B, z_channels, T * H * W)
        x = x.reshape(B, -1, T, H, W)  # (B, z_channels, T, H, W)
        return x


class DecoderViT(nn.Module):
    """Vision Transformer Decoder for 3D Video"""
    def __init__(
        self, 
        out_channels: int = 3, 
        z_channels: int = 512,
        spatial_compression: int = 8, 
        temporal_compression: int = 16, 
        **kwargs
    ):
        super().__init__()
        # ---------- Config for ElasticTok ----------
        self.config = kwargs.get('vit_config')

        # ---------- Unpatchify ----------
        # self.patch_size = kwargs.get('patch_size', 8)
        self.patch_method = kwargs.get('patch_method', "rearrange")
        self.spatial_compression = spatial_compression
        self.temporal_compression = temporal_compression

        self.unpatcher = UnPatcher3DArbitrary(
            spatial_patch_size=self.spatial_compression,
            temporal_patch_size=self.temporal_compression,
            patch_method=self.patch_method,
        )

        # ---------- Input Projection ----------
        self.input_proj = nn.Linear(z_channels, self.config.hidden_size, bias=False)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.config.initializer_range)

        # ---------- Temporal Embedding ----------
        self.max_num_video_frames = kwargs.get('max_num_video_frames', 121)
        self.temporal_embed = nn.Parameter(torch.empty(self.max_num_video_frames, self.config.hidden_size))
        nn.init.kaiming_normal_(self.temporal_embed)

        # TODO: Add masked tokens & Learnable Positional Embedding

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.config, 'num_decoder_layers')
        self.all_blocks = nn.ModuleList([TransformerBlock(self.config, layer_loc=(num_layers - i - 1)/num_layers) for i in range(num_layers)])
        print("1D block across decoder:", [w.attention.use_1d_rotary for w in self.all_blocks])
        
        # ---------- Output Projection ----------
        patch_dims = out_channels * self.temporal_compression * (self.spatial_compression ** 2)
        self.patch_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.patch_proj = nn.Linear(self.config.hidden_size, patch_dims, bias=False)
        nn.init.normal_(self.patch_proj.weight, mean=0.0, std=self.config.initializer_range)

        # ---------- Causal Mask ----------
        if self.config.use_causal_decode_1d:
            self.register_buffer('causal_mask', torch.triu(torch.ones(self.config.max_sequence_length_1d, self.config.max_sequence_length_1d, dtype=torch.bool), diagonal=1))
            # This is used for concat_decode_2d, the attention mask should include 2d tokens as well
            self.register_buffer('neg_attn_mask', torch.ones(self.config.max_sequence_length_1d * 2, self.config.max_sequence_length_1d * 2, dtype=torch.bool))

        # ---------- 2D Decoding ----------
        if self.config.concat_decode_2d:
            assert not self.all_blocks[-1].use_1d_rotary, ValueError("When using concat_decode_2d, at least the last decoder layer should be using 2d embeddings.")
            assert self.all_blocks[0].use_1d_rotary, ValueError("When using concat_decode_2d, the first decoder layer should be using 1d embeddings.")
            self.pos_emb_1d = nn.Parameter(torch.randn(self.config.hidden_size) * self.config.initializer_range)
            self.pos_emb_2d = nn.Parameter(torch.randn(self.config.hidden_size) * self.config.initializer_range)
            self.masked_token = nn.Parameter(torch.randn(self.config.max_sequence_length, self.config.max_sequence_length, self.config.hidden_size) * self.config.initializer_range)


    def forward(
            self, 
            x: torch.Tensor,
            encoding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass of the ViT-based 3D decoder.
        
        Args:
            x: Input tensor of shape [B, z_channels, T, H, W]
            encoding_mask: [B, T*H*W], 1 indicates the token is masked, 0 indicates the token is visible, used for adaptive tokenization
            attention_mask: [T*H*W, T*H*W], 1 indicates the token is masked, 0 indicates the token is visible, currently as placeholder
            cache: Optional key-value cache for transformer blocks
            
        Returns:
            Decoded tensor of shape [B, out_channels, T, H, W]
        """
        # Input shape: (B, z_channels, T, H, W)
        # ---------- 1. Reshape to 1D ----------
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, -1)  # (B, z_channels, T*H*W)
        x = x.permute(0, 2, 1)  # (B, T*H*W, z_channels)
        
        # ---------- 2. Input Projection ----------
        x = self.input_proj(x)  # (B, T*H*W, C)
        x = x.permute(0, 2, 1)  # (B, C, T*H*W)
        x = x.reshape(B, -1, T, H, W)  # (B, C, T, H, W)

        x = x + self.temporal_embed.permute(1, 0)[None, :, :T, None, None]  # (B, C, T, H, W)

        B, C, T, H, W = x.shape
        N = T * H * W
        x = x.reshape(B, C, -1)  # (B, C, T*H*W)
        x = x.permute(0, 2, 1)  # (B, T*H*W, C)
        switch_2d = False
        for blk in self.all_blocks:
            # Add causal mask at 1d decoding stage
            applied_attention_mask = None
            applied_encoding_mask = None
            if self.config.use_causal_decode_1d and blk.use_1d_rotary:
                causal_mask = self.causal_mask[:x.shape[1], :x.shape[1]]
                applied_attention_mask = causal_mask if attention_mask is None else attention_mask | causal_mask
                applied_encoding_mask = encoding_mask
            # Expanding encoding mask and attention mask to adapt to [x_1d, x_2d] rather than x_1d
            elif self.config.concat_decode_2d and not blk.use_1d_rotary:
                applied_attention_mask = self.neg_attn_mask[:2*N, :2*N]
                causal_mask_1d = self.causal_mask[:N, :N]
                applied_attention_mask[:N, :N] = causal_mask_1d if attention_mask is None else attention_mask | causal_mask_1d
                block_causal_mask = build_block_causal_mask(T, H, W).to(applied_attention_mask.device)
                applied_attention_mask[N:, :N] = block_causal_mask # 2d refer to 1d
                applied_attention_mask[N:, N:] = block_causal_mask # 2d refer to 2d
                if encoding_mask is not None:
                    applied_encoding_mask = torch.cat([encoding_mask, torch.zeros_like(encoding_mask)], dim=-1) # zero indicates all 2d concated tokens are visible for each 1d token
            # Otherwise, keep original attention mask and encoding mask
            else:
                applied_attention_mask = attention_mask
                applied_encoding_mask = encoding_mask
            
            # Switch from 1d to 2d by concatenating tokens, only applicable at first 2d block
            if self.config.concat_decode_2d and not blk.use_1d_rotary and not switch_2d: # the first 2d block
                x_1d = x + self.pos_emb_1d # x_1d can be adaptive in the future
                x_2d = self.masked_token[:H, :W].reshape(H*W, -1)
                x_2d = x_2d[None, None].repeat(B, T, 1, 1) + self.pos_emb_2d
                x_2d = (x_2d + self.temporal_embed[None, :T, None]).view(B, T*H*W, -1)
                x = torch.cat([x_1d, x_2d], dim=1)
                switch_2d = True

            x = blk(
                x,
                encoding_mask=applied_encoding_mask,
                attention_mask=applied_attention_mask,
                position_ids=(T, H, W),
                cache=cache
            )

        if self.config.concat_decode_2d:
            x = x[:, -T*H*W:, :] # The output is 2d tokens
        
        x = self.patch_norm(x)  # (B, T*H*W, C)
        x = self.patch_proj(x)  # (B, T*H*W, D)
        x = x.permute(0, 2, 1)  # (B, D, T*H*W)
        x = x.reshape(B, -1, T, H, W)  # (B, D, T, H, W)
        x = self.unpatcher(x)  # (B, D', T', H', W')
            
        return x
    
def build_block_causal_mask(T: int, H: int, W: int) -> torch.Tensor:
    """
    Constructs a block causal attention mask for a sequence of T blocks,
    each containing H*W tokens.

    Args:
        T (int): Number of blocks.
        H (int): Height of each block.
        W (int): Width of each block.

    Returns:
        torch.Tensor: A (T*H*W, T*H*W) attention mask tensor with True indicating
                      positions that should be masked (i.e., not attended to).
    """
    tokens_per_block = H * W  # Tokens per block
    block_mask = torch.triu(torch.ones((T, T), dtype=torch.bool), diagonal=1)
    token_mask = block_mask.repeat_interleave(tokens_per_block, dim=0).repeat_interleave(tokens_per_block, dim=1)

    return token_mask
