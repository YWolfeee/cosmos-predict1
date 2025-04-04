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

from cosmos_predict1.tokenizer.modules.patching import Patcher3D, UnPatcher3D


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

def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    q, k: [B, S, n_heads, head_dim], head_dim must be even
    freqs_cis: [max_seq_len, head_dim/2, 2]
    """
    bsz, seq_len, n_heads, head_dim = q.shape
    # slice out the needed positions
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
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        self.head_dim = embed_dim // num_heads

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        # Build a buffer for rotary
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(self.head_dim, config.max_sequence_length, config.theta),
            persistent=False
        )

        # Initialize parameters as in the JAX code
        nn.init.normal_(self.mha.in_proj_weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.mha.in_proj_bias)
        nn.init.normal_(self.mha.out_proj.weight, mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None, cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        hidden_states: [B, S, E]
        attention_mask: [B, S] where 0=masked, 1=keep, or None
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
        q = q.view(B, S, n_heads, head_dim)
        k = k.view(B, S, n_heads, head_dim)

        # If position_ids is None, assume range(S)
        if position_ids is None:
            position_ids = torch.arange(S, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(B, S) # this is not used?

        # For advanced usage, you'd gather exact freq for each position, but here we
        # assume sequences are uniform (like Llama).
        q, k = apply_rotary_emb(q, k, self.freqs_cis)

        # Reshape back to [B, S, E]
        q = q.view(B, S, E)
        k = k.view(B, S, E)

        # Optional caching for incremental decode
        # if cache is not None:
        #   handle extending k, v, etc.
        #   omitted here for brevity

        # MHA expects (batch_first=True)
        # Convert an attention_mask [B, S] of 1/0 into a "key_padding_mask" of shape [B, S].
        # We interpret 1 => keep, 0 => pad
        key_padding_mask = None
        if attention_mask is not None:
            # We want 1 => "non-masked" and 0 => "masked" for MHA's key_padding_mask
            # However, PyTorch MHA's key_padding_mask has True => masked, False => keep
            # So we invert
            key_padding_mask = (attention_mask < 1)

        out, _ = self.mha(
            q, k, v,
            key_padding_mask=key_padding_mask,
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
    def __init__(self, config: Dict):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.attention = RotaryMultiheadAttention(config)
        self.mlp = MLP(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None, cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        # Attention
        attn_in = self.attention_norm(hidden_states)
        attn_out = self.attention(attn_in, attention_mask=attention_mask, position_ids=position_ids, cache=cache)
        hidden_states = hidden_states + attn_out

        # MLP
        ffn_in = self.ffn_norm(hidden_states)
        ffn_out = self.mlp(ffn_in)
        hidden_states = hidden_states + ffn_out
        return hidden_states

# ------------------------------------------------------------------
# EncoderViT
# ------------------------------------------------------------------

# TODO: handle x with different aspect ratio:
        #       During video training, x can have varying shapes (e.g., [256, 192] or [256, 256])
        #       which results in different token counts. We need to handle these variable dimensions.
        #       This can be implemented by padding dummy tokens & modify attention mask accordingly.
        #       This should take effect for both EncoderViT & DecoderViT.

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

        # ---------- Config for ViT ----------
        self.vit_config = kwargs.get('vit_config')

        # ---------- Patchify ----------
        self.patch_size = kwargs.get('patch_size', 8)
        self.patch_method = kwargs.get('patch_method', "rearrange")
        assert self.patch_size <= spatial_compression
        assert spatial_compression % self.patch_size == 0, f"spatial_compression ({spatial_compression}) must be divisible by patch_size ({self.patch_size}) for proper unpatchification"
        assert self.patch_size == temporal_compression, f"patch_size ({self.patch_size}) must equal temporal_compression ({temporal_compression}) for proper unpatchification"
        # Currently Patcher3D does not support tuple patch_size
        self.patcher = Patcher3D(patch_size=self.patch_size, patch_method=self.patch_method) 
        # We need to further rearange the tokens to satisfy compression rate
        self.extra_spatial_compression = spatial_compression // self.patch_size
        self.extra_temporal_compression = temporal_compression // self.patch_size

        # ---------- Preprocess Token & Positional Embedding ----------
        if self.vit_config.use_latent:
            self.num_patch_tokens = kwargs.get('num_patch_tokens')
            self.num_latent_tokens = kwargs.get('num_latent_tokens')
            scale = self.vit_config.hidden_size ** -0.5
            self.latent_tokens = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.vit_config.hidden_size))
            # Additional Positional Embedding to distinguish latent tokens from patch_tokens
            self.patch_tokens_pos_emb = nn.Parameter(torch.randn(self.num_patch_tokens, self.vit_config.hidden_size))  
            self.latent_tokens_pos_emb = nn.Parameter(torch.randn(self.num_latent_tokens, self.vit_config.hidden_size))
            nn.init.kaiming_normal_(self.latent_tokens_pos_emb)
            nn.init.kaiming_normal_(self.patch_tokens_pos_emb)
        else: # if not use_latent: not use attention mask but use embed to distinguish masked tokens
            self.is_kept_embed = nn.Parameter(torch.empty(1, self.vit_config.hidden_size))
            self.is_masked_embed = nn.Parameter(torch.empty(1, self.vit_config.hidden_size))
            nn.init.kaiming_normal_(self.is_kept_embed)
            nn.init.kaiming_normal_(self.is_masked_embed)
        
        # ---------- Input Projection ----------
        patch_dim = in_channels * temporal_compression * (spatial_compression ** 2)
        self.input_proj = nn.Linear(patch_dim, self.vit_config.hidden_size, bias=False)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.vit_config.initializer_range)

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.vit_config, 'num_encoder_layers')
        self.blocks = nn.ModuleList([TransformerBlock(self.vit_config) for _ in range(num_layers)])
        
        # ---------- Output Projection ----------
        self.norm = RMSNorm(self.vit_config.hidden_size, eps=self.vit_config.rms_norm_eps)
        self.output_proj = nn.Linear(self.vit_config.hidden_size, z_channels, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.vit_config.initializer_range)

    def forward(self, 
                x: torch.Tensor,         # [B, S, hidden_size]
                encoding_mask: torch.Tensor = None,  # [B, S] bool
                attention_mask: torch.Tensor = None, # [B, S] float
                position_ids: Optional[torch.Tensor] = None,
                cache: Optional[Dict[str, torch.Tensor]] = None,
                shape_out: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Input shape: (B, C', T', H', W')
        # Where:
        #   B = batch size
        #   C' = number of channels (3 for RGB)
        #   T' = number of frames (equals kwargs.get('num_video_frames', 121) in __init__)
        #   H' = height of frames (equals kwargs.get('crop_height', 256) in __init__)
        #   W' = width of frames (equals kwargs.get('crop_width', 256) in __init__)
        # Note: This is the raw input before any patchification is applied
        B = x.shape[0]

        # ---------- 1.Patchify to 1D ---------- 
        x = self.patcher(x)  # (B, C' * patch_size**3, T' / patch_size, H' / patch_size, W' / patch_size)
        ### Further rearange the token to satisfy compression rate
        if self.extra_spatial_compression > 1 or self.extra_temporal_compression > 1:
            x = rearrange(
                x,
                "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
                p1=self.extra_temporal_compression,
                p2=self.extra_spatial_compression,
                p3=self.extra_spatial_compression,
            ).contiguous() 
        # x: (B, C' * (sc ** 2) * tc, T' / tc, H' / sc, W' / sc), postpatchified
        # sc: spatial_compression; tc: temporal_compression
        B, C, T, H, W = x.shape
        ### Flatten the spatial and temporal dimensions
        x = x.reshape(B, C, -1) # (B, C, T * H * W)
        x = x.permute(0, 2, 1) # (B, T * H * W, C)
        
        # ---------- 2. Input Projection ----------
        x = self.input_proj(x) # (B, N1, D), N1 = T * H * W, D = self.config.hidden_size

        # ---------- 3. Preprocess Token & Positional Embedding ----------
        if self.vit_config.use_latent:
            x = x + self.patch_tokens_pos_emb.unsqueeze(0) # [B, N1, D]
            latent_x = self.latent_tokens + self.latent_tokens_pos_emb # [N2, D]
            latent_x = latent_x.unsqueeze(0).repeat(B, 1, 1)  # [B, N2, D]
            x = torch.cat([x, latent_x], dim=1)  # [B, N1+N2, D]
        elif encoding_mask is not None:
            keep_embed = self.is_kept_embed.unsqueeze(1)  # (1, 1, hidden_size)
            mask_embed = self.is_masked_embed.unsqueeze(1)  # (1, 1, hidden_size)
            x = x + torch.where(encoding_mask.unsqueeze(-1), keep_embed, mask_embed)
        
        # ---------- 4. Transformer Forward ----------
        for blk in self.blocks:
            x = blk(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache
            )  # Maintains shape # use_latent: (B, N1+N2, D); else: (B, N1, D)
        
        # ---------- 5. Output Projection ----------
        if self.vit_config.use_latent:
            x = x[:, -self.num_latent_tokens:, :]
        x = self.norm(x)  # use_latent: (B, N2, D); else: (B, N1, D)
        x = self.output_proj(x)  # use_latent: (B, N2, z_channels); else: (B, N1, z_channels)

        # ---------- 6. Reshape to Pesudo 3D ----------
        x = x.permute(0, 2, 1) # (B, z_channels, N1)
        x = x.reshape(B, x.shape[1], x.shape[2], 1, 1) # (B, z_channels, N1, 1, 1)
        
        if shape_out:
            return x, (T, H, W)
        else:
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
        # ---------- Config for ViT ----------
        self.vit_config = kwargs.get('vit_config')
        self.default_num_video_frames = kwargs.get('num_video_frames', 121)
        self.default_video_height = kwargs.get('crop_height', 256)

        # ---------- Input Projection ----------
        self.input_proj = nn.Linear(z_channels, self.vit_config.hidden_size, bias=False)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.vit_config.initializer_range)

        # ---------- Preprocess Token & Positional Embedding ----------
        if self.vit_config.use_latent:
            self.num_patch_tokens = kwargs.get('num_patch_tokens')
            self.num_latent_tokens = kwargs.get('num_latent_tokens')
            scale = self.vit_config.hidden_size ** -0.5
            self.mask_token = nn.Parameter(scale * torch.randn(1, self.vit_config.hidden_size))
            self.patch_tokens_pos_emb = nn.Parameter(torch.randn(self.num_patch_tokens, self.vit_config.hidden_size))
            self.latent_tokens_pos_emb = nn.Parameter(torch.randn(self.num_latent_tokens, self.vit_config.hidden_size))
            nn.init.kaiming_normal_(self.latent_tokens_pos_emb)
            nn.init.kaiming_normal_(self.patch_tokens_pos_emb)
        else:
            self.is_kept_embed = nn.Parameter(torch.empty(1, self.vit_config.hidden_size))
            self.is_masked_embed = nn.Parameter(torch.empty(1, self.vit_config.hidden_size))
            nn.init.kaiming_normal_(self.is_kept_embed)
            nn.init.kaiming_normal_(self.is_masked_embed)

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.vit_config, 'num_encoder_layers')
        self.blocks = nn.ModuleList([TransformerBlock(self.vit_config) for _ in range(num_layers)])
        
        # ---------- Output Projection ----------
        self.norm = RMSNorm(self.vit_config.hidden_size, eps=self.vit_config.rms_norm_eps)
        out_channels = out_channels * (spatial_compression ** 2) * temporal_compression
        self.output_proj = nn.Linear(self.vit_config.hidden_size, out_channels, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.vit_config.initializer_range)

        # ---------- Unpatchify ----------
        self.patch_size = kwargs.get('patch_size', 8)
        assert self.patch_size <= spatial_compression
        assert spatial_compression % self.patch_size == 0, f"spatial_compression ({spatial_compression}) must be divisible by patch_size ({self.patch_size}) for proper unpatchification"
        assert self.patch_size == temporal_compression, f"patch_size ({self.patch_size}) must equal temporal_compression ({temporal_compression}) for proper unpatchification"
        # Currently UnPatcher3D does not support tuple patch_size
        self.unpatcher = UnPatcher3D(patch_size=self.patch_size) 
        # We need to further rearange the tokens to satisfy compression rate
        self.extra_spatial_compression = spatial_compression // self.patch_size
        self.extra_temporal_compression = temporal_compression // self.patch_size

    def forward(
            self, 
            x: torch.Tensor,
            clip_shape: Tuple[int, int, int] = None,
            encoding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            cache: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Input shape: use_latent: (B, z_channels, N2, 1, 1); else: (B, z_channels, N1, 1, 1)
        B, C, N, _, _ = x.shape # Dummy Feature Map: (B, C, N, 1, 1)
        if clip_shape is None:
            # Calculate dimensions based on compression factors
            default_t = self.default_num_video_frames // self.extra_temporal_compression
            default_h = self.default_video_height // self.extra_spatial_compression
            default_w = self.default_video_height // self.extra_spatial_compression
            clip_shape = (default_t, default_h, default_w)
        T, H, W = clip_shape

        # ---------- 1. Reshape to 1D ----------
        x = x.reshape(B, N, -1)  # use_latent: (B, N2, z_channels); else: (B, N1, z_channels)
        
        # ---------- 2. Input Projection ----------
        x = self.input_proj(x)  # use_latent: (B, N2, hidden_size); else: (B, N1, hidden_size)
        
        # ---------- 3. Preprocess Token & Positional Embedding ----------
        if self.vit_config.use_latent:
            x = x + self.latent_tokens_pos_emb.unsqueeze(0) # [B, N2, D]
            masked_x = self.mask_token + self.patch_tokens_pos_emb # [N1, D]
            masked_x = masked_x.unsqueeze(0).repeat(B, 1, 1) # [B, N1, D]
            x = torch.cat([masked_x, x], dim=1) # [B, N1+N2, D], the concatenate order is to ensure attention mask is as the same as EncoderViT
        elif encoding_mask is not None: # use patch_only
            keep_embed = self.is_kept_embed.unsqueeze(1)  # (1, 1, hidden_size)
            mask_embed = self.is_masked_embed.unsqueeze(1)  # (1, 1, hidden_size)
            x = x + torch.where(encoding_mask.unsqueeze(-1), keep_embed, mask_embed)
        
        # ---------- 4. Transformer Forward ----------
        for blk in self.blocks:
            x = blk(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache
            )  # Maintains shape: use_latent: (B, N1+N2, hidden_size); else: (B, N1, hidden_size)
        
        # ---------- 5. Output Projection ----------
        if self.vit_config.use_latent:
            x = x[:, :self.num_patch_tokens, :]
        x = self.norm(x)  # use_latent: (B, N1, hidden_size)
        x = self.output_proj(x)  # use_latent: (B, N1, out_channels*patch_size^3);
        
        # ---------- 6. Unpatchify to 3D ----------
        x = x.permute(0, 2, 1)  # (B, out_channels*patch_size^3, T*H*W)
        x = x.reshape(B, -1, T, H, W)  # (B, out_channels*patch_size^3, T, H, W)
        if self.extra_spatial_compression > 1 or self.extra_temporal_compression > 1:
            x = rearrange(
                x,
                "b (c p1 p2 p3) t h w -> b c (t p1) (h p2) (w p3)",
                p1=self.extra_temporal_compression,
                p2=self.extra_spatial_compression,
                p3=self.extra_spatial_compression,
            ).contiguous()
        
        x = self.unpatcher(x)  # (B, out_channels, T*patch_size, H*patch_size, W*patch_size)
        
        return x
