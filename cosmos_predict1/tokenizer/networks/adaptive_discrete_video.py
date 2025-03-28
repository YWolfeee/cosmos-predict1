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

"""The network definition for 1D-adaptive discrete video tokenizer with VQ, LFQ, FSQ or ResidualFSQ."""
from collections import OrderedDict, namedtuple
from typing import Tuple, Dict, Any, Optional, Union

import torch
from loguru import logger as logging
from torch import nn

from cosmos_predict1.tokenizer.modules import Decoder3DType, DiscreteQuantizer, Encoder3DType
from cosmos_predict1.tokenizer.modules.layers3d import CausalConv3d
from cosmos_predict1.tokenizer.modules.quantizers import InvQuantizerJit

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "quant_loss", "quant_info"])

class AdaptiveTokenizationModule(nn.Module):
    """Module for computing adaptive token allocation and masking.
    
    Takes rate allocation scores (or elbo estimates) and computes a token allocation
    and corresponding mask matrix.
    """
    
    def __init__(self, 
                 min_tokens: int = 256, 
                 max_tokens: int = 2048,
                 rate_strategy: str = 'uniform',
                 **kwargs) -> None:
        """Initialize the adaptive tokenization module.
        
        Args:
            min_tokens: Minimum number of tokens to allocate per video.
            max_tokens: Maximum number of tokens to allocate per video.
            rate_strategy: Strategy for determining token allocation rate.
                Options: 'uniform', 'elbo'.
            **kwargs: Additional kwargs.
        """
        super().__init__()
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.rate_strategy = rate_strategy
        
    def compute_token_allocation(self, 
                                 rate_scores: torch.Tensor, 
                                 total_token_budget: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token allocation based on rate scores.
        
        Args:
            rate_scores: Tensor of shape [B, T] with scores for each video block.
                         Higher scores indicate more tokens should be allocated.
            total_token_budget: Optional total token budget for the batch.
                                If None, uses max_tokens * batch_size.
        
        Returns:
            tokens_per_block: Tensor of shape [B, T] with number of tokens per block.
            mask: Binary mask of shape [B, T, max_spatial_tokens] indicating which tokens to keep.
        """
        batch_size, num_blocks = rate_scores.shape
        
        # Default budget is max_tokens per video
        if total_token_budget is None:
            total_token_budget = self.max_tokens * batch_size
            
        # Apply rate strategy to convert scores to allocation ratios [0, 1]
        if self.rate_strategy == 'uniform':
            # Uniform random allocation
            allocation_ratios = torch.rand_like(rate_scores)
        elif self.rate_strategy == 'elbo':
            # Higher rate_scores (lower ELBOs) get more tokens
            allocation_ratios = rate_scores / (rate_scores.sum(dim=1, keepdim=True) + 1e-8)
        else:
            raise ValueError(f"Unknown rate strategy: {self.rate_strategy}")
        
        # Compute tokens per block
        total_blocks = batch_size * num_blocks
        min_tokens_per_block = torch.ones_like(allocation_ratios) * self.min_tokens / num_blocks
        remaining_budget = total_token_budget - (self.min_tokens * batch_size)
        
        # Allocate remaining budget according to allocation ratios
        additional_tokens = allocation_ratios * remaining_budget / batch_size
        tokens_per_block = min_tokens_per_block + additional_tokens
        
        # Ensure max_tokens constraint per video
        tokens_per_video = tokens_per_block.sum(dim=1, keepdim=True)
        scaling_factor = torch.minimum(
            torch.ones_like(tokens_per_video),
            self.max_tokens / tokens_per_video
        )
        tokens_per_block = tokens_per_block * scaling_factor
        
        return tokens_per_block.int(), self._create_mask_from_allocation(tokens_per_block)
    
    def _create_mask_from_allocation(self, tokens_per_block: torch.Tensor) -> torch.Tensor:
        """Create a mask from token allocation.
        
        Args:
            tokens_per_block: Tensor of shape [B, T] with number of tokens per block.
            
        Returns:
            mask: Binary mask of shape [B, T, max_spatial_tokens] 
                  where max_spatial_tokens is the spatial dimension of tokens.
        """
        # This is a placeholder implementation
        # Real implementation will depend on the spatial dimension of tokens
        batch_size, num_blocks = tokens_per_block.shape
        max_spatial_tokens = self.max_tokens // num_blocks
        
        # Create mask of ones followed by zeros for each block
        masks = []
        for b in range(batch_size):
            block_masks = []
            for t in range(num_blocks):
                num_tokens = min(tokens_per_block[b, t].item(), max_spatial_tokens)
                block_mask = torch.zeros(max_spatial_tokens, device=tokens_per_block.device)
                block_mask[:num_tokens] = 1.0
                block_masks.append(block_mask)
            masks.append(torch.stack(block_masks, dim=0))
        
        return torch.stack(masks, dim=0)

class AdaptiveDiscreteVideoTokenizer(nn.Module):
    """1D-adaptive discrete video tokenizer.
    
    Implements a tokenizer that:
    1. Processes video into 2D spatial patches
    2. Converts these patches to a 1D sequence
    3. Adaptively selects varying number of tokens per video chunk
    4. Applies discrete quantization (FSQ or LFQ)
    """
    
    def __init__(self, 
                 z_channels: int, 
                 z_factor: int, 
                 embedding_dim: int, 
                 **kwargs) -> None:
        """Initialize the 1D-adaptive discrete video tokenizer.
        
        Args:
            z_channels: Number of channels in the latent representation.
            z_factor: Factor to multiply z_channels for the encoder output.
            embedding_dim: Dimension of the embedding for quantization.
            **kwargs: Additional kwargs for configuration.
        """
        super().__init__()
        self.name = kwargs.get("name", "AdaptiveDiscreteVideoTokenizer")
        self.embedding_dim = embedding_dim
        
        # Tokenization parameters
        self.min_tokens = kwargs.get("min_tokens", 256)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.rate_strategy = kwargs.get("rate_strategy", "uniform")
        
        # Initialize 3D encoder (similar to CausalDiscreteVideoTokenizer)
        encoder_name = kwargs.get("encoder", Encoder3DType.BASE.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)
        
        # Initialize 3D decoder (similar to CausalDiscreteVideoTokenizer)
        decoder_name = kwargs.get("decoder", Decoder3DType.BASE.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)
        
        # Convolutional layers for dimensionality transformations
        self.quant_conv = CausalConv3d(z_factor * z_channels, embedding_dim, kernel_size=1, padding=0)
        self.post_quant_conv = CausalConv3d(embedding_dim, z_channels, kernel_size=1, padding=0)
        
        # Spatial to 1D sequence transformation
        self.spatial_to_sequence = None  # Implemented in setup based on config
        self.sequence_to_spatial = None  # Implemented in setup based on config
        
        # Initialize quantizer (FSQ or LFQ)
        quantizer_name = kwargs.get("quantizer", DiscreteQuantizer.FSQ.name)
        if quantizer_name == DiscreteQuantizer.LFQ.name:
            assert "codebook_size" in kwargs, f"`codebook_size` must be provided for {quantizer_name}."
            assert "codebook_dim" in kwargs, f"`codebook_dim` must be provided for {quantizer_name}."
        elif quantizer_name == DiscreteQuantizer.FSQ.name:
            assert "levels" in kwargs, f"`levels` must be provided for {quantizer_name}."
        self.quantizer = DiscreteQuantizer[quantizer_name].value(**kwargs)
        
        # Adaptive tokenization module
        self.adaptive_tokenization = AdaptiveTokenizationModule(
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
            rate_strategy=self.rate_strategy,
            **kwargs
        )
        
        logging.info(f"{self.name} based on {quantizer_name}, with {kwargs}.")
        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")
    
    def to(self, *args, **kwargs):
        """Override to method to ensure quantizer dtype is set."""
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(AdaptiveDiscreteVideoTokenizer, self).to(*args, **kwargs)
    
    def reshape_3d_to_1d(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape 3D tokens to 1D sequence.
        
        Args:
            x: Tensor of shape [B, C, T, H, W]
            
        Returns:
            Tensor of shape [B, C, T*H*W]
        """
        B, C, T, H, W = x.shape
        return x.reshape(B, C, T*H*W)
    
    def reshape_1d_to_3d(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """Reshape 1D sequence back to 3D tokens.
        
        Args:
            x: Tensor of shape [B, C, T*H*W]
            T: Time dimension
            H: Height dimension
            W: Width dimension
            
        Returns:
            Tensor of shape [B, C, T, H, W]
        """
        B, C, _ = x.shape
        return x.reshape(B, C, T, H, W)
    
    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to the 1D sequence.
        
        Args:
            x: Tensor of shape [B, C, L] where L is the sequence length
            mask: Tensor of shape [B, L] with 1s for tokens to keep, 0s for tokens to mask
            
        Returns:
            Masked tensor of shape [B, C, L]
        """
        # Expand mask to match dimensions
        mask = mask.unsqueeze(1)  # [B, 1, L]
        return x * mask
    
    def encode(self, x: torch.Tensor, rate_scores: Optional[torch.Tensor] = None) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """Encode input video to discrete tokens.
        
        Args:
            x: Input tensor of shape [B, C, T, H, W]
            rate_scores: Optional tensor of shape [B, T] with scores for token allocation
            
        Returns:
            quant_info: Dictionary with quantization information
            quant_codes: Quantized codes
            quant_loss: Quantization loss
        """
        # Encode input to latent representation
        h = self.encoder(x)
        h = self.quant_conv(h)
        
        # Store original 3D shape
        B, C, T, H, W = h.shape
        
        # Reshape to 1D sequence
        h_1d = self.reshape_3d_to_1d(h)
        
        # Compute adaptive token allocation if rate_scores provided
        mask = None
        if rate_scores is not None:
            _, mask = self.adaptive_tokenization.compute_token_allocation(rate_scores)
            mask_1d = mask.reshape(B, -1)  # Reshape to [B, T*H*W]
            
            # Apply mask to 1D sequence
            h_1d = self.apply_mask(h_1d, mask_1d)
        
        # Quantize the 1D sequence
        quant_info, quant_codes, quant_loss = self.quantizer(h_1d)
        
        # Reshape back to 3D for decoder
        quant_codes_3d = self.reshape_1d_to_3d(quant_codes, T, H, W)
        
        return quant_info, quant_codes_3d, quant_loss
    
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """Decode quantized representation to video.
        
        Args:
            quant: Quantized tensor of shape [B, C, T, H, W]
            
        Returns:
            Reconstructed video
        """
        quant = self.post_quant_conv(quant)
        return self.decoder(quant)
    
    def decode_code(self, code_b: torch.Tensor) -> torch.Tensor:
        """Decode from discrete codes.
        
        Args:
            code_b: Discrete codes
            
        Returns:
            Reconstructed video
        """
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b)
    
    def forward(self, input: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None, rate_scores: Optional[torch.Tensor] = None) -> Union[Dict[str, torch.Tensor], NetworkEval]:
        """Forward pass of the tokenizer.
        
        Args:
            input: Input video of shape [B, C, T, H, W]
            mask_matrix: Optional mask matrix for tokens
            rate_scores: Optional tensor of shape [B, T] with scores for token allocation
            
        Returns:
            Dictionary or NetworkEval with reconstructions, quant_loss, and quant_info
        """
        quant_info, quant_codes, quant_loss = self.encode(input, rate_scores)
        reconstructions = self.decode(quant_codes)
        
        if self.training:
            return dict(
                reconstructions=reconstructions,
                quant_loss=quant_loss,
                quant_info=quant_info,
                mask_matrix=mask_matrix,  # Pass mask matrix along if provided
            )
        return NetworkEval(
            reconstructions=reconstructions,
            quant_loss=quant_loss,
            quant_info=quant_info,
        ) 