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
                 min_tokens_rate: float = 0.06, 
                 mean_tokens_rate: float = 0.5,
                 rate_strategy: str = 'uniform',
                 **kwargs) -> None:
        """Initialize the adaptive tokenization module.
        
        Args:
            min_tokens_rate: Minimum number of tokens / Full number of tokens, allocated per video.
            mean_tokens_rate: Mean number of tokens / Full number of tokens, allocated per video.
            rate_strategy: Strategy for determining token allocation rate.
                Options: 'uniform', 'elbo'.
            **kwargs: Additional kwargs.
        """
        super().__init__()
        self.min_tokens_rate = min_tokens_rate
        self.mean_tokens_rate = mean_tokens_rate
        self.rate_strategy = rate_strategy
        
    def compute_token_allocation(self,
                                 x: torch.Tensor, 
                                 rate_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute token allocation based on rate scores.
        
        Args:
            x: Tensor of shape [B, C, T, H, W], providing information of T, H, W.
            rate_scores: Tensor of shape [B, T] with scores for each video block.
                         Higher scores indicate more tokens should be allocated.
        
        Returns:
            tokens_per_block: Tensor of shape [B, T] with number of tokens per block.
            mask: Binary mask of shape [B, T, max_spatial_tokens] indicating which tokens to keep.
        """
        batch_size, _, num_blocks, height, width = x.shape
        max_tokens_per_video = num_blocks * height * width
            
        # Apply rate strategy to convert scores to allocation ratios of each block in each batch (sum over blocks of all batch is one)
        if self.rate_strategy == 'uniform':
            # Uniform random allocation
            allocation_ratios = torch.rand(batch_size, num_blocks, device=x.device, dtype=x.dtype).clip(0.0625)
        elif self.rate_strategy == 'unibin':
            # Uniform binary allocation from predefined values
            bins = torch.tensor([0.0625, 0.125, 0.25, 0.5, 0.75, 1.0], 
                                          device=x.device, dtype=x.dtype)
            # Randomly select indices for each batch and block
            indices = torch.randint(0, len(bins), 
                                   (batch_size, num_blocks), 
                                   device=x.device)
            # Use indices to select values from possible_values
            allocation_ratios = bins[indices]
        elif self.rate_strategy == 'elbo':
            # Higher rate_scores (lower ELBOs) get more tokens
            assert rate_scores is not None, "rate_scores must be provided for elbo strategy"
            allocation_ratios = rate_scores
        elif self.rate_strategy == 'static':
            allocation_ratios = torch.ones(batch_size, num_blocks, device=x.device, dtype=x.dtype)
        else:
            raise ValueError(f"Unknown rate strategy: {self.rate_strategy}")
        
        # Compute tokens per block
        max_tokens_per_block = max_tokens_per_video // num_blocks
        tokens_per_block = (allocation_ratios * max_tokens_per_block).int()
        
        return tokens_per_block, self._create_mask_from_allocation(tokens_per_block, max_tokens_per_video)
    
    def _create_mask_from_allocation(self, tokens_per_block: torch.Tensor, max_tokens_per_video: int) -> torch.Tensor:
        """Create a mask from token allocation.
        
        Args:
            tokens_per_block: Tensor of shape [B, T] with number of tokens per block.
            
        Returns:
            mask: Binary mask of shape [B, T, max_tokens_per_block], 1 for visible tokens and 0 for masked tokens, e.g., [0,0,0,1,1,1,1,1,1,1]
                  where max_tokens_per_block is the spatial dimension of tokens.
        """
        batch_size, num_blocks = tokens_per_block.shape
        max_tokens_per_block = max_tokens_per_video // num_blocks
        # Clamp tokens_per_block to max_tokens_per_block
        tokens_per_block = torch.clamp(tokens_per_block, max=max_tokens_per_block)
        # Create indices tensor for each position in max_tokens_per_block
        indices = torch.arange(max_tokens_per_block, device=tokens_per_block.device)
        indices = indices.unsqueeze(0).unsqueeze(0).expand(batch_size, num_blocks, -1)
        tokens_per_block = tokens_per_block.unsqueeze(-1)
        # Create mask: [0,0,0,1,1,1,1,1,1,1]
        mask = torch.where(indices < tokens_per_block, 
                          torch.zeros_like(indices, dtype=torch.float), 
                          torch.ones_like(indices, dtype=torch.float))
        encoding_mask_1d = mask.reshape(batch_size, -1).to(torch.bool)
        return encoding_mask_1d

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
        
        # Initialize 3D encoder (similar to CausalDiscreteVideoTokenizer)
        encoder_name = kwargs.get("encoder", Encoder3DType.ViT.name)
        self.encoder = Encoder3DType[encoder_name].value(z_channels=z_factor * z_channels, **kwargs)
        
        # Initialize 3D decoder (similar to CausalDiscreteVideoTokenizer)
        decoder_name = kwargs.get("decoder", Decoder3DType.ViT.name)
        self.decoder = Decoder3DType[decoder_name].value(z_channels=z_channels, **kwargs)
        
        # Convolutional layers for dimensionality transformations, treated as MLP
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
        
        # Adaptive tokenization module, min_tokens_rate, mean_tokens_rate, rate_strategy are included in kwargs
        self.adaptive_tokenization = AdaptiveTokenizationModule(**kwargs)
        
        logging.info(f"{self.name} based on {quantizer_name}, with {kwargs}.")
        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(f"z_channels={z_channels}, embedding_dim={self.embedding_dim}.")
    
    def to(self, *args, **kwargs):
        """Override to method to ensure quantizer dtype is set."""
        setattr(self.quantizer, "dtype", kwargs.get("dtype", torch.bfloat16))
        return super(AdaptiveDiscreteVideoTokenizer, self).to(*args, **kwargs)
    
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
        h = self.encoder(x, encoding_mask=None, attention_mask=None)
        h = self.quant_conv(h) # h: (B, embedding_dim, T, H, W)
        
        # Store original 3D shape
        B, C, T, H, W = h.shape
        
        # Compute adaptive token allocation
        if self.training:
            tokens_per_block, mask = self.adaptive_tokenization.compute_token_allocation(h, rate_scores)
            encoding_mask_1d = mask.reshape(B, -1)  # Reshape to [B, T*H*W]
        else:
            encoding_mask_1d = None
        
        # Quantize the 1D sequence
        quant_info, quant_codes, quant_loss = self.quantizer(h)
        
        return quant_info, quant_codes, quant_loss, encoding_mask_1d
    
    def decode(self, quant: torch.Tensor, encoding_mask_1d: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode quantized representation to video.
        
        Args:
            quant: Quantized tensor of shape [B, C, T, H, W]
            encoding_mask_1d: Binary mask of shape [B, T*H*W], 1 for masked tokens and 0 for visible tokens, e.g., [[0,0,0,1,1,1,1,1,1,1]]
            
        Returns:
            Reconstructed video
        """
        quant = self.post_quant_conv(quant)
        return self.decoder(quant, attention_mask=None, encoding_mask=encoding_mask_1d)
    
    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("quantizer", self.quantizer),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("inv_quant", InvQuantizerJit(self.quantizer)),
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )
    
    def decode_code(self, code_b: torch.Tensor) -> torch.Tensor:
        """Decode from discrete codes.
        
        Args:
            code_b: Discrete codes
            
        Returns:
            Reconstructed video
        """
        quant_b = self.quantizer.indices_to_codes(code_b)
        quant_b = self.post_quant_conv(quant_b)
        return self.decoder(quant_b, attention_mask=None, encoding_mask=None)
    
    def forward(self, input: torch.Tensor, mask_matrix: Optional[torch.Tensor] = None, rate_scores: Optional[torch.Tensor] = None) -> Union[Dict[str, torch.Tensor], NetworkEval]:
        """Forward pass of the tokenizer.
        
        Args:
            input: Input video of shape [B, C, T, H, W]
            mask_matrix: Optional mask matrix for tokens
            rate_scores: Optional tensor of shape [B, T] with scores for token allocation
            
        Returns:
            Dictionary or NetworkEval with reconstructions, quant_loss, and quant_info
        """
        quant_info, quant_codes, quant_loss, encoding_mask_1d = self.encode(input, rate_scores)
        reconstructions = self.decode(quant_codes, encoding_mask_1d)
        
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
