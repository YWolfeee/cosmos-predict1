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

"""A library for Causal Video Tokenizer inference."""

from typing import Any
from einops import rearrange

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from cosmos_predict1.tokenizer.inference.utils import (
    load_decoder_model,
    load_encoder_model,
    load_model,
    numpy2tensor,
    pad_video_batch,
    tensor2numpy,
    unpad_video_batch,
)


class CausalVideoTokenizer(torch.nn.Module):
    def __init__(
        self,
        checkpoint: str = None,
        checkpoint_enc: str = None,
        checkpoint_dec: str = None,
        tokenizer_config: dict[str, Any] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()
        self._device = device
        self._dtype = getattr(torch, dtype)
        self._full_model = (
            load_model(checkpoint, tokenizer_config, device).to(self._dtype) if checkpoint is not None else None
        )
        self._enc_model = (
            load_encoder_model(checkpoint_enc, tokenizer_config, device).to(self._dtype)
            if checkpoint_enc is not None
            else None
        )
        self._dec_model = (
            load_decoder_model(checkpoint_dec, tokenizer_config, device).to(self._dtype)
            if checkpoint_dec is not None
            else None
        )
    
    @torch.no_grad()
    def collect_video_elbo(self, input_tensor: torch.Tensor, elbo_base: float = 1.0) -> torch.Tensor:
        """Collect ELBO (Evidence Lower Bound) loss for each temporal block.
        
        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            The ELBO loss for each temporal block.
        """
        # Check if self has elbos attribute, if not, create a list
        if not hasattr(self, 'elbos'):
            self.elbos = []
        if self._full_model is not None:
            hidden_tensor = self._full_model.encode(input_tensor)[1] # output tensor: [1, 6, 13, 32, 52]
            block_loss = self.compute_chunk_loss(hidden_tensor, input_tensor)
            video_elbo = torch.mean(block_loss)
            self.elbos.append(video_elbo)
        else:
            raise NotImplementedError("collect_elbo is not implemented for AdaptiveVideoTokenizer")
        
    @torch.no_grad()
    def compute_chunk_loss(self, hidden_tensor: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        recon_full = self._full_model.decode(hidden_tensor)
        full_loss = (input_tensor - recon_full).abs().mean(dim=1)
        chunk_loss = full_loss[:, 1:].reshape(full_loss.shape[0], -1, self._full_model.temporal_compression, *full_loss.shape[2:]).mean(dim=2)
        chunk_loss = torch.concat([full_loss[:, :1], chunk_loss], dim=1)
        chunk_loss = rearrange(chunk_loss, "b t (h u) (w v) -> b t h u w v", u=self._full_model.spatial_compression, v=self._full_model.spatial_compression).mean(dim=(-1,-3))
        return chunk_loss

    @torch.no_grad()
    def autoencode(self, input_tensor: torch.Tensor, strategy: str = "static", avg_rate: float = 0.5) -> torch.Tensor:
        """Reconstrcuts a batch of video tensors after embedding into a latent.

        Args:
            video: The input video Bx3xTxHxW layout, range [-1..1].
        Returns:
            The reconstructed video, layout Bx3xTxHxW, range [-1..1].
        """
        if self._full_model is not None:
            # input tensor: [1, 3, 33, 256, 416]
            hidden_tensor = self._full_model.encode(input_tensor)[1] # output tensor: [1, 6, 9, 32, 52]
            if hasattr(self._full_model, 'method'):
                if self._full_model.method == "mse" or self._full_model.special_attn:
                    mask_seq = True
                else:
                    mask_seq = False
                chunk_loss = self.compute_chunk_loss(hidden_tensor, input_tensor)
                if strategy == "global_elbo":
                    rate = avg_rate * torch.mean(chunk_loss) / self.elbo_mean
                    # Round rate to nearest quarter (0.25, 0.5, 0.75, 1.0)
                    rate = 0.25 * round(rate.item() / 0.25)
                    rate = min(max(rate, 0.25), 1.0)
                    use_strategy = "elbo"
                else:
                    rate = avg_rate
                    use_strategy = strategy
                allocated_ratios = self._full_model.get_allocated_ratios(hidden_tensor.shape, use_adaptive=True, chunk_loss=chunk_loss, manual_base_rate=rate, mask_seq=mask_seq, overwrite_strategy=use_strategy, rescale=False)
                print("Current allocated ratios: ", allocated_ratios)
                hidden_tensor, _ = self._full_model.mask_tokens(hidden_tensor, allocated_ratios, mask_method=self._full_model.method, chunk_loss=chunk_loss)
            output_tensor = self._full_model.decode(hidden_tensor)
            allocated_ratios = torch.ones_like(hidden_tensor[:, 0, :, 0, 0])
        else:
            output_latent = self.encode(input_tensor)[0]
            output_tensor = self.decode(output_latent)
        
        return output_tensor, allocated_ratios
    
    @torch.no_grad()
    def create_mask(self, rate: float, H: int, W: int) -> torch.Tensor:
        token_length = H * W
        indices = torch.arange(token_length, device=self._device)
        mask = torch.where(indices < token_length * rate, 
                           torch.ones_like(indices, dtype=torch.float32), 
                           torch.zeros_like(indices, dtype=torch.float32))
        mask = mask.reshape(H, W)
        return mask
    
    @torch.no_grad()
    def get_block_loss(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        l2_loss = torch.nn.functional.mse_loss(output_tensor, input_tensor, reduction="none")
        l2_loss = l2_loss.mean(dim=(0,1,3,4)) # loss along frames
        num_groups = (l2_loss.shape[0] + 3) // 4
        padded_length = num_groups * 4 - l2_loss.shape[0]
        padded_loss = torch.nn.functional.pad(l2_loss, (0, padded_length), value=l2_loss[0].item())
        block_score = torch.sum(padded_loss.reshape(num_groups, 4), dim=1)
        return block_score
    
    @torch.no_grad()
    def get_token_loss_curve(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self._full_model is not None:
            # input tensor: [1, 3, 49, 256, 416]
            full_tensor = self._full_model.encode(input_tensor)[1] # output tensor: [1, 6, 13, 32, 52]
            output_tensor = self._full_model.decode(full_tensor)
            
            # Get the block score
            block_score = self.get_block_loss(output_tensor, input_tensor)
            
            # Create a list to store rates and scores for all blocks
            rates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.3, 0.1, 0.05]
            num_blocks = len(block_score)
            all_scores = {i: [] for i in range(num_blocks)}
            
            # Mask each block separately with different rates
            current_tensor = full_tensor
            for block_idx in range(num_blocks):
                min_loss = float('inf')
                min_rate = 1.0
                for rate in rates:
                    mask = self.create_mask(rate, full_tensor.shape[-2], full_tensor.shape[-1])
                    # Create a new tensor using multiplication without in-place operations
                    block_mask = torch.ones_like(full_tensor)
                    block_mask[:, :, block_idx] = block_mask[:, :, block_idx] * mask
                    partial_tensor = current_tensor * block_mask
                    output_tensor = self._full_model.decode(partial_tensor)
                    partial_block_score = self.get_block_loss(output_tensor, input_tensor)
                    if torch.sum(partial_block_score[:block_idx+1]) < min_loss:
                        min_loss = torch.sum(partial_block_score[:block_idx+1])
                        min_rate = rate
                        current_tensor[:, :, block_idx] = partial_tensor[:, :, block_idx]
                    # print(f"rate: {rate}, block_idx: {block_idx}, partial_block_score: {partial_block_score}")
                    all_scores[block_idx].append(partial_block_score[block_idx].item())
                print(f"block_idx: {block_idx}, min_rate: {min_rate}")
            
            # Plot and save the results with all blocks in one figure
            print("------ Start All Scores ------")
            print(all_scores)
            print("------ End All Scores ------")
            plt.figure(figsize=(10, 8))
            for block_idx in range(num_blocks):
                plt.plot(rates, all_scores[block_idx], marker='o', linestyle='-', label=f'Block {block_idx}')
            
            plt.xlabel('Rate')
            plt.ylabel('Block Score')
            plt.title('Block Score vs Rate for All Blocks')
            plt.legend()
            plt.grid(True)
            plt.savefig('block_scores.png')
            plt.close()
        else:
            raise NotImplementedError("get_token_loss_curve is not implemented for AdaptiveVideoTokenizer")
        return None

    @torch.no_grad()
    def encode(self, input_tensor: torch.Tensor) -> tuple[torch.Tensor]:
        """Encodes a numpy video into a CausalVideo latent or code.

        Args:
            input_tensor: The input tensor Bx3xTxHxW layout, range [-1..1].
        Returns:
            For causal continuous video (CV) tokenizer, the tuple contains:
                - The latent embedding, Bx16x(t)x(h)x(w), where the compression
                rate is (T/t x H/h x W/w), and channel dimension of 16.
            For causal discrete video (DV) tokenizer, the tuple contains:
              1) The indices, Bx(t)x(h)x(w), from a codebook of size 64K, which
                is formed by FSQ levels of (8,8,8,5,5,5).
              2) The discrete code, Bx6x(t)x(h)x(w), where the compression rate
                is again (T/t x H/h x W/w), and channel dimension of 6.
        """
        assert input_tensor.ndim == 5, "input video should be of 5D."

        output_latent = self._enc_model(input_tensor)
        if isinstance(output_latent, torch.Tensor):
            return output_latent
        return output_latent[:-1]

    @torch.no_grad()
    def decode(self, input_latent: torch.Tensor) -> torch.Tensor:
        """Encodes a numpy video into a CausalVideo latent.

        Args:
            input_latent: The continuous latent Bx16xtxhxw for CV,
                        or the discrete indices Bxtxhxw for DV.
        Returns:
            The reconstructed tensor, layout [B,3,1+(T-1)*8,H*16,W*16] in range [-1..1].
        """
        assert input_latent.ndim >= 4, "input latent should be of 5D for continuous and 4D for discrete."
        return self._dec_model(input_latent)

    def forward(
        self,
        video: np.ndarray,
        temporal_window: int = 17,
        strategy: str = "static",
        avg_rate: float = 0.5,
        collect_elbo_only: bool = False,
    ) -> np.ndarray:
        """Reconstructs video using a pre-trained CausalTokenizer autoencoder.
        Given a video of arbitrary length, the forward invokes the CausalVideoTokenizer
        in a sliding manner with a `temporal_window` size.

        Args:
            video: The input video BxTxHxWx3 layout, range [0..255].
            temporal_window: The length of the temporal window to process, default=25.
        Returns:
            The reconstructed video in range [0..255], layout BxTxHxWx3.
        """
        assert video.ndim == 5, "input video should be of 5D."
        num_frames = video.shape[1]  # can be of any length.
        output_video_list = []

        if "video_elbo" in strategy:
            raise NotImplementedError("video_elbo is bugged and not fully tested yet.")

        if "video_elbo" in strategy or ("global_elbo" in strategy and collect_elbo_only):
            print("Start collecting global ELBO ...")
            if strategy != "video_elbo" and strategy != "global_elbo":
                elbo_base = float(strategy[:3]) # [1.0, 0.5]
            else:
                elbo_base = 1.0
            for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
                # Input video for the current window.
                start, end = idx * temporal_window, (idx + 1) * temporal_window
                input_video = video[:, start:end, ...]

                # Spatio-temporally pad input_video so it's evenly divisible.
                padded_input_video, crop_region = pad_video_batch(input_video)
                input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
                self.collect_video_elbo(input_tensor, elbo_base)
            print("Collected ELBO: ", self.elbos[-1])
        
        if collect_elbo_only:
            return        

        for idx in tqdm(range(0, (num_frames - 1) // temporal_window + 1)):
            # Input video for the current window.
            start, end = idx * temporal_window, (idx + 1) * temporal_window
            input_video = video[:, start:end, ...]

            # Spatio-temporally pad input_video so it's evenly divisible.
            padded_input_video, crop_region = pad_video_batch(input_video)
            input_tensor = numpy2tensor(padded_input_video, dtype=self._dtype, device=self._device)
            # self.get_token_loss_curve(input_tensor)
            # assert False
            output_tensor, token_rate = self.autoencode(input_tensor, strategy=strategy, avg_rate=avg_rate)
            padded_output_video = tensor2numpy(output_tensor)
            output_video = unpad_video_batch(padded_output_video, crop_region)

            output_video_list.append(output_video)
        
        # Convert token_rate from torch.tensor to list of float if needed
        if isinstance(token_rate, torch.Tensor):
            token_rate = token_rate.detach().cpu().tolist()
        return np.concatenate(output_video_list, axis=1), token_rate
    
class AdaptiveVideoTokenizer(torch.nn.Module):
    # TODO: Implement the inference code of AdaptiveVideoTokenizer
    def __init__(self, **kwargs):
        super().__init__()
        pass

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        pass
