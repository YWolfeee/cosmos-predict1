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
from loguru import logger as logging
from einops import rearrange
from transformers import PretrainedConfig

from cosmos_predict1.tokenizer.modules.patching import Patcher, Patcher3D, UnPatcher, UnPatcher3D
from cosmos_predict1.tokenizer.modules.utils import (
    CausalNormalize,
    batch2space,
    batch2time,
    cast_tuple,
    is_odd,
    nonlinearity,
    replication_pad,
    space2batch,
    time2batch,
)

_LEGACY_NUM_GROUPS = 32


########################################################
# Causal 3D CNN-based Tokenizer Layers
########################################################

class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in: int = 1,
        chan_out: int = 1,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        pad_mode: str = "constant",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)
        time_stride = kwargs.pop("time_stride", 1)
        time_dilation = kwargs.pop("time_dilation", 1)
        padding = kwargs.pop("padding", 1)

        self.pad_mode = pad_mode
        time_pad = time_dilation * (time_kernel_size - 1) + (1 - time_stride)
        self.time_pad = time_pad

        self.spatial_pad = (padding, padding, padding, padding)

        stride = (time_stride, stride, stride)
        dilation = (time_dilation, dilation, dilation)
        self.conv3d = nn.Conv3d(
            chan_in,
            chan_out,
            kernel_size,
            stride=stride,
            dilation=dilation,
            **kwargs,
        )

    def _replication_pad(self, x: torch.Tensor) -> torch.Tensor:
        x_prev = x[:, :, :1, ...].repeat(1, 1, self.time_pad, 1, 1)
        x = torch.cat([x_prev, x], dim=2)
        padding = self.spatial_pad + (0, 0)
        return F.pad(x, padding, mode=self.pad_mode, value=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._replication_pad(x)
        return self.conv3d(x)


class CausalUpsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
        time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()
        x = x.repeat_interleave(int(time_factor), dim=2)
        x = self.conv(x)
        return x[..., int(time_factor - 1) :, :, :]


class CausalDownsample3d(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = CausalConv3d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            time_stride=2,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1, 0, 0)
        x = F.pad(x, pad, mode="constant", value=0)
        x = replication_pad(x)
        x = self.conv(x)
        return x


class CausalHybridUpsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_up: bool = True,
        temporal_up: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=1, padding=0)
            if temporal_up
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=1, time_stride=1, padding=1)
            if spatial_up
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)
            if spatial_up or temporal_up
            else nn.Identity()
        )
        self.spatial_up = spatial_up
        self.temporal_up = temporal_up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_up and not self.temporal_up:
            return x

        # hybrid upsample temporally.
        if self.temporal_up:
            time_factor = 1.0 + 1.0 * (x.shape[2] > 1)
            if isinstance(time_factor, torch.Tensor):
                time_factor = time_factor.item()
            x = x.repeat_interleave(int(time_factor), dim=2)
            x = x[..., int(time_factor - 1) :, :, :]
            x = self.conv1(x) + x

        # hybrid upsample spatially.
        if self.spatial_up:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv2(x) + x

        # final 1x1x1 conv.
        x = self.conv3(x)
        return x


class CausalHybridDownsample3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_down: bool = True,
        temporal_down: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(1, 3, 3), stride=2, time_stride=1, padding=0)
            if spatial_down
            else nn.Identity()
        )
        self.conv2 = (
            CausalConv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, time_stride=2, padding=0)
            if temporal_down
            else nn.Identity()
        )
        self.conv3 = (
            CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, time_stride=1, padding=0)
            if spatial_down or temporal_down
            else nn.Identity()
        )

        self.spatial_down = spatial_down
        self.temporal_down = temporal_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.spatial_down and not self.temporal_down:
            return x

        # hybrid downsample spatially.
        if self.spatial_down:
            pad = (0, 1, 0, 1, 0, 0)
            x = F.pad(x, pad, mode="constant", value=0)
            x1 = self.conv1(x)
            x2 = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            x = x1 + x2

        # hybrid downsample temporally.
        if self.temporal_down:
            x = replication_pad(x)
            x1 = self.conv2(x)
            x2 = F.avg_pool3d(x, kernel_size=(2, 1, 1), stride=(2, 1, 1))
            x = x1 + x2

        # final 1x1x1 conv.
        x = self.conv3(x)
        return x


class CausalResnetBlock3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=num_groups)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalResnetBlockFactorized3d(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int = None,
        dropout: float,
        num_groups: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalNormalize(in_channels, num_groups=1)
        self.conv1 = nn.Sequential(
            CausalConv3d(
                in_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.norm2 = CausalNormalize(out_channels, num_groups=num_groups)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Sequential(
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(
                out_channels,
                out_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )
        self.nin_shortcut = (
            CausalConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        x = self.nin_shortcut(x)

        return x + h


class CausalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size = time2batch(q)
        k, batch_size = time2batch(k)
        v, batch_size = time2batch(v)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = batch2time(h_, batch_size)
        h_ = self.proj_out(h_)
        return x + h_


class CausalTemporalAttnBlock(nn.Module):
    def __init__(self, in_channels: int, num_groups: int) -> None:
        super().__init__()

        self.norm = CausalNormalize(in_channels, num_groups=num_groups)
        self.q = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = CausalConv3d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        q, batch_size, height = space2batch(q)
        k, _, _ = space2batch(k)
        v, _, _ = space2batch(v)

        bhw, c, t = q.shape
        q = q.permute(0, 2, 1)  # (bhw, t, c)
        k = k.permute(0, 2, 1)  # (bhw, t, c)
        v = v.permute(0, 2, 1)  # (bhw, t, c)

        w_ = torch.bmm(q, k.permute(0, 2, 1))  # (bhw, t, t)
        w_ = w_ * (int(c) ** (-0.5))

        # Apply causal mask
        mask = torch.tril(torch.ones_like(w_))
        w_ = w_.masked_fill(mask == 0, float("-inf"))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        h_ = torch.bmm(w_, v)  # (bhw, t, c)
        h_ = h_.permute(0, 2, 1).reshape(bhw, c, t)  # (bhw, c, t)

        h_ = batch2space(h_, batch_size, height)
        h_ = self.proj_out(h_)
        return x + h_


class EncoderBase(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher = Patcher(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        in_channels = in_channels * patch_size * patch_size

        # downsampling
        self.conv_in = CausalConv3d(in_channels, channels, kernel_size=3, stride=1, padding=1)

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)
        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = CausalDownsample3d(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(block_in, z_channels, kernel_size=3, stride=1, padding=1)

    def patcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.patcher(x)
        x = batch2time(x, batch_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher3d(x)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
            else:
                # temporal downsample (last level)
                time_factor = 1 + 1 * (hs[-1].shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                hs[-1] = replication_pad(hs[-1])
                hs.append(
                    F.avg_pool3d(
                        hs[-1],
                        kernel_size=[time_factor, 1, 1],
                        stride=[2, 1, 1],
                    )
                )

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DecoderBase(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher = UnPatcher(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        out_ch = out_channels * patch_size * patch_size

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.info("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # num of groups for GroupNorm, num_groups=1 for LayerNorm.
        num_groups = ignore_kwargs.get("num_groups", _LEGACY_NUM_GROUPS)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.mid.attn_1 = CausalAttnBlock(block_in, num_groups=num_groups)
        self.mid.block_2 = CausalResnetBlock3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=num_groups,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlock3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=num_groups,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(CausalAttnBlock(block_in, num_groups=num_groups))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = CausalUpsample3d(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=num_groups)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def unpatcher3d(self, x: torch.Tensor) -> torch.Tensor:
        x, batch_size = time2batch(x)
        x = self.unpatcher(x)
        x = batch2time(x, batch_size)

        return x

    def forward(self, z):
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
            else:
                # temporal upsample (last level)
                time_factor = 1.0 + 1.0 * (h.shape[2] > 1)
                if isinstance(time_factor, torch.Tensor):
                    time_factor = time_factor.item()
                h = h.repeat_interleave(int(time_factor), dim=2)
                h = h[..., int(time_factor - 1) :, :, :]

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        h = self.unpatcher3d(h)
        return h


class EncoderFactorized(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ) -> None:
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # Patcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.patcher3d = Patcher3D(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        in_channels = in_channels * patch_size * patch_size * patch_size

        # calculate the number of downsample operations
        self.num_spatial_downs = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert (
            self.num_spatial_downs <= self.num_resolutions
        ), f"Spatially downsample {self.num_resolutions} times at most"

        self.num_temporal_downs = int(math.log2(temporal_compression)) - int(math.log2(patch_size))
        assert (
            self.num_temporal_downs <= self.num_resolutions
        ), f"Temporally downsample {self.num_resolutions} times at most"

        # downsampling
        self.conv_in = nn.Sequential(
            CausalConv3d(
                in_channels,
                channels,
                kernel_size=(1, 3, 3),
                stride=1,
                padding=1,
            ),
            CausalConv3d(channels, channels, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        curr_res = resolution // patch_size
        in_ch_mult = (1,) + tuple(channels_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * in_ch_mult[i_level]
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                spatial_down = i_level < self.num_spatial_downs
                temporal_down = i_level < self.num_temporal_downs
                down.downsample = CausalHybridDownsample3d(
                    block_in,
                    spatial_down=spatial_down,
                    temporal_down=temporal_down,
                )
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, z_channels, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(
                z_channels,
                z_channels,
                kernel_size=(3, 1, 1),
                stride=1,
                padding=0,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape: [1, 3, 121, 144, 256], B, C, T, H, W
        x = self.patcher3d(x)
        # x.shape: [1, 192, 31, 36, 64] # H -> H / 4, W -> W / 4, T -> T / 4, C -> C * 64

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # hs[i].shape: 

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h.shape: [1, 512, 16, 18, 32], # Compared to the original: H, W -> H/8, W/8
        return h


class DecoderFactorized(nn.Module):
    def __init__(
        self,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        attn_resolutions: list[int],
        dropout: float,
        resolution: int,
        z_channels: int,
        spatial_compression: int = 16,
        temporal_compression: int = 8,
        **ignore_kwargs,
    ):
        super().__init__()
        self.num_resolutions = len(channels_mult)
        self.num_res_blocks = num_res_blocks

        # UnPatcher.
        patch_size = ignore_kwargs.get("patch_size", 1)
        self.unpatcher3d = UnPatcher3D(patch_size, ignore_kwargs.get("patch_method", "rearrange"))
        out_ch = out_channels * patch_size * patch_size * patch_size

        # calculate the number of upsample operations
        self.num_spatial_ups = int(math.log2(spatial_compression)) - int(math.log2(patch_size))
        assert self.num_spatial_ups <= self.num_resolutions, f"Spatially upsample {self.num_resolutions} times at most"
        self.num_temporal_ups = int(math.log2(temporal_compression)) - int(math.log2(patch_size))
        assert (
            self.num_temporal_ups <= self.num_resolutions
        ), f"Temporally upsample {self.num_resolutions} times at most"

        block_in = channels * channels_mult[self.num_resolutions - 1]
        curr_res = (resolution // patch_size) // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        logging.info("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = nn.Sequential(
            CausalConv3d(z_channels, block_in, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(block_in, block_in, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )
        self.mid.attn_1 = nn.Sequential(
            CausalAttnBlock(block_in, num_groups=1),
            CausalTemporalAttnBlock(block_in, num_groups=1),
        )
        self.mid.block_2 = CausalResnetBlockFactorized3d(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            num_groups=1,
        )

        legacy_mode = ignore_kwargs.get("legacy_mode", False)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = channels * channels_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    CausalResnetBlockFactorized3d(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        num_groups=1,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(
                        nn.Sequential(
                            CausalAttnBlock(block_in, num_groups=1),
                            CausalTemporalAttnBlock(block_in, num_groups=1),
                        )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                # The layer index for temporal/spatial downsampling performed
                # in the encoder should correspond to the layer index in
                # reverse order where upsampling is performed in the decoder.
                # If you've a pre-trained model, you can simply finetune.
                i_level_reverse = self.num_resolutions - i_level - 1
                if legacy_mode:
                    temporal_up = i_level_reverse < self.num_temporal_ups
                else:
                    temporal_up = 0 < i_level_reverse < self.num_temporal_ups + 1
                spatial_up = temporal_up or (
                    i_level_reverse < self.num_spatial_ups and self.num_spatial_ups > self.num_temporal_ups
                )
                up.upsample = CausalHybridUpsample3d(block_in, spatial_up=spatial_up, temporal_up=temporal_up)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = CausalNormalize(block_in, num_groups=1)
        self.conv_out = nn.Sequential(
            CausalConv3d(block_in, out_ch, kernel_size=(1, 3, 3), stride=1, padding=1),
            CausalConv3d(out_ch, out_ch, kernel_size=(3, 1, 1), stride=1, padding=0),
        )

    def forward(self, z):
        # z.shape: [1, 16, 16, 18, 32], From Encoder output C after Quantization: 512 -> 16
        h = self.conv_in(z)

        # middle block.
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # decoder blocks.
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)

        # h.shape: [1, 192, 31, 36, 64]
        h = self.unpatcher3d(h)
        # h.shape: [1, 3, 121, 144, 256]
        return h
    

########################################################
# Causal ViT-based Tokenizer Layers
########################################################

# ------------------------------------------------------------------
# ElasticTokConfig
# ------------------------------------------------------------------

class ViTConfig(Dict):
    model_type = "elastic_tok"

    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_encoder_layers: int = 16,
        num_decoder_layers: int = 16,
        num_attention_heads: int = 32,
        max_sequence_length: int = 4096,
        theta: float = 10000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        patch_size: Tuple[int, int, int] = (1, 8, 8),
        # Additional placeholders from JAX version
        mask_type: str = 'elastic',
        min_toks: int = 256,
        max_toks: int = 2048,
        frames_per_block: int = 1,
        lpips_loss_ratio: float = 0.1,
        bottleneck_type: str = 'fsq',
        fsq_quant_levels: Tuple[int, ...] = (8, 8, 8, 5, 5, 5),
        vae_bottleneck_dim: int = 8,
        scan_layers: bool = True,
        scan_attention: bool = False,
        # ... etc
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.max_sequence_length = max_sequence_length
        self.theta = theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.patch_size = patch_size
        self.bottleneck_type = bottleneck_type
        self.vae_bottleneck_dim = vae_bottleneck_dim
        self.mask_type = mask_type
        self.min_toks = min_toks
        self.max_toks = max_toks
        self.frames_per_block = frames_per_block
        self.lpips_loss_ratio = lpips_loss_ratio
        self.scan_layers = scan_layers
        self.scan_attention = scan_attention

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
    print(f"freqs_cis.shape={freqs_cis.shape}") # DEBUG
    print(f"bsz={bsz}, seq_len={seq_len}, n_heads={n_heads}, head_dim={head_dim}") # DEBUG
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
    def __init__(self, config: ViTConfig):
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
            position_ids = position_ids.unsqueeze(0).expand(B, S)

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
    def __init__(self, config: ViTConfig):
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
    def __init__(self, config: ViTConfig):
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
        self.config = kwargs.get('config', ViTConfig())

        # ---------- Patchify ----------
        self.patch_size = kwargs.get('patch_size', 8)
        assert self.patch_size <= spatial_compression
        assert spatial_compression % self.patch_size == 0, f"spatial_compression ({spatial_compression}) must be divisible by patch_size ({self.patch_size}) for proper unpatchification"
        assert self.patch_size == temporal_compression, f"patch_size ({self.patch_size}) must equal temporal_compression ({temporal_compression}) for proper unpatchification"
        # Currently Patcher3D does not support tuple patch_size
        self.patcher = Patcher3D(patch_size=self.patch_size) 
        # We need to further rearange the tokens to satisfy compression rate
        self.extra_spatial_compression = spatial_compression // self.patch_size
        self.extra_temporal_compression = temporal_compression // self.patch_size
        
        # ---------- Positional Embedding ----------
        self.crop_height = kwargs.get('crop_height', 256)
        self.num_video_frames = kwargs.get('num_video_frames', 121)
        self.is_kept_embed = nn.Parameter(torch.empty(1, self.config.hidden_size))
        self.is_masked_embed = nn.Parameter(torch.empty(1, self.config.hidden_size))
        nn.init.kaiming_normal_(self.is_kept_embed)
        nn.init.kaiming_normal_(self.is_masked_embed)

        # TODO: Add latent tokens & Learnable Positional Embedding
        # ---------- Latent Tokens ----------

        # ---------- Input Projection ----------
        patch_dim = in_channels * temporal_compression * (spatial_compression ** 2)
        self.input_proj = nn.Linear(patch_dim, self.config.hidden_size, bias=False)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.config.initializer_range)

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.config, 'num_encoder_layers')
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(num_layers)])
        
        # ---------- Output Projection ----------
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.output_proj = nn.Linear(self.config.hidden_size, z_channels, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, 
                x: torch.Tensor,         # [B, S, hidden_size]
                encoding_mask: torch.Tensor = None,  # [B, S] bool
                attention_mask: torch.Tensor = None, # [B, S] float
                position_ids: Optional[torch.Tensor] = None,
                cache: Optional[Dict[str, torch.Tensor]] = None,
                training: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Input shape: (B, _C, _T, _H, _W), prepatchified
        # _C = 3, _T = self.num_video_frames 
        # _H = self.crop_height * h, _W = self.crop_height * w, h, w <= 1
        B = x.shape[0]

        # ---------- 1.Patchify to 1D ---------- 
        x = self.patcher(x)  # (B, _C * patch_size**3, _T / patch_size, _H / patch_size, _W / patch_size)
        ### Further rearange the token to satisfy compression rate
        if self.extra_spatial_compression > 1 or self.extra_temporal_compression > 1:
            x = rearrange(
                x,
                "b c (t p1) (h p2) (w p3) -> b (c p1 p2 p3) t h w",
                p1=self.extra_temporal_compression,
                p2=self.extra_spatial_compression,
                p3=self.extra_spatial_compression,
            ).contiguous() 
        # x: (B, _C * (sc ** 2) * tc, _T / tc, _H / sc, _W / sc), postpatchified
        # sc: spatial_compression; tc: temporal_compression
        B, C, T, H, W = x.shape
        ### Flatten the spatial and temporal dimensions
        x = x.reshape(B, C, -1) # (B, C, T * H * W)
        x = x.permute(0, 2, 1) # (B, T * H * W, C)
        
        # ---------- 2. Input Projection ----------
        x = self.input_proj(x) # (B, T * H * W, D), D = self.config.hidden_size

        # TODO: Add forward with latent tokens
        
        # ---------- 3. Add Positional Embedding ----------
        if encoding_mask is not None:
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
            )  # Maintains shape # (B, T * H * W, D)

        # TODO: Extract latent tokens
        
        # ---------- 5. Output Projection ----------
        x = self.norm(x)  # (B, T * H * W, D)
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
        self.config = kwargs.get('config', ViTConfig())

        # ---------- Input Projection ----------
        self.input_proj = nn.Linear(z_channels, self.config.hidden_size, bias=False)
        nn.init.normal_(self.input_proj.weight, mean=0.0, std=self.config.initializer_range)

        # ---------- Positional Embedding ----------
        self.crop_height = kwargs.get('crop_height', 256)
        self.num_video_frames = kwargs.get('num_video_frames', 121)
        self.is_kept_embed = nn.Parameter(torch.empty(1, self.config.hidden_size))
        self.is_masked_embed = nn.Parameter(torch.empty(1, self.config.hidden_size))
        nn.init.kaiming_normal_(self.is_kept_embed)
        nn.init.kaiming_normal_(self.is_masked_embed)

        # TODO: Add masked tokens & Learnable Positional Embedding

        # ---------- Transformer Blocks ----------
        num_layers = getattr(self.config, 'num_encoder_layers')
        self.blocks = nn.ModuleList([TransformerBlock(self.config) for _ in range(num_layers)])
        
        # ---------- Output Projection ----------
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        out_channels = out_channels * (spatial_compression ** 2) * temporal_compression
        self.output_proj = nn.Linear(self.config.hidden_size, out_channels, bias=False)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=self.config.initializer_range)

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
            encoding_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.Tensor] = None,
            cache: Optional[Dict[str, torch.Tensor]] = None,
            training: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Input shape: (B, z_channels, T, H, W)
        B, C, T, H, W = x.shape
        
        # ---------- 1. Reshape to 1D ----------
        x = x.reshape(B, C, -1)  # (B, z_channels, T*H*W)
        x = x.permute(0, 2, 1)  # (B, T*H*W, z_channels)
        
        # ---------- 2. Input Projection ----------
        x = self.input_proj(x)  # (B, T*H*W, hidden_size)
        
        # ---------- 3. Add Positional Embedding ----------
        if encoding_mask is not None:
            keep_embed = self.is_kept_embed.unsqueeze(1)  # (1, 1, hidden_size)
            mask_embed = self.is_masked_embed.unsqueeze(1)  # (1, 1, hidden_size)
            x = x + torch.where(encoding_mask.unsqueeze(-1), keep_embed, mask_embed)

        # TODO: Add forward concatenated with latent tokens
        
        # ---------- 4. Transformer Forward ----------
        for blk in self.blocks:
            x = blk(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache=cache
            )  # Maintains shape (B, T*H*W, hidden_size)
        
        # ---------- 5. Output Projection ----------
        x = self.norm(x)  # (B, T*H*W, hidden_size)
        x = self.output_proj(x)  # (B, T*H*W, out_channels*patch_size^3)

        # TODO: Extract masked tokens
        
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