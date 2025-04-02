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

from hydra.core.config_store import ConfigStore

from cosmos_predict1.tokenizer.training.configs.experiments.utils import create_debug_job_with_mock_data
from cosmos_predict1.tokenizer.networks.configs import config as vi_config
from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict

# -------------------------------------------------
# Hyperparameters of experiments on adaptive tokenization
# -------------------------------------------------

NUM_VIDEO_FRAMES = 33 # 121 for full, smaller for DEBUG: this should be temporal_compression_rate * n + 1 (n=1,2,...)
CROP_HEIGHT = 256 # This indicates the largest width / height of the video
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 16
MAX_NUM_TOKENS = (NUM_VIDEO_FRAMES - 1) // TEMPORAL_COMPRESSION * (CROP_HEIGHT // SPATIAL_COMPRESSION) ** 2 # 15 (if 121) * 16 * 16 = 3840, if 129, 16 * 16 * 16 = 4096
MIN_NUM_TOKENS = MAX_NUM_TOKENS // 16 # if MAX_NUM_TOKENS = 4096, MIN_NUM_TOKENS = 256

# ------------ ViT backbone config ------------

vit_config = dict(
    hidden_size=768, # smaller for DEBUG, 4096 for full
    intermediate_size=768, # smaller for DEBUG, 11008 for full
    num_encoder_layers=4, # smaller for DEBUG, 16 for full
    num_decoder_layers=4, # smaller for DEBUG, 16 for full
    num_attention_heads=16, # smaller for DEBUG, 32 for full
    max_sequence_length=4096, # Not sure about this
    theta=10000.0,
    rms_norm_eps=1e-5,
    initializer_range=0.02
)

Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "adaptive_discrete_video"},
            {"override /data_train": "hdvila_video720"},
            {"override /data_val": "hdvila_video720"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=CROP_HEIGHT,
                num_video_frames=NUM_VIDEO_FRAMES, # smaller for DEBUG
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=CROP_HEIGHT,
                num_video_frames=NUM_VIDEO_FRAMES, # smaller for DEBUG
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    patch_size=TEMPORAL_COMPRESSION,
                    legacy_mode=False,
                    temporal_compression=TEMPORAL_COMPRESSION, # This should be exactly the same as patch_size to ensure the code is excutable
                    spatial_compression=SPATIAL_COMPRESSION, # This should be patch_size*n (n=1,2,...)
                    num_video_frames=NUM_VIDEO_FRAMES,
                    crop_height=CROP_HEIGHT,
                    config=vit_config,
                    min_tokens=MIN_NUM_TOKENS,
                    max_tokens=MAX_NUM_TOKENS, # TODO: Check the relationship with max_sequence_length
                    rate_strategy='uniform'
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA",
        ),
        checkpoint=dict(
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, NUM_VIDEO_FRAMES, CROP_HEIGHT, CROP_HEIGHT]),
        ),
    )
)

# Post-training config for Cosmos-Tokenize1-CV8x8x8-720p-HDVILA
Cosmos_Tokenize1_CV8x8x8_720p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "continuous_factorized_video"},
            {"override /data_train": "hdvila_video720"},
            {"override /data_val": "hdvila_video720"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=121,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=121,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=8,
                    spatial_compression=8,
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Cosmos-Tokenize1-CV8x8x8-720p-HDVILA",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenize1-CV8x8x8-720p/model.pt",
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, NUM_VIDEO_FRAMES, 256, 256]),
        ),
    )
)

# Post-training config for Cosmos-Tokenize1-DV8x16x16-720p-HDVILA
Cosmos_Tokenize1_DV8x16x16_720p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "discrete_factorized_video"},
            {"override /data_train": "hdvila_video720"},
            {"override /data_val": "hdvila_video720"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    persistent_quantizer=False,
                    z_channels=16,
                    channels_mult=[2, 4, 4],
                    patch_size=4,
                    legacy_mode=False,
                    temporal_compression=8,
                    spatial_compression=16,
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Cosmos-Tokenize1-DV8x16x16-720p-HDVILA",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenize1-DV8x16x16-720p/model.pt",
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, 17, 512, 512]),
        ),
    )
)

# Post-training config for Cosmos-Tokenize1-CV4x8x8-360p-HDVILA
Cosmos_Tokenize1_CV4x8x8_360p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "continuous_factorized_video"},
            {"override /data_train": "hdvila_video360"},
            {"override /data_val": "hdvila_video360"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    channels_mult=[2, 4, 4],
                    patch_size=2,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=8,
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Cosmos-Tokenize1-CV4x8x8-360p-HDVILA",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenize1-CV4x8x8-360p/model.pt",
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, 17, 512, 512]),
        ),
    )
)

# Post-training config for Cosmos-Tokenize1-DV4x8x8-360p-HDVILA
Cosmos_Tokenize1_DV4x8x8_360p_HDVILA: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "discrete_factorized_video"},
            {"override /data_train": "hdvila_video360"},
            {"override /data_val": "hdvila_video360"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=49,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    persistent_quantizer=False,
                    z_channels=256,
                    channels_mult=[2, 4, 4],
                    patch_size=2,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=8,
                )
            )
        ),
        job=dict(
            project="posttraining",
            group="tokenizer",
            name="Cosmos-Tokenize1-DV4x8x8-360p-HDVILA",
        ),
        checkpoint=dict(
            load_path="checkpoints/Cosmos-Tokenize1-DV4x8x8-360p/model.pt",
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, 17, 512, 512]),
        ),
    )
)

cs = ConfigStore.instance()

for _item in [
    Cosmos_Tokenize1_CV8x8x8_720p_HDVILA,
    Cosmos_Tokenize1_DV8x16x16_720p_HDVILA,
    Cosmos_Tokenize1_CV4x8x8_360p_HDVILA,
    Cosmos_Tokenize1_DV4x8x8_360p_HDVILA,
    Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA, # Add this for executable
]:
    experiment_name = [name for name, value in globals().items() if value is _item][0]

    log.info(f"Registering experiment: {experiment_name}")
    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )

    mock_experiment = f"mock_{experiment_name}"
    log.info(f"Registering mock experiment: {mock_experiment}")
    _debug_item = create_debug_job_with_mock_data(_item["job"]["name"])
    cs.store(
        group="experiment",
        package="_global_",
        name=mock_experiment,
        node=_debug_item,
    )
