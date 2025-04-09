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

from cosmos_predict1.utils import log
from cosmos_predict1.utils.lazy_config import LazyDict
from cosmos_predict1.tokenizer.training.configs.experiments.utils import create_debug_job_with_mock_data

# -------------------------------------------------
# Hyperparameters of experiments on adaptive tokenization of HDVILA videos
# -------------------------------------------------

NUM_VIDEO_FRAMES = 121 # this should be temporal_compression_rate * n + 1 (n=1,2,...)
CROP_HEIGHT = 256 # This indicates the largest width / height of the video
TEMPORAL_COMPRESSION = 8
SPATIAL_COMPRESSION = 16
num_temporal_patches = (NUM_VIDEO_FRAMES + TEMPORAL_COMPRESSION - 1) // TEMPORAL_COMPRESSION
num_spatial_patches = CROP_HEIGHT // SPATIAL_COMPRESSION
num_patch_tokens = num_temporal_patches * num_spatial_patches ** 2
num_latent_tokens = num_patch_tokens
MIN_TOKENS_RATIO = 1/16

# ------------ ViT backbone config ------------


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
                num_video_frames=NUM_VIDEO_FRAMES,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=CROP_HEIGHT,
                num_video_frames=NUM_VIDEO_FRAMES,
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
                    vit_config=dict(
                        hidden_size=1024,
                        intermediate_size=2048,
                        num_encoder_layers=10,
                        num_decoder_layers=10,
                        num_attention_heads=16,
                        max_sequence_length=4096,
                        theta=10000.0,
                        rms_norm_eps=1e-5,
                        initializer_range=0.02,
                    ),
                    min_tokens_ratio=MIN_TOKENS_RATIO,
                    num_patch_tokens=num_patch_tokens,
                    num_latent_tokens=num_latent_tokens,
                    rate_strategy='uniform',
                    use_latent_tokens=False,
                    quantizer="CASFSQ",
                    num_quantizers=4,
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

# -------------------------------------------------
# Hyperparameters of experiments on adaptive tokenization of ImageNet
# -------------------------------------------------

NUM_VIDEO_FRAMES = 1 # Image as 1-Frame Video Training
CROP_HEIGHT = 256 # This indicates the largest width / height of the video
BATCH_SIZE = 32
TEMPORAL_COMPRESSION = 1
SPATIAL_COMPRESSION = 16
num_temporal_patches = (NUM_VIDEO_FRAMES + TEMPORAL_COMPRESSION - 1) // TEMPORAL_COMPRESSION
num_spatial_patches = CROP_HEIGHT // SPATIAL_COMPRESSION
num_patch_tokens = num_temporal_patches * num_spatial_patches ** 2
num_latent_tokens = num_patch_tokens
MIN_TOKENS_RATIO = 1 / 16

# ------------ ViT backbone config ------------

ADV8x16x16_256p_ImageNet_Posttrain: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "adaptive_discrete_video"},
            {"override /data_train": "imagenet_video256"},
            {"override /data_val": "imagenet_video256"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=CROP_HEIGHT,
                num_video_frames=NUM_VIDEO_FRAMES,
            ),
            batch_size=BATCH_SIZE,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=CROP_HEIGHT,
                num_video_frames=NUM_VIDEO_FRAMES,
            ),
            batch_size=BATCH_SIZE,
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
                    vit_config=dict(
                        hidden_size=1024,
                        intermediate_size=2048,
                        num_encoder_layers=10,
                        num_decoder_layers=10,
                        num_attention_heads=16,
                        max_sequence_length=4096,
                        theta=10000.0,
                        rms_norm_eps=1e-5,
                        initializer_range=0.02,
                    ),
                    min_tokens_ratio=MIN_TOKENS_RATIO,
                    num_patch_tokens=num_patch_tokens,
                    num_latent_tokens=num_latent_tokens,
                    rate_strategy='uniform',
                    use_latent_tokens=False,
                    quantizer="CASFSQ",
                    num_quantizers=4,
                )
            )
        ),
        job=dict(
            project="imagenet_posttraining",
            group="tokenizer",
            name="ADV8x16x16_256p_ImageNet_Posttrain",
        ),
        checkpoint=dict(
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, NUM_VIDEO_FRAMES, CROP_HEIGHT, CROP_HEIGHT]),
        ),
    )
)


cs = ConfigStore.instance()

for _item in [
    Adaptive_Tokenize1_ADV8x16x16_720p_HDVILA, # Register this for post-train verification
    ADV8x16x16_256p_ImageNet_Posttrain, # Register this for ImageNet training (image-as-video)
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
