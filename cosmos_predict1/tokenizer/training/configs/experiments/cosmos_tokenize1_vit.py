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
MAX_NUM_TOKENS = (NUM_VIDEO_FRAMES - 1) // TEMPORAL_COMPRESSION * (CROP_HEIGHT // SPATIAL_COMPRESSION) ** 2 # 15 (if 121) * 16 * 16 = 3840, if 129, 16 * 16 * 16 = 4096
MIN_NUM_TOKENS = MAX_NUM_TOKENS // 16 # if MAX_NUM_TOKENS = 4096, MIN_NUM_TOKENS = 256

# ------------ ViT backbone config ------------

ADV4x8x8_256p_HDVILA_Posttrain: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "adaptive_discrete_video"},
            {"override /data_train": "hdvila_video256"},
            {"override /data_val": "hdvila_video256"},
            {"override /scheduler": "warmup_cosine"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=17,
            ),
            batch_size=1,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=17,
            ),
            batch_size=1,
        ),
        model=dict(
            config=dict(
                network=dict(
                    quantizer="CASFSQ",
                    num_quantizers=4,
                    patch_size=1,
                    legacy_mode=False,
                    temporal_compression=4,
                    spatial_compression=8,
                    num_video_frames=49,
                    # model specific parameters
                    crop_height=256,
                    vit_config=dict(
                        hidden_size=1024, # 768 for DEBUG, 4096 for full
                        intermediate_size=2048, # 768 for DEBUG, 4096 for full
                        num_encoder_layers=10, # 4 for DEBUG, 10 for full
                        num_decoder_layers=10, # 4 for DEBUG, 10 for full
                        num_attention_heads=32, # 16 for DEBUG, 32 for full
                        theta=10000.0,
                        rms_norm_eps=1e-5,
                        initializer_range=0.02,
                        switch_rotary_to_1d=0.5,
                        max_sequence_length=96,
                        max_sequence_length_1d=8192,
                        use_causal_decode_1d=True,
                        concat_decode_2d=True,
                    ),
                    # adaptive settings: IMPORTANT UPDATE
                    min_tokens_rate=0.06,
                    mean_tokens_rate=0.5, # Recover originally full training via mean_tokens_rate=1.0, static rate_strategy
                    rate_strategy="static", # ["static", "elbo"]
                )
            )
        ),
        job=dict(
            project="imagenet_posttraining",
            group="tokenizer",
            name="ADV4x8x8_256p_HDVILA_Posttrain",
        ),
        checkpoint=dict(
            strict_resume=True,
            load_training_state=True,
            jit=dict(input_shape=[1, 3, 1, 256, 256]),
        ),
        scheduler=dict(
            warmup_iters=5000,
            lr_decay_iters=100000,
            min_lr=1e-5,
        ),

    )
)


# -------------------------------------------------
# Hyperparameters of experiments on adaptive tokenization of ImageNet
# -------------------------------------------------


# ------------ ViT backbone config ------------

ADV8x16x16_256p_ImageNet_Posttrain: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/video_basic",
            {"override /network": "adaptive_discrete_video"},
            {"override /data_train": "imagenet_video256"},
            {"override /data_val": "imagenet_video256"},
            {"override /scheduler": "warmup_cosine"},
            "_self_",
        ],
        dataloader_train=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=1,
            ),
            batch_size=16,
        ),
        dataloader_val=dict(
            dataset=dict(
                crop_height=256,
                num_video_frames=1,
            ),
            batch_size=16,
        ),
        model=dict(
            config=dict(
                network=dict(
                    quantizer="CASFSQ",
                    num_quantizers=4,
                    patch_size=1,
                    legacy_mode=False,
                    temporal_compression=8,
                    spatial_compression=8,
                    num_video_frames=49,
                    # model specific parameters
                    crop_height=256,
                    vit_config=dict(
                        hidden_size=1024, # 768 for DEBUG, 4096 for full
                        intermediate_size=2048, # 768 for DEBUG, 4096 for full
                        num_encoder_layers=10, # 4 for DEBUG, 10 for full
                        num_decoder_layers=10, # 4 for DEBUG, 10 for full
                        num_attention_heads=32, # 16 for DEBUG, 32 for full
                        theta=10000.0,
                        rms_norm_eps=1e-5,
                        initializer_range=0.02,
                        switch_rotary_to_1d=0.5,
                        max_sequence_length=96,
                        max_sequence_length_1d=8192,
                        use_causal_decode_1d=True,
                        concat_decode_2d=True,
                    ),
                    # adaptive settings
                    min_tokens_rate=0.06,
                    mean_tokens_rate=0.5, # Recover originally full training via mean_tokens_rate=1.0, static rate_strategy
                    rate_strategy="uniform", # ["static", "elbo"]
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
            jit=dict(input_shape=[1, 3, 1, 256, 256]),
        ),
        scheduler=dict(
            warmup_iters=5000,
            lr_decay_iters=100000,
            min_lr=1e-5,
        ),

    )
)


cs = ConfigStore.instance()

for _item in [
    ADV8x16x16_256p_ImageNet_Posttrain, # Register this for ImageNet training (image-as-video)
    ADV4x8x8_256p_HDVILA_Posttrain, # Register this for HDVILA training (video), DEBUG used
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

