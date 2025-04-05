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

"""Implementations of dataset settings and augmentations for tokenization

Run this command to interactively debug:
python3 -m cosmos_predict1.tokenizer.training.datasets.dataset_provider

"""

from cosmos_predict1.tokenizer.training.datasets.augmentation_provider import (
    video_train_augmentations,
    video_val_augmentations,
)
from cosmos_predict1.tokenizer.training.datasets.utils import categorize_aspect_and_store
from cosmos_predict1.tokenizer.training.datasets.video_dataset import Dataset
from cosmos_predict1.utils.lazy_config import instantiate

_VIDEO_PATTERN_DICT = {
    "hdvila_video": "datasets/hdvila/videos/*.mp4", # Use videos_org from hdvila in HuggingFace
    "imagenet_video": "datasets/imagenet/train/*.jpg", # Use image as video
    "imagenet_val_video": "datasets/imagenet/val/*.jpg", # Use image as video
}


def apply_augmentations(data_dict, augmentations_dict):
    """
    Loop over each LazyCall object and apply it to data_dict in place.
    """
    for aug_name, lazy_aug in augmentations_dict.items():
        aug_instance = instantiate(lazy_aug)
        data_dict = aug_instance(data_dict)
    return data_dict


class AugmentDataset(Dataset):
    def __init__(self, base_dataset, augmentations_dict):
        """
        base_dataset: the video dataset instance
        augmentations_dict: the dictionary returned by
                            video_train_augmentations() or video_val_augmentations()
        """
        self.base_dataset = base_dataset

        # Pre-instantiate every augmentation ONCE:
        self.augmentations = []
        for aug_name, lazy_aug in augmentations_dict.items():
            aug_instance = instantiate(lazy_aug)  # build the actual augmentation
            self.augmentations.append((aug_name, aug_instance))

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get the raw sample from the base dataset
        data = self.base_dataset[index]
        data = categorize_aspect_and_store(data)

        # Apply each pre-instantiated augmentation
        for aug_name, aug_instance in self.augmentations:
            data = aug_instance(data)

        return data


def dataset_entry(
    dataset_name: str,
    dataset_type: str,
    is_train: bool = True,
    resolution="720",
    crop_height=256,
    num_video_frames=25,
) -> AugmentDataset:
    if dataset_type != "video":
        raise ValueError(f"Dataset type {dataset_type} is not supported")

    # Instantiate the video dataset
    base_dataset = Dataset(
        video_pattern=_VIDEO_PATTERN_DICT[dataset_name.lower()],
        num_video_frames=num_video_frames,
    )

    # Pick the training or validation augmentations
    if is_train:
        aug_dict = video_train_augmentations(
            input_keys=["video"],  # adjust if necessary
            resolution=resolution,
            crop_height=crop_height,
        )
    else:
        aug_dict = video_val_augmentations(
            input_keys=["video"],
            resolution=resolution,
            crop_height=crop_height,
        )

    # Wrap the dataset with the augmentations
    return AugmentDataset(base_dataset, aug_dict)


if __name__ == "__main__":
    import os
    from PIL import Image
    import torch
    import numpy as np
    
    # Example usage / quick test
    dataset = dataset_entry(
        dataset_name="imagenet_video",
        dataset_type="video",
        is_train=False,
        resolution="256",
        crop_height=256,
        num_video_frames=1,
    )

    # Print out some basic info:
    print(f"Total samples in dataset: {len(dataset)}")

    # Create output directory for saving images
    output_dir = "dataset_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize the first 8 samples
    num_samples = min(8, len(dataset))
    
    for i in range(num_samples):
        sample = dataset[i]
        print(f"Sample index {i} keys: {list(sample.keys())}")
        
        if "video" in sample:
            video_tensor = sample["video"]
            print(f"Video shape: {video_tensor.shape}")
            
            # For single frame "videos", shape is [C, T, H, W] where T=1
            # Convert to PIL image: [C, T, H, W] -> [H, W, C]
            if video_tensor.shape[1] == 1:  # T dimension = 1
                # Extract the single frame and rearrange dimensions
                image_tensor = video_tensor[:, 0, :, :]  # [C, H, W]
                image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                image_np = (image_np + 1) / 2  # Normalize to [0,1]
                
                # Convert from float [0,1] to uint8 [0,255] if needed
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                
                # Create and save PIL image
                image = Image.fromarray(image_np)
                image_path = os.path.join(output_dir, f"sample_{i}.png")
                image.save(image_path)
                print(f"Saved image to {image_path}")
        
        if "video_name" in sample:
            print(f"Video metadata: {sample['video_name']}")
        
        print("---")
    
    print(f"Saved {num_samples} sample images to {output_dir}")
