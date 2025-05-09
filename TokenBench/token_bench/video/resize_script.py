# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import imageio
import numpy as np
from glob import glob
import mediapy as media


def resize_video(video: np.ndarray, short_size: int = None) -> np.ndarray:
    """Resizes a video to have the short side of `short_size`."""
    if short_size is None:
        return video
    height, width = video.shape[-3:-1]
    if height <= width:
        height_new, width_new = short_size, int(width * short_size / height + 0.5)
    else:
        height_new, width_new = int(height * short_size / width + 0.5), short_size
    return media.resize_video(video, shape=(height_new, width_new))


# Try to import ffmpeg backend for imageio
try:
    import imageio_ffmpeg
    print("Using imageio_ffmpeg backend")
except ImportError:
    print("Warning: imageio_ffmpeg not found. Attempting to install...")
    try:
        import subprocess
        subprocess.check_call(["pip", "install", "imageio[ffmpeg]"])
        print("Successfully installed imageio[ffmpeg]")
    except Exception as e:
        print(f"Failed to install imageio[ffmpeg]: {e}")
        print("Please manually install with: pip install imageio[ffmpeg]")

aspect = 256
num_convert = 10000
# input_dir = "/mnt/rdata8/qy_dataset/TokenBench/tokenbench"
# output_dir = f"/mnt/rdata8/qy_dataset/TokenBench/tokenbench_{aspect}p"

input_dir = "/mnt/rdata8/qy_dataset/TokenBench/subset_square_v1_fps_30_240p_clip"
output_dir = f"/mnt/rdata8/qy_dataset/TokenBench/subset_square_v1_fps_24_240p_clip"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all mp4 files in the input directory
input_files = sorted(glob(os.path.join(input_dir, "*.mp4")))
print(f"Found {len(input_files)} videos to process")

for idx, video_file in enumerate(input_files):
    if idx >= num_convert:
        break
    print(f"Processing video {idx+1}/{len(input_files)}: {video_file}")
    
    try:
        # Read the video using mediapy instead of imageio directly
        input_video = media.read_video(video_file)
        video_fps = 24  # Default to 30 fps if not available
        
        # # Try to get the actual fps from mediapy if possible
        # try:
        #     metadata = media.read_video_metadata(video_file)
        #     if 'fps' in metadata:
        #         video_fps = metadata['fps']
        # except:
        #     print(f"Warning: Could not read fps from video, using default {video_fps}")
        
        T, H, W, C = input_video.shape
        print(f"Original dimensions: {(T, H, W, C)}")
        
        # Resize the video to aspect
        output_video = resize_video(input_video, aspect)
        print(f"Resized dimensions: {output_video.shape}")
        
        # Create output filename
        output_filename = os.path.join(output_dir, os.path.basename(video_file))
        print(f"Writing to: {output_filename}")
        
        # Write the resized video
        media.write_video(output_filename, output_video, fps=video_fps)
        
        # Free memory
        del input_video
        del output_video
    
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
        print("Skipping to next video...")
        continue

print(f"All videos have been resized to {aspect}p and saved to the output directory")
