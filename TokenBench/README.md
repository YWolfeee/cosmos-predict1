## Installation

Notice that this is TokenBench environment, different from cosmos-predict1. Suppose you are currently in ```cosmos-predict1/```.
```
cd TokenBench
conda create -n TokenBench python=3.10
conda activate TokenBench
pip3 install -r requirements_updated.txt
apt-get install -y ffmpeg
```

## Configuration and File Structure

The file structure looks like:

```
├── cosmos-predict1
    ├── checkpoints
        ├── model_name
            ├── iter_000080000.pt
    ├── TokenBench
        ├── pretrained_ckpts
            ├── LanguageBind
                ├── Open-Sora-Plan-v1.0.0
                    ├── ...
        ├── env.sh
        ├── eval.sh
        ├── ...
├── DATASET_DIR
    ├── GT_VIDEO_DIR
        ├── xxx.mp4
        ├── ...
    ├── GT_VIDEO_CLIPS_DIR
        ├── xxx_0.mp4
        ├── xxx_1.mp4
        ├── yyy_0.mp4
        ├── ...
    ├── OUTPUT_VIDEO_CLIPS_DIR
        ├── model_name
            ├── iter00080000_static0.75
                ├── xxx.mp4
                ├── ...
            ├── ...
        ├── ...
```

Remember to configure **env.sh**, in terms of model setup:

| Variable | Description |
|----------|-------------|
| ```model_name``` | This is the folder include the checkpoint, e.g., "mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_concat2d_layer88_static_nofreeze". |
| ```pt_name``` | The checkpoint name of the specific model settings, e.g., "iter_000080000". |
| ```strategy``` | Inference strategy. Choose from ["static", "causal_choice", "0.9elbo", "elbo", "global_elbo", "prior"]. Notice that "video_elbo" currently is buggy, **DO NOT USE IT**.|
| ```avg_rate``` | The average rate of token usage during inference. |
| ```tokenizer_type``` | Choose from ["CV8x8x8-720p", "DV8x16x16-720p", "CV4x8x8-360p", "DV4x8x8-360p", "OURS4x8x8-256p", "OURS4x8x8-256p-88", "OURS4x8x8-concat-256p-88", "OURS4x8x8-special-256p-88"]. **The tokenizer must match the proper settings to make it work**. Details can be checked in ```cosmos-predict1/cosmos_predict1/tokenizer/networks/__init__.py```.|

Details of the ```strategy``` and ```avg_rate``` can be found in **cosmos_predict1/tokenizer/inference/video_lib.py**.

In terms of dataset input and output setup:

| Variable | Description |
|----------|-------------|
| ```DATASET_DIR``` | The root folder of all saved data. |
| ```GT_VIDEO_DIR``` | The folder includes all ground truth videos. |
| ```GT_VIDEO_CLIPS_DIR``` | The folder includes all separated clips of each ground truth videos. |
| ```OUTPUT_VIDEO_CLIPS_DIR``` | All reconstructed video clips will be saved here. |
| ```CHECKPOINT_DIR``` | Checkpoint Folder. |

## Download StyleGAN Checkpoints For FVD Evaluation

Remember to replace ```HF_TOKEN``` in the script with your huggingface token.
```
python util_scripts/download_stylegan.py
```
## Access to TokenBench

Access TokenBench data through [this OneDrive link](https://office365stanford-my.sharepoint.com/personal/haotiany_stanford_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhaotiany%5Fstanford%5Fedu%2FDocuments%2FPHD%2F2025Spring%2Ftokenizer%5Fckpt&ga=1). Download **tokenbench_v1_fps_30_240p.tar.gz**. Uncompress it, this will include 500 videos in the resolution of 256 x 416. Save these files to ```DATASET_DIR/GT_VIDEO_DIR```.

## Reconstruction on the TokenBench


### Download the model

Access model through [this OneDrive link](https://office365stanford-my.sharepoint.com/personal/haotiany_stanford_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fhaotiany%5Fstanford%5Fedu%2FDocuments%2FPHD%2F2025Spring%2Ftokenizer%5Fckpt&ga=1). The checkpoint is at, for example: **mock_DV4x8x8_t49_256p_c16_merge_DV_haotian_finetune_from_DV_concat2d_layer88_static_nofreeze/iter_000080000.pt**. Preserve this structure and save it under your ```CHECKPOINT_DIR```.

### Switch to Cosmos working environment

The details for environment setup can be found in cosmos-predcit1/REAMD.md.

```
cd ~/cosmos-predict1
conda activate cosmos
```

### Seperate the videos into video clips

```
bash exp_scripts/reconstruct_clip.sh
```

This will seperate the videos into clips and save in ```GT_VIDEO_CLIPS_DIR```. If only using square clip, add ```--only_square_clips``` in the bash file.

### Test on subset

Go to your ```DATASET_DIR```, use following script to extract subset:

```
import os
import random
import shutil
from pathlib import Path

source_dir = "BASE_CLIP_DIR"
target_dir = "SUBSET_CLIP_DIR"
rate = 0.1

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Get all files from source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Randomly select (100*rate)% of the files
sample_size = max(1, int(len(files) * rate))
selected_files = random.sample(files, sample_size)

# Copy selected files to target directory
for file in selected_files:
    source_path = os.path.join(source_dir, file)
    target_path = os.path.join(target_dir, file)
    shutil.copy2(source_path, target_path)
    print(f"Copied {file} to {target_dir}")

print(f"Copied {len(selected_files)} files ({sample_size/len(files)*100:.2f}% of total)")
```

If you are using the subset, remember to replace ```GT_VIDEO_CLIPS_DIR``` in your ```env.sh```.

### Reconstruct the video clips

```
bash exp_scripts/reconstruct_tokenbench.sh
```

The reconstructed video will be saved in ```OUTPUT_VIDEO_CLIPS_DIR```.

## Evaluation on the TokenBench

Remember to switch back to TokenBench environment and project directory, replace the argument accordingly.
```
cd TokenBench
conda activate TokenBench
bash eval.sh
```

Alternatively, if you are evaluating other results such as those from ElasticTok, modify the ```target_path``` in the following:

```
bash eval_elastic.sh
```

## Visualize improved results

Remember to edit the arguments listed in ```main()```, running this will create a folder including videos where each includes 3 sub-clip, from left to right: ground truth, result_1, result_2.
```
python util_scripts/pick_video.py
```