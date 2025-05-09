import os
import csv
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import subprocess
from pathlib import Path

def read_metric_csv(csv_path, lower_is_better=False):
    """Read metric CSV file and return a dictionary mapping filepath to metric value."""
    metric_dict = {}
    with open(csv_path, 'r') as f:
        content = f.read()
        # Parse the array structure
        if content.strip():
            try:
                # The file contains an array of arrays: [["filepath", value], ["filepath", value], ...]
                data = eval(content)
                for item in data:
                    if len(item) == 2:
                        filepath = item[0]
                        value = float(item[1])
                        metric_dict[filepath] = value
            except Exception as e:
                print(f"Error parsing CSV content: {e}")
    return metric_dict

def read_token_rate_csv(csv_path):
    """Read token rate CSV file and return a dictionary mapping filepath to token rate values."""
    token_rate_dict = {}
    with open(csv_path, 'r') as f:
        content = f.read()
        # Parse the array structure
        if content.strip():
            try:
                # The file contains an array of arrays: [["filepath", [rate_1, rate_2, ...]], ["filepath", [rate_1, rate_2, ...]], ...]
                data = eval(content)
                for item in data:
                    if len(item) == 2:
                        filepath = item[0]
                        token_rates = item[1]
                        # Ensure token_rates is a list
                        if not isinstance(token_rates, list):
                            token_rates = [token_rates]
                        token_rate_dict[filepath] = token_rates
            except Exception as e:
                print(f"Error parsing CSV content: {e}")
    return token_rate_dict

def find_better_videos(folder1, folder2, gt_folder, output_dir):
    """Find videos where metrics in folder1 are better than folder2, and create comparison videos."""
    # Read metric values from both folders
    psnr_dict1 = read_metric_csv(os.path.join(folder1, 'psnr.csv'))
    psnr_dict2 = read_metric_csv(os.path.join(folder2, 'psnr.csv'))
    
    ssim_dict1 = read_metric_csv(os.path.join(folder1, 'ssim.csv'))
    ssim_dict2 = read_metric_csv(os.path.join(folder2, 'ssim.csv'))
    
    lpips_dict1 = read_metric_csv(os.path.join(folder1, 'lpips.csv'), lower_is_better=True)
    lpips_dict2 = read_metric_csv(os.path.join(folder2, 'lpips.csv'), lower_is_better=True)
    
    # Read token rate values from folder1
    token_rate_dict = read_token_rate_csv(os.path.join(folder1, 'token_rate.csv'))
    
    # Find common filepaths where folder1 has better metrics
    better_filepaths = []
    for filepath in psnr_dict1:
        if filepath in psnr_dict2 and filepath in ssim_dict1 and filepath in ssim_dict2 and filepath in lpips_dict1 and filepath in lpips_dict2:
            # Check if folder1 has higher PSNR and SSIM (higher is better) and lower LPIPS (lower is better)
            if (
                psnr_dict1[filepath] > psnr_dict2[filepath] 
                and ssim_dict1[filepath] > ssim_dict2[filepath]
                and lpips_dict1[filepath] < lpips_dict2[filepath]
                ):
                better_filepaths.append(filepath)
    
    print(f"Found {len(better_filepaths)} videos where {os.path.basename(folder1)} has better metrics than {os.path.basename(folder2)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory if it doesn't exist
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Process each better video
    for filepath in tqdm(better_filepaths):
        # Get the full paths to the videos
        video1_path = os.path.join(folder1, filepath)
        video2_path = os.path.join(folder2, filepath)
        gt_video_path = os.path.join(gt_folder, filepath)
        
        # Check if all videos exist
        if not (os.path.exists(video1_path) and os.path.exists(video2_path) and os.path.exists(gt_video_path)):
            print(f"Skipping {filepath} - one or more videos not found")
            continue
        
        # Get token rates for folder1 video (or default to [0.5] if not found)
        token_rates = token_rate_dict.get(filepath, [0.5])
        
        # If we only have a single token rate, create a varying pattern to simulate change
        if len(token_rates) == 1:
            # Create a varying token rate pattern (e.g., sinusoidal variation)
            base_rate = token_rates[0]
            num_frames = get_video_frame_count(video1_path)
            token_rates = generate_varying_token_rates(base_rate, num_frames)
        
        print(filepath, ":", token_rates)
        # Create output filename
        output_filename = os.path.join(output_dir, f"comparison_{Path(filepath).stem}.mp4")
        
        # Concatenate videos side by side (GT, video1, video2)
        concatenate_videos(gt_video_path, video1_path, video2_path, output_filename, 
                          "Ground Truth", 
                          token_rates, 
                          "100% tokens")
        
        # Save metric information
        with open(os.path.join(output_dir, "metrics_info.txt"), "a") as f:
            f.write(f"{filepath}:\n")
            f.write(f"  PSNR: {os.path.basename(folder1)}={psnr_dict1[filepath]:.2f}, {os.path.basename(folder2)}={psnr_dict2[filepath]:.2f}, diff={psnr_dict1[filepath]-psnr_dict2[filepath]:.2f}\n")
            f.write(f"  SSIM: {os.path.basename(folder1)}={ssim_dict1[filepath]:.4f}, {os.path.basename(folder2)}={ssim_dict2[filepath]:.4f}, diff={ssim_dict1[filepath]-ssim_dict2[filepath]:.4f}\n")
            f.write(f"  LPIPS: {os.path.basename(folder1)}={lpips_dict1[filepath]:.4f}, {os.path.basename(folder2)}={lpips_dict2[filepath]:.4f}, diff={lpips_dict2[filepath]-lpips_dict1[filepath]:.4f}\n\n")

def get_video_frame_count(video_path):
    """Get the total number of frames in a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def generate_varying_token_rates(base_rate, num_frames):
    """Generate varying token rates for a video with the given number of frames."""
    # Create token rates that change at regular intervals
    token_rates = []
    
    # Define different token rate levels to cycle through
    rate_levels = [0.3, 0.5, 0.7, 0.9]
    
    # Calculate how many frames each rate should be applied to
    # For the first n-1 rates, they will have the same number of frames
    frames_per_rate = num_frames // len(rate_levels)
    
    # Generate the token rates
    for i in range(num_frames):
        if i < frames_per_rate * (len(rate_levels) - 1):
            # For the first n-1 rates, distribute evenly
            rate_index = i // frames_per_rate
            rate = rate_levels[rate_index]
        else:
            # The last rate will hold all remaining frames
            rate = rate_levels[-1]
        token_rates.append(rate)
    
    return token_rates

def concatenate_videos(gt_video_path, video1_path, video2_path, output_path, gt_title, token_rates, video2_title):
    """Concatenate three videos side by side."""
    # Open all three videos
    cap_gt = cv2.VideoCapture(gt_video_path)
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    # Check if videos opened successfully
    if not (cap_gt.isOpened() and cap1.isOpened() and cap2.isOpened()):
        print(f"Error opening videos: {gt_video_path}, {video1_path}, {video2_path}")
        return
    
    # Get video properties
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = os.path.join("temp", os.path.basename(output_path))
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width*3, height))
    
    # Distribute token rates across frames
    frame_token_rates = []
    if len(token_rates) > 0:
        frames_per_rate = total_frames // len(token_rates)
        for i, rate in enumerate(token_rates):
            if i < len(token_rates) - 1:
                # For all rates except the last one
                frame_token_rates.extend([rate] * frames_per_rate)
            else:
                # For the last rate, fill all remaining frames
                remaining_frames = total_frames - len(frame_token_rates)
                frame_token_rates.extend([rate] * remaining_frames)
    else:
        # Default to 50% if no token rates provided
        frame_token_rates = [0.5] * total_frames
    
    frame_count = 0
    while True:
        # Read frames from all videos
        ret_gt, frame_gt = cap_gt.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        # Break if any video ends
        if not (ret_gt and ret1 and ret2):
            break
        
        # Resize frames to ensure they're all the same size
        frame_gt = cv2.resize(frame_gt, (width, height))
        frame1 = cv2.resize(frame1, (width, height))
        frame2 = cv2.resize(frame2, (width, height))
        
        # Get current token rate for this frame
        if frame_count < len(frame_token_rates):
            current_rate = frame_token_rates[frame_count]
        else:
            # If we somehow have more frames than token rates, use the last rate
            current_rate = frame_token_rates[-1]
        
        # Format token rate as percentage
        current_rate = current_rate - 0.25
        token_rate_percent = int(current_rate * 100)
        video1_title = f"{token_rate_percent}% tokens"
        
        # Add title at the top of each video
        cv2.putText(frame_gt, gt_title, (width//2 - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame1, video1_title, (width//2 - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Blue text
        cv2.putText(frame2, video2_title, (width//2 - 80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Blue text
        
        # Concatenate frames horizontally
        combined_frame = np.hstack((frame_gt, frame1, frame2))
        
        # Write the frame
        out.write(combined_frame)
        
        # Increment frame counter
        frame_count += 1
    
    # Release resources
    cap_gt.release()
    cap1.release()
    cap2.release()
    out.release()
    
    # Convert to H.264 for better compatibility
    subprocess.run([
        "ffmpeg", "-i", temp_output, 
        "-c:v", "libx264", "-crf", "23", 
        "-preset", "medium", output_path, 
        "-y", "-loglevel", "error"
    ])

def main():
    parser = argparse.ArgumentParser(description="Compare videos based on multiple metrics")
    parser.add_argument("--folder1", default="DIRPATH_OF_THE_OUTPUT_ADAPTIVE_MODEL", help="First folder with metric CSVs (better metrics)")
    parser.add_argument("--folder2", default="DIRPATH_OF_THE_OUTPUT_STATIC_MODEL", help="Second folder with metric CSVs (worse metrics)")
    parser.add_argument("--gt_folder", default="DIRPATH_OF_THE_GROUND_TRUTH_VIDEOS", help="Folder with ground truth videos")
    parser.add_argument("--output_dir", default="OUTPUT_DIR", help="Output directory for comparison videos")
    
    args = parser.parse_args()
    
    find_better_videos(args.folder1, args.folder2, args.gt_folder, args.output_dir)

if __name__ == "__main__":
    main()
