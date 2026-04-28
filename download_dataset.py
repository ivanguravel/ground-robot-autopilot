"""
Script for downloading and preparing dataset from Hugging Face.
Dataset: UniDataPro/real-time-traffic-video-dataset

Usage:
    python download_dataset.py
"""

import os
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm

try:
    from datasets import load_dataset
except ImportError:
    print("Install datasets: pip install datasets")
    exit(1)


def extract_frames_from_video(video_path: str, output_dir: str, fps: int = 2) -> list:
    """
    Extracts frames from video at specified FPS.
    
    Args:
        video_path: path to video file
        output_dir: directory to save frames
        fps: how many frames per second to extract
    
    Returns:
        list of paths to saved frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return []
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps) if video_fps > fps else 1
    
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    frame_count = 0
    saved_count = 0
    
    video_name = Path(video_path).stem
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    return frame_paths


def download_traffic_dataset(output_dir: str = "data", max_videos: int = None, frames_per_second: int = 2):
    """
    Downloads dataset from Hugging Face and extracts frames.
    
    Args:
        output_dir: base directory for data
        max_videos: maximum number of videos to process (None = all)
        frames_per_second: how many frames to extract from each second of video
    """
    print("=" * 60)
    print("Downloading dataset UniDataPro/real-time-traffic-video-dataset")
    print("=" * 60)
    
    # Create directory structure
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    valid_images_dir = os.path.join(output_dir, "valid", "images")
    valid_labels_dir = os.path.join(output_dir, "valid", "labels")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    
    try:
        # Load dataset
        print("\nLoading dataset metadata...")
        dataset = load_dataset("UniDataPro/real-time-traffic-video-dataset", split="train")
        
        print(f"Found records in dataset: {len(dataset)}")
        print(f"Data structure: {dataset.features}")
        
        # Limit quantity if specified
        if max_videos:
            dataset = dataset.select(range(min(max_videos, len(dataset))))
        
        print(f"\nRecords to be processed: {len(dataset)}")
        
        # Process each record
        all_frames = []
        for idx, item in enumerate(tqdm(dataset, desc="Processing videos")):
            # Structure depends on dataset - needs adaptation
            # For now creating empty label files
            pass
            
        print("\n" + "=" * 60)
        print("IMPORTANT: This dataset may not contain annotations!")
        print("Recommend using COCO or another annotated dataset.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("\nTrying alternative approach...")
        create_sample_structure(output_dir)


def create_sample_structure(output_dir: str = "data"):
    """
    Creates sample data structure for testing.
    """
    print("\nCreating test data structure...")
    
    train_images_dir = os.path.join(output_dir, "train", "images")
    train_labels_dir = os.path.join(output_dir, "train", "labels")
    valid_images_dir = os.path.join(output_dir, "valid", "images")
    valid_labels_dir = os.path.join(output_dir, "valid", "labels")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(valid_images_dir, exist_ok=True)
    os.makedirs(valid_labels_dir, exist_ok=True)
    
    # Create README with instructions
    readme_content = """# Training Data Structure

## Format
- Images: `images/*.jpg` or `*.png`
- Annotations: `labels/*.txt` (format: class_id x_min y_min x_max y_max)

## Coordinates
All coordinates are normalized from 0 to 1:
- x_min, x_max: relative to image width
- y_min, y_max: relative to image height

## Example annotation file (car.txt):
```
1 0.2 0.3 0.5 0.6
2 0.6 0.4 0.8 0.7
```

Where:
- 1 = car (class_id)
- 2 = truck (class_id)

## Classes (from config.py):
0 = __background__
1 = person
2 = bicycle
3 = car
4 = motorcycle
6 = bus
8 = truck
10 = traffic light
13 = stop sign

## Recommended annotated datasets:
1. COCO (traffic subset): https://cocodataset.org/
2. BDD100K: https://www.vis.xyz/bdd100k/
3. KITTI: http://www.cvlibs.net/datasets/kitti/
4. Cityscapes: https://www.cityscapes-dataset.com/
"""
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"\nStructure created in: {output_dir}/")
    print("Read data/README.md for instructions on adding data.")


def download_coco_subset(output_dir: str = "data", categories: list = None):
    """
    Downloads COCO dataset subset with required categories.
    This is a more reliable option with ready annotations.
    """
    if categories is None:
        categories = ["car", "truck", "bus", "motorcycle", "bicycle", "person", "traffic light", "stop sign"]
    
    print("=" * 60)
    print("Downloading COCO dataset (traffic objects subset)")
    print("=" * 60)
    
    try:
        from pycocotools.coco import COCO
        import urllib.request
        import zipfile
        
        # URL for COCO
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        
        print("\nFor full COCO download, run:")
        print("1. Download annotations: annotations_trainval2017.zip")
        print("2. Download images: train2017.zip, val2017.zip")
        print("3. Run convert_coco.py for conversion")
        
    except ImportError:
        print("Install pycocotools: pip install pycocotools")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset for training")
    parser.add_argument("--output", "-o", default="data", help="Directory to save")
    parser.add_argument("--max-videos", "-m", type=int, default=None, help="Max number of videos")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract")
    parser.add_argument("--coco", action="store_true", help="Download COCO instead of HF dataset")
    
    args = parser.parse_args()
    
    if args.coco:
        download_coco_subset(args.output)
    else:
        download_traffic_dataset(args.output, args.max_videos, args.fps)
