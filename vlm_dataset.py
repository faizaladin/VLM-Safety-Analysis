"""
PyTorch Dataset for VLM training: loads first 16 frames from each sequence folder and label from CSV.
Assumes shuffled CSV and frames folders are aligned.
"""
import os
import csv
from typing import List, Tuple
from PIL import Image
from torch.utils.data import Dataset

class VLMTrajectoryDataset(Dataset):
    def save_to_folder(self, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        # Save a CSV with folder paths and labels
        csv_path = os.path.join(output_folder, "dataset.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "label"])
            for entry in self.entries:
                writer.writerow([entry["folder"], entry["label"]])
    def __init__(self, frames_root: str, num_frames: int = 16, transform=None):
        self.entries = []
        self.num_frames = num_frames
        self.transform = transform
        self.label_map = {"success": 0, "failure": 1}

        # Collect all success and failure trajectories from extracted frames
        success_paths = []
        failure_paths = []
        for town in os.listdir(frames_root):
            town_path = os.path.join(frames_root, town)
            if not os.path.isdir(town_path):
                continue
            for weather in os.listdir(town_path):
                weather_path = os.path.join(town_path, weather)
                if not os.path.isdir(weather_path):
                    continue
                # Success
                success_dir = os.path.join(weather_path, "success")
                if os.path.isdir(success_dir):
                    for traj in os.listdir(success_dir):
                        traj_folder = os.path.join(success_dir, traj)
                        if os.path.isdir(traj_folder):
                            success_paths.append({
                                "folder": traj_folder,
                                "label": "success"
                            })
                # Failure: collision and lane_violation
                for fail_type in ["collision", "lane_violation"]:
                    fail_dir = os.path.join(weather_path, fail_type)
                    if os.path.isdir(fail_dir):
                        for traj in os.listdir(fail_dir):
                            traj_folder = os.path.join(fail_dir, traj)
                            if os.path.isdir(traj_folder):
                                failure_paths.append({
                                    "folder": traj_folder,
                                    "label": "failure"
                                })
        # Balance dataset
        min_count = min(len(success_paths), len(failure_paths))
        success_paths = success_paths[:min_count]
        failure_paths = failure_paths[:min_count]
        self.entries = success_paths + failure_paths
        # Shuffle for randomness
        import random
        random.shuffle(self.entries)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Tuple[List[Image.Image], int]:
        entry = self.entries[idx]
        folder = entry["folder"]
        frames = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:self.num_frames]
        images = []
        for fname in frames:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)
        label = self.label_map[entry["label"]]
        return images, label

# Example usage:

if __name__ == "__main__":
    dataset = VLMTrajectoryDataset("edge_masks", num_frames=16)
    # Save to new folder
    dataset.save_to_folder("binary_dataset")
    images, label = dataset[0]
    print(f"Label: {label}")
    print(f"Number of images: {len(images)}")
    print(f"First image size: {images[0].size if images else 'N/A'}")
    print(f"Image types: {[type(img) for img in images]}")
