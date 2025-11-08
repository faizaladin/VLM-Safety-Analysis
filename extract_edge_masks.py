import os
import cv2
import numpy as np

EXTRACTED_FRAMES = "extracted_frames"
EDGE_MASKS_ROOT = "edge_masks"

for town in os.listdir(EXTRACTED_FRAMES):
    town_path = os.path.join(EXTRACTED_FRAMES, town)
    if not os.path.isdir(town_path):
        continue
    for weather in os.listdir(town_path):
        weather_path = os.path.join(town_path, weather)
        if not os.path.isdir(weather_path):
            continue
        for category in ["success", "collision", "lane_violation"]:
            cat_path = os.path.join(weather_path, category)
            if not os.path.isdir(cat_path):
                continue
            for traj in os.listdir(cat_path):
                traj_folder = os.path.join(cat_path, traj)
                if not os.path.isdir(traj_folder):
                    continue
                out_folder = os.path.join(EDGE_MASKS_ROOT, town, weather, category, traj)
                os.makedirs(out_folder, exist_ok=True)
                for frame_file in os.listdir(traj_folder):
                    if not frame_file.endswith(".jpg"):
                        continue
                    frame_path = os.path.join(traj_folder, frame_file)
                    img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                    edges = cv2.Canny(img, 100, 200)
                    # Convert to binary mask (0 or 255)
                    mask = np.where(edges > 0, 255, 0).astype(np.uint8)
                    out_path = os.path.join(out_folder, frame_file)
                    cv2.imwrite(out_path, mask)
print("Edge mask extraction complete.")
