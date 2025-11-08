import os
import cv2

# Extract frames from all mp4 files in Combined Data
COMBINED_DATA = "Combined Data"
FRAMES_ROOT = "extracted_frames"

for town in os.listdir(COMBINED_DATA):
    town_path = os.path.join(COMBINED_DATA, town)
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
            for mp4_file in os.listdir(cat_path):
                if not mp4_file.endswith(".mp4"):
                    continue
                video_path = os.path.join(cat_path, mp4_file)
                out_folder = os.path.join(FRAMES_ROOT, town, weather, category, mp4_file[:-4])
                os.makedirs(out_folder, exist_ok=True)
                cap = cv2.VideoCapture(video_path)
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = os.path.join(out_folder, f"frame_{frame_idx:04d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_idx += 1
                cap.release()
print("Frame extraction complete.")
