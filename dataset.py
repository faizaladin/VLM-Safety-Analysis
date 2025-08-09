import os
import csv
import re

frame_dir = "paired_frames"
failure_frames_file = "normalized_failure_frames.txt"

# Parse failure frames as (folder, frame) pairs
failure_set = set()
with open(failure_frames_file, "r") as f:
    for line in f:
        # Split only on the first comma to handle multiple frames
        folder, frames_str = line.strip().split(",", 1)
        folder = folder.strip()
        frame_nums = re.findall(r"\d+", frames_str)
        for frame in frame_nums:
            failure_set.add((folder, frame))

# Recursively find all images in all subfolders
image_paths = []
for subdir in os.listdir(frame_dir):
    subdir_path = os.path.join(frame_dir, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_num_match = re.search(r"\d+", os.path.splitext(file)[0])
                frame_num = str(int(frame_num_match.group())) if frame_num_match else os.path.splitext(file)[0]
                image_paths.append((os.path.join(subdir_path, file), subdir, frame_num))

# Sort image_paths by folder, then by frame number (as int if possible)
def try_int(x):
    try:
        return int(x)
    except ValueError:
        return x

image_paths.sort(key=lambda x: (x[1], try_int(x[2])))


# Assign labels and write to CSV
with open("llava_dataset.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "label"])
    for img_path, folder, frame_num in image_paths:
        label = 0 if (folder, frame_num) in failure_set else 1
        writer.writerow([img_path, label])

# Create JSON for LLaVA 1.5 finetuning
import json
finetune_prompt = (
    "This is a paired image of two cars using a vision based algorithm to steer under different weather conditions, while following the yellow line on the road. First, describe the image on the right. What is the setting (e.g., road type, environment)? What is the weather condition? Next, describe the image on the left. What are the key differences compared to the right image, specifically regarding the time of day, weather, and visibility? Identify the key static objects present in both scenes. Static objects are things that don't move, such as the road, guardrail, fence, mountains, and streetlights. Based on the path indicated by the speckle pattern, is the vehicle heading towards a safe path on the road or is it heading towards one of the static objects you listed earlier (like the guardrail)? This determines if a failure is occurring. Using this information determine if there is a cause of failure in the image on the left, and explain why. If there is no cause of failure, please answer no and explain why there is no cause of failure. If there is, please answer yes followed by reasoning specific to the image and pertains to the weather condition. Create a list of these failures and provide bullet points of reasoning for each one. Lastly, see if there are similar failures that could arise in different weather conditions."
)

jsonl_data = []
for img_path, folder, frame_num in image_paths:
    label = 0 if (folder, frame_num) in failure_set else 1
    entry = {
        "image": img_path,
        "prompt": finetune_prompt,
        "label": label
    }
    jsonl_data.append(entry)

with open("llava_finetune.json", "w") as jf:
    json.dump(jsonl_data, jf, indent=2)