"""
Export VLM training data to JSON: for each sequence, store initial 16 image paths, label, collision object (if any), and a prompt.
"""
import os
import csv
import json

CSV_PATH = "binary_dataset/dataset.csv"
OUT_JSON = "vlm_sequences.json"
NUM_FRAMES = 50

 # label_map removed, only use label words

prompt_text = (
    "You are analyzing the trajectory of an **autonomous car** that uses a **vision-based controller**.\n\n**Input:** The images provided are a time-sequence of **edge-masks**, which is extracted from the visual data the controller is using to navigate. The full sequence of frames represents the car's predicted trajectory.\n\n**Task:** Analyze the movement implied by this sequence of edge-masks and predict the final outcome as **Success** or **Failure**.\n\n**Definitions:**\n* **Success** = The car stays safely on the road.\n* **Failure** = The car drives off the road or collides with an object (like a curb or building).\n\n**Reasoning:** Explain *why* this sequence leads to your prediction. Describe how the edge lines (representing curbs, buildings, etc.) *move and change* across the frames to indicate the car's path. For example: 'The edge lines of the building on the right are rapidly expanding and moving toward the center, indicating a direct collision course.'"
)


sequences = []
with open(CSV_PATH, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        folder = row["folder"].replace("extracted_frames", "edge_masks")
        frames = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])[:NUM_FRAMES]
        frame_paths = [os.path.join(folder, fname) for fname in frames]
        label = row["label"].lower()
        sequences.append({
            "frames": frame_paths,
            "label": label,
            "prompt": prompt_text
        })


with open(OUT_JSON, "w") as f:
    json.dump(sequences, f, indent=2)
print(f"Exported {len(sequences)} sequences to {OUT_JSON}")


# Export Llava 1.5 input format
LLAVA_OUT_JSON = "llava_input.json"
llava_entries = []
for seq in sequences:
    llava_entries.append({
        "images": seq["frames"],
        "prompt": seq["prompt"],
        "expected": seq["label"]
    })
with open(LLAVA_OUT_JSON, "w") as f:
    json.dump(llava_entries, f, indent=2)
print(f"Exported {len(llava_entries)} entries to {LLAVA_OUT_JSON}")