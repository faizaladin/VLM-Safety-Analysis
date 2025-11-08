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
    "Predict the outcome of this initial trajectory as success, lane violation, or collision. If the trajectory is classified as a collision, what static object will the car collide with?"
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