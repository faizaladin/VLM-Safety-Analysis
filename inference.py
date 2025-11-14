# Batch inference over all videos in evaluation_trajectories.json
import json
import os
import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame)
        if len(frames) == len(indices):
            break
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


# 8-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
)

# Load the model with 8-bit quantization
model = VideoLlavaForConditionalGeneration.from_pretrained(
    "LanguageBind/Video-LLaVA-7B-hf",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")



# Load video paths
with open("evaluation_trajectories.json", "r") as f:
    video_paths = json.load(f)

# Load ground truth labels from metadata.json
with open("vlm_data/metadata.json", "r") as f:
    metadata = json.load(f)
video_to_label = {item["video"]: item["label"] for item in metadata}

def get_ground_truth_label(video_path):
    # Use metadata mapping
    return video_to_label.get(video_path, None)

results = []
for video_path in video_paths:
    print(f"Processing: {video_path}")
    if not os.path.exists(video_path):
        print(f"Missing: {video_path}")
        continue
    try:
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, 20).astype(int)
        video = read_video_pyav(container, indices)
        video = np.transpose(video, (0, 3, 1, 2))  # (num_frames, C, H, W)
        prompt = "USER: <video>\nThis is a video sequence from a car's vision controller. This sequence *is* the trajectory of the car.\n\nPredict: **Success** (stays on road) or **Failure** (off-road or collision).\n\nReasoning: Explain *why* based on how the where the car is heading, weather, and objects the car might collide with. ASSISTANT:"
        inputs = processor(text=prompt, videos=video, return_tensors="pt")
        device = model.device if hasattr(model, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        out = model.generate(**inputs, max_new_tokens=200)
        decoded = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        results.append({"video": video_path, "output": decoded})
        print(f"{video_path}: {decoded}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=2)