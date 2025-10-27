from PIL import Image
import matplotlib.pyplot as plt
import json
from vlm_train import LlavaSequenceDataset

def concatenate_images(image_paths, resize=(128, 128)):
    images = [Image.open(p).convert("RGB").resize(resize) for p in image_paths]
    total_width = sum(img.size[0] for img in images)
    max_height = max(img.size[1] for img in images)
    new_img = Image.new("RGB", (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img

# Example usage:
dataset = LlavaSequenceDataset("llava_input.json")
img_tensor, prompt, label_id, collision_object_id = dataset[0]
print("Prompt:", prompt)
print("Label ID:", label_id)
print("Collision Object ID:", collision_object_id)
print("Image tensor shape:", img_tensor.shape)

with open("llava_input.json", "r") as f:
    data = json.load(f)
image_paths = data[0]["images"][:20]  # Use up to 20 images from the first entry
concat_img = concatenate_images(image_paths, resize=(128, 128))

plt.figure(figsize=(20, 5))
plt.imshow(concat_img)
plt.axis('off')
plt.show()

# Show first data point details
print("Prompt:", data[0]["prompt"])
print("Expected label:", data[0]["expected"])
print("Collision object:", data[0]["collision_object"])
print("Image paths:", data[0]["images"][:20])