import json
import torch
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
import torch.nn as nn
import numpy as np

# ======= Model and Dataset Classes (reuse from train script) =======
class LlavaSequenceClassificationDataset:
    def __init__(self, processor, num_frames=50):
        self.processor = processor
        self.num_frames = num_frames

    def concatenate_images(self, image_paths, resize=(112, 112)):
        images = [Image.open(p).convert("L").resize(resize) for p in image_paths[:self.num_frames]]
        if not images:
            return Image.new("L", resize, 0)
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)
        new_img = Image.new("L", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        return new_img

class LlavaClassificationHead(nn.Module):
    def __init__(self, base_model, num_main_classes):
        super().__init__()
        self.base_model = base_model
        hidden_size = self.base_model.language_model.config.hidden_size
        self.main_classifier = nn.Linear(hidden_size, num_main_classes)
        # Freeze classification head
        for param in self.main_classifier.parameters():
            param.requires_grad = False

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        pooled_output = hidden_states.mean(dim=1)
        main_logits = self.main_classifier(pooled_output)
        return main_logits, outputs

# ======= Inference Logic =======
if __name__ == "__main__":
    # Load eval trajectories
    with open("eval_trajectories.json", "r") as f:
        eval_trajectories = json.load(f)

    model_id = "llava-hf/llava-1.5-7b-hf"
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16,
    )
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = 'right'
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    base_model = get_peft_model(base_model, lora_config)

    # Load trained weights
    model = LlavaClassificationHead(base_model, num_main_classes=2)
    model.load_state_dict(torch.load("llava-finetuned-classification.pt", map_location="cpu"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = LlavaSequenceClassificationDataset(processor, num_frames=50)
    label_map = {0: "success", 1: "failure"}

    for traj_idx, image_paths in enumerate(eval_trajectories):
        concat_img = dataset.concatenate_images(image_paths)
        prompt = "USER: <image>\nTrajectory inference ASSISTANT:"
        inputs = processor(
            text=prompt,
            images=concat_img,
            return_tensors="pt",
            padding="max_length",
            max_length=1024,
            truncation=True
        )
        pixel_values = inputs['pixel_values'].to(device)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            logits, outputs = model(pixel_values, input_ids, attention_mask)
            pred = torch.argmax(logits, dim=1).item()
            # Generate text output using base_model's generate method
            gen_ids = base_model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64
            )
            gen_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
            print(f"Trajectory {traj_idx}: Predicted class = {label_map[pred]}")
            print(f"Trajectory {traj_idx}: Generated text = {gen_text}\n")
