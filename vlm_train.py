

"""
Llava training/inference scaffold using Hugging Face transformers and PEFT (LoRA).
Loads initial sequence, prompt, and expected label from llava_input.json.
"""

import json
from PIL import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class LlavaSequenceDataset(Dataset):
    def __init__(self, json_path, num_frames=16, transform=None, concat_resize=(224, 224)):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.num_frames = num_frames
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        self.concat_resize = concat_resize
        self.label_map = {"success": 0, "collision": 1, "lane violation": 2}
        objects = set()
        for entry in self.data:
            obj = entry.get("collision_object")
            if obj:
                objects.add(obj)
        self.collision_object_map = {obj: i for i, obj in enumerate(sorted(objects))}
        print("Collision objects:", self.collision_object_map)
        print("Number of collision objects:", len(self.collision_object_map))

    def __len__(self):
        return len(self.data)

    def concatenate_images(self, image_paths):
        images = [Image.open(p).convert("RGB").resize(self.concat_resize) for p in image_paths]
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths)
        max_height = max(heights)
        new_img = Image.new("RGB", (total_width, max_height))
        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0]
        return new_img

    def __getitem__(self, idx):
        entry = self.data[idx]
        images = entry["images"][:self.num_frames]
        concat_img = self.concatenate_images(images)
        img_tensor = self.transform(concat_img)
        prompt = entry["prompt"]
        expected = entry["expected"]
        label_id = self.label_map[expected]
        collision_object = entry.get("collision_object", None)
        if collision_object:
            collision_object_id = self.collision_object_map[collision_object]
        else:
            collision_object_id = -1
        return img_tensor, prompt, label_id, collision_object_id

def get_llava_lora_model(model_name="llava-hf/llava-1.5-7b-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer

if __name__ == "__main__":
    num_frames = 16
    dataset = LlavaSequenceDataset("llava_input.json", num_frames=num_frames)
    total_len = len(dataset)
    indices = list(range(total_len))
    split = int(0.8 * total_len)
    train_indices = indices[:split]
    eval_indices = indices[split:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    eval_set = torch.utils.data.Subset(dataset, eval_indices)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)  # batch_size=1 for V100
    eval_loader = DataLoader(eval_set, batch_size=1, shuffle=False)

    wandb.init(project="vlm_llava_training", name="run_lora_v100")

    print("Training set size:", len(train_set))
    print("Eval set size:", len(eval_set))

    # Load LLaVA model with LoRA
    model, tokenizer = get_llava_lora_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Training loop (dual-label logic)
    criterion_main = torch.nn.CrossEntropyLoss()
    criterion_collision = torch.nn.CrossEntropyLoss()
    for batch_idx, batch in enumerate(train_loader):
        images, prompts, label_ids, collision_object_ids = batch
        inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
        # Forward pass
        outputs = model(**inputs)
        # Main label loss (success, lane violation, collision)
        main_logits = outputs.logits[:, -1, :3]  # Assume last token, first 3 classes
        main_loss = criterion_main(main_logits, label_ids.to(device))
        # Collision object loss (only for collision)
        collision_mask = (label_ids == 1)  # 1 = collision
        if collision_mask.any():
            collision_logits = outputs.logits[:, -1, 3:3+len(dataset.collision_object_map)]
            collision_labels = collision_object_ids[collision_mask].to(device)
            collision_logits = collision_logits[collision_mask]
            collision_loss = criterion_collision(collision_logits, collision_labels)
            total_loss = main_loss + collision_loss
        else:
            total_loss = main_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        wandb.log({"train/total_loss": total_loss.item(), "train/main_loss": main_loss.item(), "train/batch_idx": batch_idx})
        print(f"Batch {batch_idx} total loss: {total_loss.item():.4f}")
        if batch_idx > 2:
            break

    # Eval loop (dual-label logic with wandb logging of prompt, label, image, and response)
    import torchvision.utils as vutils
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader):
            images, prompts, label_ids, collision_object_ids = batch
            inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=32)
            responses = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
            main_logits = model(**inputs).logits[:, -1, :3]
            main_loss = criterion_main(main_logits, label_ids.to(device))
            collision_mask = (label_ids == 1)
            if collision_mask.any():
                collision_logits = model(**inputs).logits[:, -1, 3:3+len(dataset.collision_object_map)]
                collision_labels = collision_object_ids[collision_mask].to(device)
                collision_logits = collision_logits[collision_mask]
                collision_loss = criterion_collision(collision_logits, collision_labels)
                total_loss = main_loss + collision_loss
            else:
                total_loss = main_loss
            # Log image, prompt, label, and response to wandb
            for i in range(len(prompts)):
                wandb.log({
                    "eval/prompt": prompts[i],
                    "eval/label": label_ids[i].item(),
                    "eval/image": [wandb.Image(vutils.make_grid(images[i].unsqueeze(0), nrow=1))],
                    "eval/response": responses[i],
                    "eval/total_loss": total_loss.item(),
                    "eval/main_loss": main_loss.item(),
                    "eval/batch_idx": batch_idx
                })
            print(f"Eval batch {batch_idx} total loss: {total_loss.item():.4f}")
            if batch_idx > 2:
                break
