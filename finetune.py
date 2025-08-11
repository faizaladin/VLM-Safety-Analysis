import json
import torch
import numpy as np
from PIL import Image
from transformers import LlavaForConditionalGeneration, AutoProcessor
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer

class BatchDictDataset(Dataset):
            def __init__(self, batch):
                self.batch = batch
            def __len__(self):
                return len(self.batch)
            def __getitem__(self, idx):
                return self.batch[idx]

class LlavaJsonClassificationDataset(Dataset):
    def __init__(self, json_path, processor, max_length=128):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        text = item['prompt']
        label = item['label']
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            },
        ]
        processed = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        processed = {k: v.squeeze(0) for k, v in processed.items()}
        processed['labels'] = processed['input_ids'].clone()
        processed['label'] = torch.tensor(label, dtype=torch.float)
        return processed

def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([item[key] for item in batch])
        else:
            out[key] = [item[key] for item in batch]
    return out


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    dataset = LlavaJsonClassificationDataset("llava_finetune.json", processor)

    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True
    model = model.to(device)

    # Split dataset: 80% train, 20% val
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    training_dataset, validation_dataset = random_split(dataset, [train_len, val_len])

    # --- Batch calculation and selection logic (restored) ---
    failure_set = set()
    for idx in training_dataset.indices:
        if dataset.data[idx]['label'] == 0:
            failure_set.add(idx)
    print(len(failure_set), "failure frames in training set")

    def nearest_power_of_2(n):
        return 2 ** (n.bit_length() - 1) if n > 0 else 1

    batch_size = nearest_power_of_2(len(failure_set) * 2)
    print(f"Dynamic batch size set to: {batch_size}")

    failure_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 0]
    success_indices = [idx for idx in training_dataset.indices if dataset.data[idx]['label'] == 1]

    half_batch = batch_size // 2
    num_failures = min(half_batch, len(failure_indices))
    num_successes = min(half_batch, len(success_indices))

    selected_failure_indices = np.random.choice(failure_indices, num_failures, replace=False)
    selected_success_indices = np.random.choice(success_indices, num_successes, replace=False)
    batch_indices = np.concatenate([selected_failure_indices, selected_success_indices])
    np.random.shuffle(batch_indices)

    batch = [dataset[idx] for idx in batch_indices]
    batch_dataset = BatchDictDataset(batch)
    print(f"Number of items in the batch (from batch_dataset): {len(batch_dataset)}")

    # --- Trainer setup ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = (logits > 0).astype(int)
        acc = (preds == labels).mean()
        return {"accuracy": acc}

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=batch_dataset,  # Use the selected batch
        eval_dataset=validation_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()

