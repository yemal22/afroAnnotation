from datasets import load_dataset
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
from evaluate import load
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

bleu_metric = load("bleu")
rouge_metric = load("rouge")
#cider_metric = load("cider")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

    bleu_score = bleu_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    rouge_score = rouge_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    #cider_score = cider_metric.compute(predictions=decoded_preds, references=[[label] for label in decoded_labels])
    
    return {
        "bleu": bleu_score,
        "rouge": rouge_score
        #"cider": cider_score
    }


# Custom dataset class to handle image loading
class ImageCaptioningDataset(Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        caption = item["caption"]
        
        # Process image and text
        inputs = self.processor(
            images=image, 
            text=caption, 
            padding="max_length",
            return_tensors="pt",
            truncation=True
        )
        
        # Remove batch dimension and convert to correct types
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"]  # For BLIP, labels are same as input_ids
        
        return inputs

# Load dataset
dataset = load_dataset("yemalin/african-food")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)

# Load processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create train and test datasets
train_dataset = ImageCaptioningDataset(dataset["train"], processor)
eval_dataset = ImageCaptioningDataset(dataset["test"], processor)

# LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["query", "value", "key", "dense"]  # Simplified target modules
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
# training_args = TrainingArguments(
#     output_dir="./blip-afro-food",
#     per_device_train_batch_size=4,
#     num_train_epochs=3,
#     learning_rate=0.0001,
#     fp16=True,
#     save_strategy="epoch",
#     logging_steps=100,
#     remove_unused_columns=False,
#     push_to_hub=False,
#     report_to="none",
#     evaluation_strategy="epoch",
# )

training_args = TrainingArguments(
    output_dir="./blip-afro-food",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    learning_rate=0.01,
    fp16=True,
    save_strategy="no",
    logging_steps=100,
    remove_unused_columns=True,
    evaluation_strategy="no",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

# Save
model.save_pretrained("./blip-afro-food")
processor.save_pretrained("./blip-afro-food")
 