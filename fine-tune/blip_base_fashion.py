from datasets import load_dataset
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import Dataset
from evaluate import load
from PIL import Image
import os

bleu_metric = load("bleu")
rouge_metric = load("rouge")

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
dataset = load_dataset("yemalin/african-fashion")
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
training_args = TrainingArguments(
    output_dir="models/blip-afro-fashion",
    per_device_train_batch_size=8,
    num_train_epochs=2,
    learning_rate=0.0001,
    weight_decay=0.01,
    fp16=True,
    save_strategy="epoch",
    logging_steps=100,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    eval_strategy="epoch",
    gradient_accumulation_steps=2
)
training_args = TrainingArguments(
    output_dir="./blip-afro-fashion",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=0.0001,
    fp16=True,
    logging_steps=100,
    weight_decay=0.01,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    label_names=["input_ids"],
    eval_strategy="epoch",
    logging_dir="./logs",
    save_strategy="epoch",                   # Enregistre à chaque époque
    save_total_limit=2,                      # Garde max 2 checkpoints
    save_steps=500,
    load_best_model_at_end=True,             # Charge le meilleur modèle à la fin
)

# Custom compute_metrics function
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = processor.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)
    
    # BLEU avec différentes longueurs de n-grammes
    bleu_results = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels],
        max_order=4  # Pour BLEU-1 à BLEU-4
    )
    
    # ROUGE plus détaillé
    rouge_results = rouge_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    
    return {
        "bleu-1": bleu_results["precisions"][0],
        "bleu-4": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"].mid.fmeasure,
        "rougeL": rouge_results["rougeL"].mid.fmeasure
    }

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
model.save_pretrained("models/blip-afro-fashion")
processor.save_pretrained("models/blip-afro-fashion")
