"""
This script prepares the dataset for training and evaluation.
It aims to load Hugging Face datasets containing images, labels and IDs,
with the json file containing the captions and prompt for each image,
to build a Hugging Face Dataset involving on the concatenation of features.
The final dataset's features are:
- image: Image
- prompt: Value(dtype="string")
- caption: Value(dtype="string")

"""

from datasets import load_from_disk, Dataset, Features, Value, ClassLabel, Image as HfImage
import pandas as pd
from PIL import Image
import os
import json
from typing import List, Dict
from tqdm import tqdm

def load_caption_file(caption_file: str) -> Dict[str, List[Dict]]:
    """
    Load the caption file and return the prompt and captions.
    
    Args:
        caption_file (str): Path to the caption file.
        
    Returns:
        Dict[str, List[Dict]]: Dictionary containing the prompt and captions.
    """
    with open(caption_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_hf_dataset(
    dataset_path: str,
    caption_file: str,
    output_path: str
) -> Dataset:
    """
    Prepare the Hugging Face dataset by loading images and captions.
    
    Args:
        dataset_path (str): Path to the dataset.
        caption_file (str): Path to the caption file.
        output_path (str): Path to save the prepared dataset.
        
    Returns:
        Dataset: Prepared Hugging Face dataset.
    """
    # Load the dataset
    print(f"[INFO] 📦 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Load the caption file
    print(f"[INFO] 📜 Loading captions from: {caption_file}")
    captions_data = load_caption_file(caption_file)
    
    # Prepare the features
    features = Features({
        "image": HfImage(),
        "prompt": Value(dtype="string"),
        "caption": Value(dtype="string")
    })
    
    # Create a new dataset with images, prompts and captions
    new_dataset = []
    
    for example in tqdm(dataset, desc="🛠️ Preparing dataset"):
        id = example['id']
        image = example['image']
        label = example['label']
        
        # Get the corresponding caption and prompt
        caption_info = next((item for item in captions_data['captions'] if item['id'] == id), None)
        
        if caption_info:
            prompt = captions_data['prompt']
            caption = caption_info['caption']
            
            new_dataset.append({
                "image": image,
                "prompt": prompt,
                "caption": caption
            })
    
    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(new_dataset, features=features)
    
    # Save the prepared dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    hf_dataset.save_to_disk(output_path)
    
    print(f"[INFO] 💾 Prepared dataset saved to: {output_path}")
    
    return hf_dataset

if __name__ == '__main__':
    
    # For fashion dataset
    prepare_hf_dataset(
        dataset_path="data/processed/african-fashion",
        caption_file="data/captions/african-fashion-04.json",
        output_path="data/processed/african-fashion-hf"
    )

    # For food dataset
    prepare_hf_dataset(
        dataset_path="data/processed/african-food",
        caption_file="data/captions/african-food-04.json",
        output_path="data/processed/african-food-hf"
    )
