from datasets import load_from_disk, DatasetDict
from tqdm import tqdm
import json
from PIL import Image
from utils.call_openai_api import describe_image
import os

def generate_captions(dataset_path, output_path, category="fashion", max_samples=None):
    """
    Generates image captions using GPT-4o for a HuggingFace dataset.
    The dataset should contain images and labels.
    The generated captions are saved in a JSON file.
    
    Parameters:
        - dataset_path (str): Path to the dataset.
        - output_path (str): Path to save the generated captions.
        - category (str): Category of the dataset (e.g., "fashion", "food").
        - max_samples (int): Maximum number of samples to process. If None, processes all samples.
    
    Returns:
        - None
    """
    
    print(f"\n[INFO] 📦 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"[INFO] 📦 Dataset loaded successfully.")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"[INFO] 📦 Processing {len(dataset)} samples in '{category}' category.")
    
    results = []
    
    for example in tqdm(dataset, desc=f"{category.title()} captions"):
        image = example['image']
        label = example['label']
        label_str = str(label) if isinstance(label, str) else dataset.features['label'].int2str(label)
        
        if category == "fashion":
            prompt = "Describe this piece of African clothing."
        elif category == "food":
            prompt = "Describe this African dish."
        else:
            prompt = "Describe this image."
        try:
            caption = describe_image(prompt, image=image, label=label_str)
            results.append({
                #"image": image,
                "label": label_str,
                "caption": caption
            })
        except Exception as e:
            print(f"[ERROR] ❌ Failed to generate caption for image {image}: {e}")
            results.append({
                "image": image,
                "label": label_str,
                "caption": None
            })
    print(f"\n[INFO] 💾 Saving {len(results)} captions to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("✅ Captions saved successfully.")
    
if __name__ == "__main__":
    
    # Fashion dataset
    generate_captions(
        dataset_path="data/processed/african-fashion",
        output_path="data/processed/african-fashion-captions.json",
        category="fashion",
        max_samples=20 # Optional: Set to None to process all samples
    )
    
    # Food dataset
    generate_captions(
        dataset_path="data/processed/african-food",
        output_path="data/processed/african-food-captions.json",
        category="food",
        max_samples=20 # Optional: Set to None to process all samples
    )