from datasets import load_from_disk
from tqdm import tqdm
import json
from PIL import Image
from utils.call_openai_api import describe_image
import os

def get_prompt(category, label):
    """
    Returns a refined, contextual, image-grounded prompt.
    """
    if category == "fashion":
        return (
            f"In this image, describe the person or group wearing '{label}' clothing. "
            "Indicate whether it's a man, woman, child, or group, and focus on their clothing style, colors, and patterns. "
            "Mention any visible accessories, fabrics (like wax or vlisco), and cultural hints if they are visually obvious. "
            "Do not explain what the label means. Only describe what can be seen."
        )
    elif category == "food":
        return (
            f"In this image, describe the African dish labeled '{label}'. "
            "Mention the visible ingredients, how it's served, and any accompaniments. "
            "If the dish is part of a meal or has a certain presentation style, describe it. "
            "Do not explain the label; stay focused on what is visible."
        )
    else:
        return (
            "Briefly describe what is happening in the image, focusing only on visual details."
        )


def generate_captions(dataset_path, output_path, category="fashion", max_samples=None):
    """
    Generates image captions using GPT-4o for a HuggingFace dataset.
    The dataset should contain images and labels.

    Args:
        dataset_path (str): Path to the dataset.
        output_path (str): Path to save the generated captions.
        category (str): Category of the dataset ("fashion" or "food").
        max_samples (int): Maximum number of samples to process.

    Returns:
        None
    """
    print(f"\n[INFO] 📦 Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    print(f"[INFO] ✅ Dataset loaded successfully.")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    print(f"[INFO] 🎯 Processing {len(dataset)} samples in '{category}' category.")

    results = []
    prompt_base = None  # Will store the prompt from first item (they're similar within category)

    for example in tqdm(dataset, desc=f"{category.title()} Captions"):
        id = example['id']
        image = example['image']
        label = example['label']
        label_str = str(label) if isinstance(label, str) else dataset.features['label'].int2str(label)

        # Refined and label-aware prompt
        prompt = get_prompt(category, label_str)
        if prompt_base is None:
            prompt_base = prompt  # Save the first used prompt

        try:
            caption = describe_image(prompt, image=image, label=label_str, show_image=False)
            results.append({
                "id": id,
                "label": label_str,
                "caption": caption
            })
        except Exception as e:
            print(f"[ERROR] ❌ Failed to generate caption for image ID={id}: {e}")
            results.append({
                "id": id,
                "label": label_str,
                "caption": None
            })

    print(f"\n[INFO] 💾 Saving {len(results)} captions to {output_path}...")

    output_data = {
        "prompt": prompt_base,
        "captions": results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print("✅ Captions saved successfully.\n")

if __name__ == "__main__":

    # Fashion dataset
    generate_captions(
        dataset_path="data/processed/african-fashion",
        output_path="data/captions/african-fashion-04.json",
        category="fashion",
        max_samples=10
    )

    # Food dataset
    generate_captions(
        dataset_path="data/processed/african-food",
        output_path="data/captions/african-food-04.json",
        category="food",
        max_samples=10
    )
