"""
🧩 Dataset Merger for African Fashion & Cuisine (Hugging Face Format)

This script merges multiple Hugging Face datasets stored locally into two
main domains: Fashion and Food. It ensures label consistency by:

1. Mapping numeric labels to their corresponding class names using ClassLabel.
2. Unifying the label space across datasets.
3. Concatenating datasets with and without labels, preserving metadata.
4. Saving the merged datasets for future use (e.g., image captioning, classification).

"""

from datasets import load_from_disk, Dataset, concatenate_datasets, Value, Features, Image as HfImage
import pandas as pd
import os

def convert_class_labels_to_strings(dataset):
    """
    Converts numeric labels in a dataset to their corresponding string labels.

    Parameters:
    - dataset (Dataset): A HuggingFace dataset with numeric labels.

    Returns:
    - Dataset: The dataset with string-based labels.
    """
    label_feature = dataset.features['label'].names
    df = dataset.to_pandas()
    df['label'] = df['label'].apply(lambda x: label_feature[x])
    features = Features({
        "image": HfImage(),
        "label": Value(dtype="string")
    })
    return Dataset.from_pandas(df, features=features).cast(features)

def merge_fashion_datasets():
    """
    Merges two fashion-related datasets:
    - African attire (with ethnic group labels)
    - Wax patterns (without labels)

    Returns:
    - Dataset: Unified fashion dataset with 'label' field where available.
    """
    attire_path = "data/raw/fashion/african-atire"
    wax_path = "data/raw/fashion/african-wax-patterns"

    # Load datasets
    attire_ds = load_from_disk(attire_path)['train']
    wax_ds = load_from_disk(wax_path)['train']
    
    # Add missing 'label' field to wax dataset with placeholder
    wax_ds = wax_ds.map(lambda x: {'label': 'wax-pattern'})
    
    # Convert numeric labels to string labels
    attire_ds = convert_class_labels_to_strings(attire_ds)

    # Merge
    merged = concatenate_datasets([attire_ds, wax_ds])
    
    # Add 5-digit ID field
    merged = merged.map(
        lambda x, idx: {'id': f"img_{idx:05d}"},
        with_indices=True
    )
    
    return merged

def merge_food_datasets():
    """
    Merges two food-related datasets:
    - Nigerian foods (10 categories)
    - Ghanaian & Cameroonian foods (6 categories)

    Returns:
    - Dataset: Unified food dataset with consistent string-based labels.
    """
    nigerian_path = "data/raw/food/nigeria-foods"
    gc_path = "data/raw/food/ghana-cameroun-foods"

    # Load datasets
    nigeria_ds = load_from_disk(nigerian_path)
    gc_ds = load_from_disk(gc_path)

    # Convert numeric labels to string labels
    nigeria_ds = convert_class_labels_to_strings(nigeria_ds)
    gc_ds = convert_class_labels_to_strings(gc_ds)
    
    # Merge
    merged = concatenate_datasets([nigeria_ds, gc_ds])
    
    # Add 5-digit ID field
    merged = merged.map(
        lambda x, idx: {'id': f"img_{idx:05d}"},
        with_indices=True
    )
    
    return merged

def save_merged_datasets():
    """
    Merges and saves fashion and food datasets into the `data/processed` directory.
    """
    os.makedirs("data/processed", exist_ok=True)

    fashion_ds = merge_fashion_datasets()
    food_ds = merge_food_datasets()

    fashion_ds.save_to_disk("data/processed/african-fashion")
    food_ds.save_to_disk("data/processed/african-food")

    print("✅ Datasets successfully merged and saved.")

# Run the pipeline
if __name__ == "__main__":
    save_merged_datasets()
