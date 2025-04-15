"""
🧩 Dataset Merger for African Fashion & Cuisine (Hugging Face Format)

This script merges multiple Hugging Face datasets stored locally into two
main domains: Fashion and Food. It ensures label consistency by:

1. Mapping numeric labels to their corresponding class names using ClassLabel.
2. Unifying the label space across datasets.
3. Concatenating datasets with and without labels, preserving metadata.
4. Saving the merged datasets for future use (e.g., image captioning, classification).

Author: Yémalin Emile Morel KPAVODE
Date: April 2025
"""

from datasets import load_from_disk, Dataset, DatasetDict, concatenate_datasets, ClassLabel
import os

def decode_label(example, label_feature):
    """
    Converts a numeric label into its corresponding string label using ClassLabel.

    Parameters:
    - example (dict): Example from the dataset containing a numeric 'label'.
    - label_feature (ClassLabel): The label feature to decode the label.

    Returns:
    - dict: Example with 'label' replaced by its string class name.
    """
    example['label'] = label_feature.int2str(example['label'])
    return example

def prepare_dataset_with_labels(dataset, label_feature):
    """
    Applies label decoding to a dataset with numeric labels.

    Parameters:
    - dataset (Dataset): A HuggingFace dataset with numeric labels.
    - label_feature (ClassLabel): Feature object to map label integers to strings.

    Returns:
    - Dataset: A new dataset with string-based labels.
    """
    return dataset.map(lambda ex: decode_label(ex, label_feature))

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

    # Decode label in attire dataset
    label_feature = attire_ds.features['label']
    attire_ds = prepare_dataset_with_labels(attire_ds, label_feature)

    # Add missing 'label' field to wax dataset with placeholder
    wax_ds = wax_ds.map(lambda x: {'label': 'unknown'})

    # Merge
    merged = concatenate_datasets([attire_ds, wax_ds])
    return merged

def merge_food_datasets():
    """
    Merges two food-related datasets:
    - Nigerian foods (10 categories)
    - Ghanaian & Cameroonian foods (6 categories)

    Returns:
    - Dataset: Unified food dataset with consistent string-based labels.
    """
    nigerian_path = "data/raw/nigerian-foods"
    gc_path = "data/raw/food/ghana-cameroun-foods"

    # Load datasets
    nigeria_ds = load_from_disk(nigerian_path)
    gc_ds = load_from_disk(gc_path)

    # Decode labels
    nigeria_ds = prepare_dataset_with_labels(nigeria_ds, nigeria_ds.features['label'])
    gc_ds = prepare_dataset_with_labels(gc_ds, gc_ds.features['label'])

    # Merge
    merged = concatenate_datasets([nigeria_ds, gc_ds])
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
