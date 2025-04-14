from datasets import DatasetDict, Dataset, Features, ClassLabel, Value, Image as HfImage
import os
import pandas as pd
from typing import List, Dict
from tqdm import tqdm

def build_image_dataset_from_folders(root_dir: str, partitions: List[str]) -> DatasetDict:
    """
    Build a Hugging Face DatasetDict from an image folder with pre-defined partitions.

    Args:
        root_dir (str): Root directory containing partition folders like 'train', 'test', etc.
                        Each partition folder must contain subfolders named after the classes.
        partitions (List[str]): List of partition folder names to load (e.g., ['train', 'test']).

    Returns:
        DatasetDict: A dictionary with keys like 'train', 'test', etc. and Dataset values.
    """
    dataset_dict = {}
    all_labels = set()
    data_by_split: Dict[str, List[Dict]] = {}

    print(f"📂 Reading image dataset from: {root_dir}")
    for split in tqdm(partitions, desc="📁 Processing partitions"):
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Partition folder '{split}' not found in '{root_dir}'")
        
        data = []
        class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]

        for class_name in tqdm(class_folders, desc=f"🔍 Scanning '{split}' classes", leave=False):
            class_dir = os.path.join(split_path, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    data.append({
                        "image": os.path.join(class_dir, fname),
                        "label": class_name
                    })
                    all_labels.add(class_name)
        data_by_split[split] = data

    label_names = sorted(all_labels)

    features = Features({
        "image": Value("string"),
        "label": ClassLabel(names=label_names)
    })

    print("🧱 Building Hugging Face datasets...")
    for split, data in tqdm(data_by_split.items(), desc="🛠️ Creating datasets"):
        df = pd.DataFrame(data)
        df["label"] = df["label"].apply(lambda x: label_names.index(x))
        dataset = Dataset.from_pandas(df, features=features)
        dataset = dataset.cast_column("image", HfImage())
        dataset_dict[split] = dataset

    return DatasetDict(dataset_dict)
