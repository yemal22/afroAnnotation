from typing import Literal
from PIL import Image
import io
import os
import torch
import requests

from transformers import BlipProcessor, BlipForConditionalGeneration

# Paths to fine-tuned model directories
FASHION_MODEL_PATH = "models/blip-afro-fashion-v1.0.0"
FOOD_MODEL_PATH = "models/blip-afro-food-v1.0.0"

# Load models and processors
fashion_model = BlipForConditionalGeneration.from_pretrained(FASHION_MODEL_PATH)
food_model = BlipForConditionalGeneration.from_pretrained(FOOD_MODEL_PATH)
fashion_processor = BlipProcessor.from_pretrained(FASHION_MODEL_PATH)
food_processor = BlipProcessor.from_pretrained(FOOD_MODEL_PATH)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fashion_model.to(device)
food_model.to(device)

def generate_caption(image_input, model_type: Literal["fashion", "food"] = "fashion") -> str:
    """
    Generate a caption for an image input.

    Args:
        image_input (str | bytes): File path, URL or raw bytes.
        model_type (str): Either 'fashion' or 'food' to select the model.

    Returns:
        str: The generated caption.
    """
    if model_type == "fashion":
        processor = fashion_processor
        model = fashion_model
    elif model_type == "food":
        processor = food_processor
        model = food_model
    else:
        raise ValueError("model_type must be 'fashion' or 'food'")

    # Load image from path, URL or bytes
    if isinstance(image_input, str):
        if image_input.startswith("http"):
            response = requests.get(image_input)
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        elif os.path.isfile(image_input):
            image = Image.open(image_input).convert("RGB")
        else:
            raise ValueError("Invalid image path or URL")
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    else:
        raise ValueError("Unsupported image input type")

    # Process and generate caption
    inputs = processor(image, return_tensors="pt").to(device)
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)

    return caption

