from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import os
from io import BytesIO
from starlette.datastructures import UploadFile

FASHION_MODEL_PATH = "models/blip-afro-fashion-v1.0.0"
FOOD_MODEL_PATH = "models/blip-afro-food-v1.0.0"

fashion_model = BlipForConditionalGeneration.from_pretrained(FASHION_MODEL_PATH)
food_model = BlipForConditionalGeneration.from_pretrained(FOOD_MODEL_PATH)
fashion_processor = BlipProcessor.from_pretrained(FASHION_MODEL_PATH)
food_processor = BlipProcessor.from_pretrained(FOOD_MODEL_PATH)

def generate_caption(input_image, model_type="fashion"):
    """
    Generate a caption for an image input using either the fashion or food model.
    
    Args:
        input_image (str | UploadFile | bytes): URL, local path, UploadFile or raw bytes.
        model_type (str): Either "fashion" or "food".
    
    Returns:
        str: Generated caption.
    """
    if model_type == "fashion":
        processor = fashion_processor
        model = fashion_model
    elif model_type == "food":
        processor = food_processor
        model = food_model
    else:
        raise ValueError("model_type must be 'fashion' or 'food'")

    # Convert input to PIL image
    try:
        if isinstance(input_image, str):
            if input_image.startswith("http"):
                response = requests.get(input_image)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            elif os.path.exists(input_image):
                image = Image.open(input_image).convert("RGB")
            else:
                raise ValueError(f"Invalid image path or URL: {input_image}")
        elif isinstance(input_image, UploadFile):
            contents = input_image.file.read()
            image = Image.open(BytesIO(contents)).convert("RGB")
        elif isinstance(input_image, bytes):
            image = Image.open(BytesIO(input_image)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(input_image)}")
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Generate caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=75)
    caption = processor.decode(out[0], skip_special_tokens=True)

    return caption
