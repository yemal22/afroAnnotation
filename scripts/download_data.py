"""
Script to download images from Bing based on African scene categories.
Uses the `icrawler` library to automate image scraping.
Each category contains relevant search keywords.

Requirements:
- pip install icrawler

Usage:
$ python scripts/download_data.py
"""

from icrawler.builtin import BingImageCrawler
import os
from tqdm.notebook import tqdm

def download_images(keyword: str, save_dir: str, max_num: int = 100):
    """
    Downloads images for a specific keyword using BingImageCrawler.

    Parameters:
    - keyword (str): Search term to use for image crawling.
    - save_dir (str): Path to save the downloaded images.
    - max_num (int): Maximum number of images to download.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    crawler = BingImageCrawler(storage={'root_dir': save_dir})
    crawler.crawl(keyword=keyword, max_num=max_num)


if __name__ == "__main__":
    categories = {
        "fashion": [
            "African traditional clothing",
            "Wax fabric fashion",
            "Bazin riche Senegal",
            "Kente wedding Ghana"
        ],
        "markets": [
            "African market scene",
            "West African street market",
            "Traditional African vendors"
        ],
        "rural_landscapes": [
            "African village scene",
            "Rural life in Africa",
            "Traditional African houses"
        ],
        "african_cuisine": [
            "Thieboudienne Senegal",
            "Fufu and soup Ghana",
            "Attiéké poisson Côte d'Ivoire"
        ]
    }

    base_path = "data"

    for category, keywords in categories.items():
        for keyword in keywords:
            print(f"[INFO] Downloading: {keyword} into {category}")
            save_path = os.path.join(base_path, category, "raw")
            download_images(keyword, save_dir=save_path, max_num=1000)
