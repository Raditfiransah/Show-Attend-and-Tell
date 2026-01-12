import os
import sys
from datasets import load_dataset

# Add current directory to path so we can import config if run from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import Config

def download_data():
    """
    Downloads the jxie/flickr8k dataset and saves it to the directory specified in Config.
    """
    data_dir = Config.DATA_DIR
    
    # Check if dataset seems to be present
    if os.path.exists(data_dir) and (os.path.exists(os.path.join(data_dir, 'dataset_dict.json')) or os.path.exists(os.path.join(data_dir, 'train'))):
        print(f"Dataset already exists in '{data_dir}'. Skipping download.")
        return

    print(f"Dataset not found in '{data_dir}'. Downloading 'jxie/flickr8k'...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Download the dataset
        ds = load_dataset("jxie/flickr8k", trust_remote_code=True)
        
        print(f"Saving dataset to '{data_dir}'...")
        ds.save_to_disk(data_dir)
        print("Dataset downloaded and saved successfully!")
        
    except Exception as e:
        print(f"An error occurred during download/save: {e}")

if __name__ == "__main__":
    download_data()
