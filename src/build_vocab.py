from dataset import Flickr8kDataset, Vocabulary
from utils import save_vocab
from config import Config
import torch
from datasets import load_from_disk
import os

def build_and_save_vocab():
    print("Loading dataset...")
    # Adjust path if necessary based on previous exploration
    dataset = load_from_disk(Config.DATA_DIR)
    
    print("Building vocabulary...")
    # Collect all captions
    # dataset[split] has columns caption_0...caption_4
    
    all_captions = []
    
    # Iterate through the dataset explicitly is slow?
    # Better: access columns directly.
    train_data = dataset['train']
    for i in range(5):
        col_name = f'caption_{i}'
        print(f"Processing {col_name}...")
        all_captions.extend(train_data[col_name])
            
    print(f"Total captions: {len(all_captions)}")
    
    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)
    
    print(f"Vocabulary built! Size: {len(vocab)}")
    
    save_vocab(vocab, Config.VOCAB_PATH)
    print(f"Vocabulary saved to {Config.VOCAB_PATH}")

if __name__ == "__main__":
    build_and_save_vocab()
