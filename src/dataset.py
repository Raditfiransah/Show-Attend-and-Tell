import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
from PIL import Image
import torchvision.transforms as transforms
from collections import Counter
import nltk

# NLTK download handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    except:
        pass # rely on external download or docker build step

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [word.lower() for word in nltk.word_tokenize(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class Flickr8kDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, vocab=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # Load dataset from Arrow files
        # Adjust path to match where the arrow files actually are.
        # Based on file listing, they seem to be in data/ directly or a subdirectory.
        # We will assume standard HF datasets structure or 'dataset_dict.json' location.
        try:
             self.dataset = load_from_disk(root_dir)[split]
        except Exception as e:
            print(f"Error loading from disk: {e}. Trying to load assuming root_dir is the dataset dict path.")
            pass 

        # self.dataset has columns: image, caption_0, caption_1, caption_2, caption_3, caption_4
        self.caption_columns = [f'caption_{i}' for i in range(5)]
        
        self.vocab = vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image = item['image']
        
        # Select a caption
        if self.split == 'train':
            # Randomly select one of the 5 captions
            import random
            cap_col = random.choice(self.caption_columns)
            caption = item[cap_col]
        else:
            # For validation/test, use the first one (or handle differently for evaluation)
            # Typically for evaluation we need all 5, but for the basic loop we return one target.
            caption = item['caption_0'] 
            # In a real eval script, we might access dataset directly to get all 5.

        if self.transform:
            image = self.transform(image)

        if self.vocab:
            numericalized_caption = [self.vocab.stoi["<SOS>"]]
            numericalized_caption += self.vocab.numericalize(caption)
            numericalized_caption.append(self.vocab.stoi["<EOS>"])
            return image, torch.tensor(numericalized_caption)
        
        return image, caption

class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets

def get_loader(root_dir, transform, batch_size=32, num_workers=4, split='train', vocab=None):
    dataset = Flickr8kDataset(root_dir=root_dir, split=split, transform=transform, vocab=vocab)
    
    pad_idx = vocab.stoi["<PAD>"]
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split=='train'),
        pin_memory=True,
        collate_fn=CustomCollate(pad_idx=pad_idx)
    )
    return loader, dataset
