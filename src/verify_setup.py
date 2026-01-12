import torch
import sys
import os

def verify():
    print("1. Checking imports...")
    try:
        import datasets
        import torchvision
        import nltk
        import PIL
        print("   Imports successful.")
    except ImportError as e:
        print(f"   Import failed: {e}")
        return

    print("\n2. Checking Data Loading...")
    try:
        # Check if vocab exists
        from config import Config
        if not os.path.exists(Config.VOCAB_PATH):
            print(f"   {Config.VOCAB_PATH} not found. Skipping data loading check requiring vocab.")
        else:
            from dataset import get_loader
            from utils import load_vocab
            
            vocab = load_vocab(Config.VOCAB_PATH)
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor()
            ])
            loader, _ = get_loader(Config.DATA_DIR, transform, split='train', vocab=vocab)
            
            imgs, caps = next(iter(loader))
            print(f"   Batch shapes: Images: {imgs.shape}, Captions: {caps.shape}")
    except Exception as e:
        print(f"   Data loading failed: {e}")

    print("\n3. Checking Model Instantiation and Forward Pass (Dummy Data)...")
    try:
        from model_rnn import Encoder, DecoderRNN, Attention
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        encoder = Encoder().to(device)
        decoder = DecoderRNN(
            attention_dim=256,
            embed_dim=256,
            decoder_dim=512,
            vocab_size=1000, # Dummy vocab size
            dropout=0.5
        ).to(device)
        
        dummy_img = torch.randn(2, 3, 224, 224).to(device)
        dummy_caps = torch.randint(0, 1000, (2, 15)).to(device)
        dummy_lens = torch.tensor([15, 10]).to(device) # Random lengths
        
        # Encoder forward
        features = encoder(dummy_img)
        print(f"   Encoder Output: {features.shape}")
        
        # Decoder forward
        scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, dummy_caps, dummy_lens)
        print(f"   Decoder Output Scores: {scores.shape}")
        
        print("   Model valid!")
    except Exception as e:
        print(f"   Model check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
