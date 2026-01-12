import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
import os

from model import Encoder, DecoderLSTM
from dataset import get_loader
from utils import load_vocab, load_embeddings, evaluate_bleu
from config import Config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/cnn_lstm/best_model.pth')
    parser.add_argument('--vocab_path', type=str, default=Config.VOCAB_PATH)
    parser.add_argument('--embedding_path', type=str, default='glove.6B.200d.txt')
    parser.add_argument('--data_dir', type=str, default=Config.DATA_DIR)
    parser.add_argument('--split', type=str, default='validation')
    args = parser.parse_args()

    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load Vocab
    if not os.path.exists(args.vocab_path):
        print(f"Vocabulary not found at {args.vocab_path}")
        return
        
    vocab = load_vocab(args.vocab_path)
    print(f"Vocabulary loaded. Size: {len(vocab)}")

    # Load Embeddings (Optional replacement)
    # Note: If loading a trained model checkpoint, it might already have embeddings.
    # The user request said "replace ... with a pretrained embedding". 
    # This implies we might be initializing a new model or replacing weights in an existing one?
    # Usually replacing embeddings happens BEFORE training.
    # If we replace embeddings in a TRAINED model, performance will drop significantly unless we retrain/finetune.
    # However, if the goal is to evaluate a model that WAS trained with pretrained embeddings, we just load the model.
    # If the goal is to SHOW how to use pretrained embeddings, we initialize a model with them.
    
    # We will assume we are loading a checkpoint validation. 
    # If embeddings path is provided and we want to replace (e.g. for training), we would do it at init.
    # But for evaluation, we rely on the checkpoint's weights usually.
    
    # BUT, if the user wants to see the effect of replacing it:
    # Maybe they want to Initialize -> Load Pretrained -> Evaluate (which will be bad without training).
    # OR Initialize -> Load Pretrained -> Train -> Evaluate.
    
    # I will demonstrate loading the pretrained embeddings into the model structure.
    
    # Load Data
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_loader, val_dataset = get_loader(
        args.data_dir,
        transform,
        batch_size=Config.BATCH_SIZE,
        split=args.split,
        vocab=vocab
    )
    
    # Initialize Model
    encoder = Encoder().to(device)
    
    # Check embedding dim
    # GloVe often comes in 50, 100, 200, 300.
    # Config.EMBED_DIM is 256.
    # If we use 200d glove, we must change ref.
    
    # Let's try to load embeddings if file exists
    embeddings = None
    embed_dim = Config.EMBED_DIM
    
    if os.path.exists(args.embedding_path):
        print(f"Loading pretrained embeddings from {args.embedding_path}")
        # We need to know the dim of the file. 
        # Typically we'd peek or user specifies.
        # We'll assume it matches config or we use a custom dim.
        # For this script we stick to Config defaults unless we detect otherwise, which is hard.
        embeddings = load_embeddings(vocab, args.embedding_path, embed_dim)
    
    decoder = DecoderLSTM(
        attention_dim=Config.ATTENTION_DIM,
        embed_dim=embed_dim,
        decoder_dim=Config.HIDDEN_DIM,
        vocab_size=len(vocab),
        dropout=Config.DROPOUT,
        pretrained_embeddings=embeddings
    ).to(device)
    
    # Load checkpoint if exists
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        decoder.load_state_dict(checkpoint['state_dict'], strict=False) 
        # strict=False allows loading if embeddings are different (though checking sizes will fail if mismatch)
        if 'encoder' in checkpoint:
             encoder.load_state_dict(checkpoint['encoder'])
    else:
        print("No checkpoint found. Evaluating with initialized weights (random or pretrained).")

    # Evaluate
    evaluate_bleu(val_dataset, encoder, decoder, device)

if __name__ == "__main__":
    main()
