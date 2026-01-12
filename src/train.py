import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from datasets import load_from_disk
from nltk.translate.bleu_score import corpus_bleu
import time
import os
import numpy as np

from dataset import get_loader
from model import Encoder, DecoderRNN
from utils import save_checkpoint, load_vocab
from config import Config

def train():
    device = Config.DEVICE
    print(f"Using device: {device}")

    # Load vocabulary
    if not os.path.exists(Config.VOCAB_PATH):
        print(f"Vocabulary file not found at {Config.VOCAB_PATH}. Please run build_vocab.py first.")
        return
    
    vocab = load_vocab(Config.VOCAB_PATH)
    Config.vocab_size = len(vocab)
    print(f"Vocabulary loaded. Size: {Config.vocab_size}")

    # Transforms (Pre-processing)
    # ResNet50 expects mean/std as below
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Data Loaders
    print("Initializing Data Loaders...")
    train_loader, _ = get_loader(
        Config.DATA_DIR, 
        transform, 
        batch_size=Config.BATCH_SIZE, 
        split='train', 
        vocab=vocab
    )
    
    # Validation loader (we might want a smaller batch size or just same)
    val_loader, _ = get_loader(
        Config.DATA_DIR,
        transform,
        batch_size=Config.BATCH_SIZE,
        split='validation',
        vocab=vocab
    )
    
    print(f"Train steps: {len(train_loader)}, Val steps: {len(val_loader)}")

    # Initialize Models
    encoder = Encoder(encoded_image_size=7).to(device)
    decoder = DecoderRNN(
        attention_dim=Config.ATTENTION_DIM,
        embed_dim=Config.EMBED_DIM,
        decoder_dim=Config.HIDDEN_DIM,
        vocab_size=Config.vocab_size,
        dropout=Config.DROPOUT
    ).to(device)

    # Optimization
    # Encoder optimizer (only if fine-tuning)
    encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                   lr=1e-4) if False else None # Start with frozen encoder
    
    decoder_optimizer = optim.Adam(params=decoder.parameters(),
                                   lr=Config.LEARNING_RATE)
                                   
    criterion = nn.CrossEntropyLoss().to(device)

    # Training Loop
    best_bleu = 0.0
    
    for epoch in range(Config.NUM_EPOCHS):
        start = time.time()
        
        # --- Training ---
        encoder.train()
        decoder.train()
        
        # Unfreeze encoder after some epochs? (Optional logic)
        
        losses = []
        
        for i, (imgs, caps) in enumerate(train_loader):
            imgs = imgs.to(device)
            caps = caps.to(device)
            
            # Forward prop
            features = encoder(imgs)
            
            # Caps input: remove <EOS> (last token) for input to decoder
            # Caps target: remove <SOS> (first token) for target
            # Wait, standard practice: Input is <SOS>...<word_N>, Target is <word_1>...<EOS>
            # But the Decoder logic takes encoded_captions.
            
            # The Decoder forward method takes the full caption and handles embedding internally.
            # But we need caption lengths.
            # And `pack_padded_sequence` usually requires sorting by length, which our custom decoder does.
            
            # Calculate lengths. Note caps includes <SOS> and <EOS>.
            # Actual content length (non-pad).
            
            # We need to compute lengths on the CPU
            # Note: caps[i] might be padded with 0 (<PAD>)
            # simple calculation assuming padding is 0.
            
            pad_idx = vocab.stoi["<PAD>"]
            
            # lengths = (caps != pad_idx).sum(dim=1) 
            # This counts SOS and EOS as part of length.
            # We need to verify what `DecoderRNN` expects.
            
            # Re-checking DecoderRNN...
            # `decode_lengths = (caption_lengths - 1).tolist()`
            # It expects `caption_lengths` to be the full length including SOS/EOS (before padding).
            # So if we have `<SOS> A cat <EOS> <PAD>`, length is 4.
            # Decode lengths becomes 3. 
            # Steps: t=0 (<SOS> -> A), t=1 (A -> cat), t=2 (cat -> <EOS>)
            
            # So we just valid lengths
            lengths = torch.sum(caps != pad_idx, dim=1).cpu()

            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(features, caps, lengths)
            
            # Since decoder sorts the input, we need to sort the targets too?
            # `caps_sorted` is already the sorted `encoded_captions` (the FULL captions)
            # Targets are `caps_sorted` shifted by one time-step?
            # Targets: caps_sorted[:, 1:]
            
            targets = caps_sorted[:, 1:]
            
            # Pack padded sequence for efficient loss calculation
            # scores shape: (batch_size, max_decode_len, vocab_size)
            # targets shape: (batch_size, max_decode_len)
            
            # Use pack_padded_sequence to ignore padding in loss calculation?
            # Or simpler: flatten non-padded elements.
            
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            
            loss = criterion(scores_packed, targets_packed)
            
            # Add doubly stochastic attention regularization
            loss += 1. * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Backprop
            decoder_optimizer.zero_grad()
            if encoder_optimizer: encoder_optimizer.zero_grad()
            
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.)
            
            decoder_optimizer.step()
            if encoder_optimizer: encoder_optimizer.step()
            
            losses.append(loss.item())
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{Config.NUM_EPOCHS}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = np.mean(losses)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_train_loss:.4f}. Time: {time.time() - start:.2f}s")
        
        # --- Validation (Simple BLEU-4 proxy or Loss) ---
        # For simplicity in this first pass, we just save the model.
        # A real validation loop would generate captions and calc BLEU.
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': decoder.state_dict(),
            'optimizer': decoder_optimizer.state_dict(),
        }, filename=Config.MODEL_SAVE_PATH)
        
        # Also save encoder if fine-tuned
        if encoder_optimizer:
             torch.save(encoder.state_dict(), 'encoder.pth')

if __name__ == '__main__':
    train()
