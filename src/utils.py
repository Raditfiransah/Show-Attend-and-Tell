import torch
import json
import os
import pickle
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def save_vocab(vocab, path):
    with open(path, 'wb') as f:
        pickle.dump(vocab, f)

def load_vocab(path):
    with open(path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def load_embeddings(vocab, file_path, embedding_dim):
    """
    Loads GloVe vectors or similar.
    Args:
        vocab: vocabulary object
        file_path: path to embedding file (word val1 val2 ...)
        embedding_dim: expected dimension
    Returns:
        embeddings: torch tensor of shape (vocab_size, embedding_dim)
    """
    # Initialize with random
    embeddings = torch.zeros((len(vocab), embedding_dim))
    
    # Random initialization for all (same as default)
    embeddings.uniform_(-0.1, 0.1)
    
    print(f"Loading embeddings from {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"WARNING: Embedding file {file_path} not found. Using random embeddings.")
        return embeddings

    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            values = line.split()
            word = values[0]
            if len(values) != embedding_dim + 1:
                # Skip malformed lines or header
                continue
                
            if word in vocab.stoi:
                vector = np.asarray(values[1:], "float32")
                embeddings[vocab.stoi[word]] = torch.from_numpy(vector)
                count += 1
    
    print(f"Loaded {count} words from embeddings file.")
    return embeddings

def evaluate_bleu(dataset, encoder, decoder, device):
    """
    Evaluate the model using BLEU score.
    """
    encoder.eval()
    decoder.eval()
    
    references = []
    hypotheses = []
    
    print("Evaluating BLEU score on validation set...")
    
    # Iterate over dataset
    # Note: This loops over the whole split provided in dataset
    length = len(dataset)
    
    with torch.no_grad():
        for i in range(length):
            if i % 200 == 0:
                print(f"Processed {i}/{length} images")
            
            # Access raw item to get all captions
            item = dataset.dataset[i]
            image = item['image']
            
            # Helper to get numeric references
            refs = []
            for j in range(5):
                key = f'caption_{j}'
                if key in item:
                    c = item[key]
                    tokens = dataset.vocab.tokenizer_eng(c)
                    refs.append(tokens)
                elif 'caption' in item: # Maybe structure is diff
                    tokens = dataset.vocab.tokenizer_eng(item['caption'])
                    refs.append(tokens)
                    break
            
            references.append(refs)
            
            # Generate hypothesis (Greedy)
            if dataset.transform:
                img_tensor = dataset.transform(image).unsqueeze(0).to(device)
            else:
                img_tensor = image
            
            features = encoder(img_tensor) # (1, 7, 7, 2048)
            encoder_dim = features.size(3)
            # Flatten
            features = features.view(1, -1, encoder_dim) # (1, 49, 2048)
            
            h, c = decoder.init_hidden_state(features)
            
            # Start token
            word_idx = dataset.vocab.stoi["<SOS>"]
            
            seq = []
            
            for _ in range(20):
                embed = decoder.embedding(torch.tensor([word_idx]).to(device)) # (1, embed_dim)
                
                weights, alpha = decoder.attention(features, h)
                gate = decoder.sigmoid(decoder.f_beta(h))
                weights = gate * weights
                
                h, c = decoder.lstm(torch.cat([embed, weights], dim=1), (h, c))
                
                preds = decoder.fc(decoder.dropout(h))
                word_idx = preds.argmax(dim=1).item()
                
                if word_idx == dataset.vocab.stoi["<EOS>"]:
                    break
                
                word = dataset.vocab.itos[word_idx]
                if word != "<UNK>": # Optional: include unk?
                    seq.append(word)
            
            hypotheses.append(seq)

    bleu4 = corpus_bleu(references, hypotheses)
    print(f"BLEU-4 Score: {bleu4:.4f}")
    return bleu4
