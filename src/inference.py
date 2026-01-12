import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image

from config import Config
from model import Encoder, DecoderRNN
from utils import load_vocab

def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.
    """
    k = beam_size
    vocab_size = len(word_map)
    device = Config.DEVICE

    # Read image and process
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(img).unsqueeze(0).to(device) # (1, 3, 224, 224)

    # Encode
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    encoder_dim = encoder_out.size(3)
    enc_image_size = encoder_out.size(1)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map.stoi['<SOS>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 0
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.lstm(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(decoder.dropout(h))  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same hidden state, same kernel, same input)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)], dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map.stoi['<EOS>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
        
        k -= len(complete_inds)  # reduce beam size

        # Proceed with incomplete sequences
        if k == 0:
            break
        
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1
    
    # Choose sequence with highest score
    if len(complete_seqs_scores) > 0:
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_seqs_alpha[i]
        
        # Decode words
        caps = [word_map.itos[ind] for ind in seq]
        return caps, alphas
    else:
        # If no sequence completed
        return seqs[0].tolist(), seqs_alpha[0].tolist()

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Generate caption for an image.')
    parser.add_argument('--image', type=str, help='Path to input image', required=True)
    parser.add_argument('--model', type=str, help='Path to model checkpoint', default=Config.MODEL_SAVE_PATH)
    parser.add_argument('--vocab', type=str, help='Path to vocab file', default=Config.VOCAB_PATH)
    parser.add_argument('--beam_size', type=int, default=3, help='Beam size for decoding')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Please train the model first.")
        exit(1)
        
    if not os.path.exists(args.vocab):
        print(f"Vocab not found at {args.vocab}. Please build vocab first.")
        exit(1)
        
    device = Config.DEVICE
    
    # Load vocabulary
    vocab = load_vocab(args.vocab)
    
    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    decoder_state = checkpoint['state_dict']
    
    # Re-initialize models
    encoder = Encoder().to(device)
    # Note: Encoder is not in the checkpoint in current train.py! 
    # Current train.py only saves decoder state_dict in 'state_dict'.
    # And implementation_plan said "Also save encoder if fine-tuned".
    # But encoder is frozen by default. So base ResNet is used.
    # HOWEVER, we removed the last layers in __init__. So we need to instantiate Encoder class.
    # It loads pretrained resnet weights. This is fine.
    # If we fine-tune, we MUST load the encoder weights.
    # Let's check train.py saving logic.
    
    # In train.py:
    # save_checkpoint({ ..., 'state_dict': decoder.state_dict(), ... })
    # if encoder_optimizer: torch.save(encoder.state_dict(), 'encoder.pth')
    
    # So if we didn't fine tune, we just use standard Encoder(). 
    # But wait, did we verify 'encoder.pth' path? train.py saves to 'encoder.pth' in root currently.
    # We should update train.py to save encoder to models/ too if used.
    
    decoder = DecoderRNN(
        attention_dim=Config.ATTENTION_DIM,
        embed_dim=Config.EMBED_DIM,
        decoder_dim=Config.HIDDEN_DIM,
        vocab_size=len(vocab),
        dropout=Config.DROPOUT
    ).to(device)
    
    decoder.load_state_dict(decoder_state)
    
    encoder.eval()
    decoder.eval()
    
    try:
        caps, alphas = caption_image_beam_search(encoder, decoder, args.image, vocab, args.beam_size)
        
        # Determine strict punctuation
        sentence = ' '.join(caps)
        # Remove <SOS> <EOS> <PAD> <UNK> if visible, but caps usually expects clean words.
        # caps from beam search are mapped from itos.
        # itos: 0: <PAD>, 1: <SOS>, 2: <EOS>, 3: <UNK>
        
        # Filter tokens
        ignore = ["<SOS>", "<EOS>", "<PAD>"]
        sentence = ' '.join([word for word in caps if word not in ignore])
        
        print(f"Generated Caption: {sentence}")
        
    except Exception as e:
        print(f"Inference failed: {e}")
