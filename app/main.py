import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

# Add src to path to import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.append(src_dir)

from model import Encoder as EncoderRNN
from model import DecoderRNN as DecoderRNN_base
from model_lstm import DecoderRNN as DecoderRNN_lstm
from utils import load_vocab
from config import Config
from inference import caption_image_beam_search

# Custom cached loader
@st.cache_resource
def load_resources(model_type):
    device = Config.DEVICE
    vocab_path = os.path.join(parent_dir, 'models/vocab.pkl') # Use absolute path relative to project root
    
    if not os.path.exists(vocab_path):
        return None, None, None, "Vocab not found"

    vocab = load_vocab(vocab_path)
    
    encoder = EncoderRNN().to(device)
    encoder.eval()
    
    if model_type == 'CNN + RNN':
        decoder = DecoderRNN_base(
            attention_dim=Config.ATTENTION_DIM,
            embed_dim=Config.EMBED_DIM,
            decoder_dim=Config.HIDDEN_DIM,
            vocab_size=len(vocab),
            dropout=Config.DROPOUT  
        ).to(device)
        model_path = os.path.join(parent_dir, 'models/cnn_rnn/best_model.pth')
    else: # CNN + LSTM
        decoder = DecoderRNN_lstm(
            attention_dim=Config.ATTENTION_DIM,
            embed_dim=Config.EMBED_DIM,
            decoder_dim=Config.HIDDEN_DIM,
            vocab_size=len(vocab),
            dropout=Config.DROPOUT
        ).to(device)
        model_path = os.path.join(parent_dir, 'models/cnn_lstm/best_model.pth')
        
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        decoder.load_state_dict(checkpoint['state_dict'])
        decoder.eval()
        return encoder, decoder, vocab, None
    else:
        return None, None, None, f"Model checkpoint not found at {model_path}"

def main():
    st.set_page_config(page_title="The Vibe Reader", page_icon="📸", layout="wide")
    
    st.title("📸 The Vibe Reader")
    st.markdown("Generate captions for your images using **Show, Attend and Tell**.")
    
    # Sidebar for controls
    st.sidebar.header("Configuration")
    model_option = st.sidebar.selectbox(
        "Select Model Architecture",
        ('CNN + RNN', 'CNN + LSTM')
    )
    
    beam_size = st.sidebar.slider("Beam Size", 1, 10, 3)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with col1:
            st.image(image, caption='Uploaded Image', use_container_width=True)
        
        with col2:
            st.subheader("Generated Caption")
            if st.button('Generate Caption'):
                with st.spinner('Thinking...'):
                    encoder, decoder, vocab, error = load_resources(model_option)
                    
                    if error:
                        st.error(error)
                    else:
                        try:
                            # Save temp file for inference script (it expects path)
                            temp_path = os.path.join(current_dir, "temp_image.jpg")
                            image.save(temp_path)
                            
                            caps, alphas = caption_image_beam_search(encoder, decoder, temp_path, vocab, beam_size)
                            
                            # Clean caption
                            ignore = ["<SOS>", "<EOS>", "<PAD>"]
                            sentence = ' '.join([word for word in caps if word not in ignore])
                            
                            st.success(f"**{sentence}**")
                            
                            # Clean up temp file
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            
                        except Exception as e:
                            st.error(f"Error generating caption: {e}")
                            import traceback
                            st.code(traceback.format_exc())

if __name__ == '__main__':
    main()
