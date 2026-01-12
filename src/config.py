import torch
import os

class Config:
    # Data parameters
    # Data parameters
    # DATA_DIR moved to bottom for env var support
    # Use absolute path if needed, but relative path is good for portability within src
    # Assuming we run from src/
    
    # Image parameters
    IMAGE_SIZE = 224
    
    # Model parameters
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    ATTENTION_DIM = 256
    ENCODER_DIM = 2048  # ResNet50 output
    DROPOUT = 0.5
    
    # Training parameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 20
    TEACHER_FORCING_RATIO = 0.5 # Optional if we want to schedule it
    
    # Tokenizer parameters
    vocab_size = None # Will be set after loading vocabulary
    
    # Device
    # Use CUDA if available, but for Docker deployment often CPU is safer unless GPU is configured
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths - Use environment variables for flexibility in Docker
    MODEL_DIR = os.getenv('MODEL_DIR', 'models/cnn_lstm')
    MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'best_model.pth')
    VOCAB_PATH = os.getenv('VOCAB_PATH', 'models/cnn_lstm/vocab.pkl')
    DATA_DIR = os.getenv('DATA_DIR', './data')
