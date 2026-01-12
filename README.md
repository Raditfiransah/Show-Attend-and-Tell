# The Vibe Reader: Show, Attend and Tell on Flickr8k

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A PyTorch implementation of **Show, Attend and Tell: Neural Image Caption Generation with Visual Attention** (Xu et al., ICML 2015). This project uses a **ResNet50** encoder and an **LSTM** decoder with **Bahdanau Attention** to generate descriptive captions for images in the **Flickr8k** dataset.

## 🧠 Architecture

The model consists of three main components:
1.  **Encoder**: A pre-trained **ResNet50** (ImageNet) without the fully connected layers. It extracts spatial feature vectors `(7x7x2048)` from input images.
2.  **Attention**: A **Bahdanau (Additive) Attention** mechanism that computes a weighted sum of encoder features based on the decoder's current hidden state. This allows the model to "look" at specific parts of the image at each time step.
3.  **Decoder**: An **LSTM** network that takes the context vector (from attention) and previous word embedding to predict the next word in the sequence.

## 📂 Directory Structure

```
Show-Attend-and-Tell/
├── app/
│   └── main.py         # Streamlit Web App
├── data/               # Dataset files
├── models/             # Saved artifacts
│   ├── cnn_lstm/       # Checkpoints for CNN+LSTM
│   ├── cnn_rnn/        # Checkpoints for CNN+RNN
│   └── vocab.pkl       # Shared Vocabulary
├── src/                # Source code
│   ├── build_vocab.py  # Vocabulary generation script
│   ├── config.py       # Configuration and hyperparameters
│   ├── dataset.py      # Custom Dataset class
│   ├── inference.py    # Inference script with Beam Search
│   ├── model.py        # Baseline Model (CNN+RNN)
│   ├── model_lstm.py   # Enhanced Model (CNN+LSTM)
│   ├── train.py        # Training loop for CNN+RNN
│   ├── train_cnn_lstm.py # Training loop for CNN+LSTM
│   ├── utils.py        # Utility functions
│   └── verify_setup.py # Environment verification script
├── Dockerfile          # Docker configuration
├── requirements.txt    # Project dependencies
├── run_docker.py       # Helper script for Docker deploy
└── README.md           # This documentation
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 1. Data Preparation

First, ensure your dataset is placed in the `data/` directory. Then, build the vocabulary:

```bash
python src/build_vocab.py
```
This will generate `models/vocab.pkl`.

### 2. Training

You can train two variations of the model.

**Option A: CNN + RNN (Baseline)**
```bash
python src/train.py
```
Checkpoints saved to `models/cnn_rnn/`.

**Option B: CNN + LSTM (Enhanced)**
```bash
python src/train_cnn_lstm.py
```
Checkpoints saved to `models/cnn_lstm/`.

### 3. Web Application (The Vibe Reader)

We provide a **Streamlit** based web interface to easily interact with the trained models.

**Run Locally:**
```bash
python -m streamlit run app/main.py
```

**Run via Docker:**
We have provided a helper script to build and deploy the app container seamlessly.

```bash
python run_docker.py
```
This will build the image `vibe-reader-app` and launch it on `http://localhost:8501`.

### 4. CLI Inference

Generate captions for new images using the command line:

```bash
python src/inference.py --image path/to/image.jpg --model models/cnn_lstm/best_model.pth --beam_size 5
```

## 🛠 Configuration

You can adjust hyperparameters in `src/config.py`:

- `BATCH_SIZE`: 32
- `EMBED_DIM`: 256
- `HIDDEN_DIM`: 512
- `ATTENTION_DIM`: 256
- `LEARNING_RATE`: 1e-4

## 📜 License

MIT License.
