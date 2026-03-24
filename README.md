EnVi Neural Machine Translation (Transformer)

A Neural Machine Translation (NMT) system for English ↔ Vietnamese built
with Transformer architecture.\
This project focuses on training, evaluation, and deployment-ready
translation models using modern deep learning techniques.

------------------------------------------------------------------------

## Features

-   Transformer-based encoder-decoder architecture
-   Multi-head attention (custom implementation)
-   Support for PhoMT & OPUS100 datasets
-   Beam search decoding
-   Training & evaluation pipeline
-   Config-driven experiments

------------------------------------------------------------------------

## Project Structure

    .
    ├── src/
    │   ├── models/        # Transformer, attention, etc.
    │   ├── data/          # Dataset processing
    │   ├── train.py       # Training script
    │   ├── evaluate.py    # Evaluation script
    │   └── utils/         # Helpers
    ├── configs/           # YAML config files
    ├── checkpoints/       # Saved models
    ├── data/              # Raw & processed data
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

##  Installation

``` bash
git clone https://github.com/your-username/envi-nmt-transformer.git
cd envi-nmt-transformer

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

------------------------------------------------------------------------

## Training

``` bash
python src/train.py --config configs/config.yaml
```

------------------------------------------------------------------------

## Evaluation

``` bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```

------------------------------------------------------------------------

## Model

-   Architecture: Transformer (Encoder-Decoder)
-   Attention: Multi-Head Attention
-   Loss: Cross-Entropy
-   Optimization: Adam / AdamW

------------------------------------------------------------------------

## Dataset

-   PhoMT (Vietnamese-English)
-   OPUS100

Preprocessing includes: - Tokenization - Cleaning & normalization -
Padding & batching

------------------------------------------------------------------------

## Results

  Model         BLEU
  ------------- ------
  Transformer   XX.X

*(Update after training)*

------------------------------------------------------------------------

## Future Work

-   Fine-tune pretrained models (PhoMT, mT5)
-   Add inference API (FastAPI)
-   Improve decoding (Top-k, nucleus sampling)
-   Mixed precision training (FP16)


