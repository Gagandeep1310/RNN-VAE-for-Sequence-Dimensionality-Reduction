# RNN-VAE on Penn Treebank (PTB)

This project implements a **Recurrent Neural Network Variational Autoencoder (RNN-VAE)** for dimensionality reduction on the **Penn Treebank (PTB)** dataset using PyTorch.

## Features
- GRU-based Encoder/Decoder
- Variational Autoencoder with latent space
- Dimensionality reduction of text sequences
- Visualization with t-SNE

## Run Training
```bash
python train.py --epochs 10 --latent_dim 32

