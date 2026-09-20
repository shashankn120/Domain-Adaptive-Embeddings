# Domain-Adaptive-Embeddings

This project explores a core challenge in medical imaging: getting an AI model to recognise the same object across two very different types of images. In radiology, for ex, a model trained on clean X Rays needs to work on images from a different scanner, lighting condition, or imaging modality even if those images look visually quite different.

## The problem at hand

We simulate this problem using Fashion-MNIST (a dataset of clothing images) as a stand-in for two imaging modalities:

- Domain A — clean, unmodified images, basically a well-calibrated scanner
- Domain B — heavily augmented images with contrast shifts, 45-degree rotation, and Gaussian noise, basically like a different, noisier modality

The model's job is to learn that a boot in Domain A and a boot in Domain B are the same thing despite looking quite different.

## How it works

1. Feature extraction — A CNN maps each image down to a compact 128 number fingerprint (an embedding). Images that represent the same class should produce similar fingerprints, regardless of which domain they came from.

2. Triplet training — We train using triplet loss. For each image in Domain A, we show the model a matching image from Domain B (same class) and a non-matching one (different class). The model is penalised whenever the non-match is closer in embedding space than the match.

3. Validation — After training, we extract all embeddings, compress them into 2D using PCA, and measure Top-1 retrieval accuracy: for each Domain A image, does its nearest neighbour in Domain B space belong to the same class?

## Why it matters

This is a lightweight proxy for domain adaptation — a real problem in medical imaging where data comes from heterogeneous sources. The same principles apply when aligning MRI to CT, or fluoroscopy to standard X-ray.

## Stack

- Python, PyTorch
- scikit-learn (PCA)
- torchvision (Fashion-MNIST)
