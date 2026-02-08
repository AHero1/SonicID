# SonicID
### High-Fidelity Urban Audio Classification via Audio Spectrogram Transformers

SonicID represents a specialized effort to discern and categorize the complex acoustic tapestry of modern urban environments. By leveraging the Audio Spectrogram Transformer (AST) architecture, this system transcends traditional convolutional approaches, applying the self-attention mechanisms typically reserved for language processing to the domain of environmental sound analysis.

The primary objective was to ameliorate the challenges associated with automated noise monitoring—specifically, the capacity to distinguish between benign background noise (like air conditioners) and critical safety events (such as gunshots or sirens) with high precision.

## Project Overview



The cacophony of a city contains distinct acoustic signatures that are often difficult for standard algorithms to untangle. SonicID processes raw audio waveforms by converting them into spectrograms—visual representations of sound—and analyzing them as a sequence of overlapping image patches. This allows the model to capture long-range temporal dependencies, resulting in a system that is both robust to background interference and highly sensitive to distinct auditory events.

The model has been fine-tuned to identify ten specific classes of urban sounds:
* Air Conditioner
* Car Horn
* Children Playing
* Dog Bark
* Drilling
* Engine Idling
* Gunshot
* Jackhammer
* Siren
* Street Music

## Methodology and Architecture

Current state-of-the-art methods often rely on Convolutional Neural Networks (CNNs), which excel at local feature detection but struggle with global context. SonicID mitigates this by utilizing a pure Transformer architecture.

The process involves:
1.  **Preprocessing** — Converting raw audio clips (sampled at 16kHz) into log-mel spectrograms.
2.  **Patch Embedding** — Splitting the spectrogram into a grid of 16x16 patches, similar to how Vision Transformers process images.
3.  **Classification** — Feeding these patches into a pre-trained AST model (roughly 86 million parameters) to predict the probability distribution across the ten target classes.

## Dataset

This project utilizes the UrbanSound8K dataset, a canonical corpus for environmental audio research. It comprises 8,732 labeled sound excerpts (<=4s) from field recordings, organized into ten folds to facilitate rigorous cross-validation.

* **Source:** [UrbanSound8K on Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k?resource=download)
* **Size:** 6.8 GB
* **Format:** .wav

## Performance and Results

The system was evaluated on a held-out test set (Fold 10) comprising approximately 800 unseen samples. The results elucidate the model's efficacy in differentiating between acoustically similar categories.

### Confusion Matrix
The confusion matrix below illustrates the discrepancy between predicted and actual labels. A strong diagonal line indicates high accuracy, while off-diagonal elements represent misclassifications.

![Confusion Matrix](images/confusion_matrix.png)

### Semantic Clustering (t-SNE)
To verify that the model is learning meaningful representations, we projected the high-dimensional internal states (logits) into a 2D plane using t-SNE. The distinct clusters demonstrate that the model has successfully learned to segregate different sound categories in its latent space.

![t-SNE Plot](images/tsne_plot.png)

### Per-Class Accuracy
While the aggregate accuracy is high, it is imperative to examine performance at a granular level. The chart below details the F1-score for each individual class, highlighting areas of particular strength (such as Gunshots) and minor ambiguity.

![Accuracy Chart](images/class_accuracy.png)

## Quick Start (Interactive Demo)

You can assess the model's performance immediately without local installation. We have provided a Colab notebook that automatically downloads the necessary weights and runs inference on the sample files provided in this repository.

[Link to Colab Demo](demo.ipynb)

## Local Installation

For those wishing to replicate the training or run the inference pipeline locally, please follow these steps:

1. Clone the repository
   ```bash
   git clone [https://github.com/YOUR_USERNAME/SonicID.git](https://github.com/YOUR_USERNAME/SonicID.git)
   cd SonicID
