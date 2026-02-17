# SonicID
### High-Fidelity Urban Audio Classification via Audio Spectrogram Transformers

SonicID represents a specialized effort to discern and categorize the complex acoustic tapestry of modern urban environments. By leveraging the Audio Spectrogram Transformer (AST) architecture, this system transcends traditional convolutional approaches, applying the self-attention mechanisms typically reserved for language processing to the domain of environmental sound analysis.

The primary objective was to ameliorate the challenges associated with automated noise monitoring specifically, the capacity to distinguish between benign background noise (like air conditioners) and critical safety events (such as gunshots or sirens) with high precision.

## Project Overview

The cacophony of a city contains distinct acoustic signatures that are often difficult for standard algorithms to untangle. SonicID processes raw audio waveforms by converting them into spectrograms visual representations of sound and analyzing them as a sequence of overlapping image patches. This allows the model to capture long-range temporal dependencies, resulting in a system that is both robust to background interference and highly sensitive to distinct auditory events.

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

The system was evaluated on the full UrbanSound8K dataset, processing over 8,000 samples to ensure statistical significance. The results elucidate the model's efficacy in differentiating between acoustically similar categories.

### 1. Quantitative Metrics
The model achieved state-of-the-art performance across all evaluation metrics.

**Overall Performance:**
| Metric | Score |
| :--- | :--- |
| **Accuracy** | **97.65%** |
| **F1 Score** | **97.65%** |
| **Precision** | **97.68%** |
| **Recall** | **97.65%** |

**Class-wise Breakdown:**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Air Conditioner** | 0.95 | 1.00 | 0.97 | 173 |
| **Car Horn** | 0.99 | 0.98 | 0.98 | 98 |
| **Children Playing** | 0.99 | 0.98 | 0.99 | 187 |
| **Dog Bark** | 0.99 | 0.97 | 0.98 | 215 |
| **Drilling** | 0.97 | 0.92 | 0.95 | 211 |
| **Engine Idling** | 1.00 | 0.99 | 1.00 | 218 |
| **Gun Shot** | 0.98 | 1.00 | 0.99 | 61 |
| **Jackhammer** | 0.94 | 0.96 | 0.95 | 182 |
| **Siren** | 0.99 | 0.98 | 0.99 | 196 |
| **Street Music** | 0.96 | 0.99 | 0.98 | 205 |

### 2. Signal Processing & Feature Extraction
Before classification, the system converts time-domain waveforms into frequency-domain features. The visualization below demonstrates this transformation on a sample input (e.g., a dog bark). The AST model sees the Mel-Spectrogram (bottom), allowing it to identify patterns in frequency modulation that are invisible in the raw waveform (top).

<img width="1200" height="600" alt="signal_processing_view" src="https://github.com/user-attachments/assets/2672ba2e-85b7-4ce9-b3b1-656dae71f373" />

### 3. Semantic Clustering (t-SNE)
To verify that the model is learning meaningful representations, we projected the high-dimensional internal states (logits) into a 2D plane using t-SNE. The distinct, well-separated clusters demonstrate that the model has successfully learned to segregate different sound categories in its latent space with minimal overlap.

<img width="1000" height="800" alt="tsne_plot" src="https://github.com/user-attachments/assets/3c238f68-ab78-4f1f-9ab6-ed1b854e0b47" />

### 4. Model Confidence Analysis
A robust model should not only be accurate but also certain in its predictions. The density plot below shows the distribution of confidence scores for all correct predictions. The extreme skew towards 1.0 indicates that SonicID rarely "guesses"; when it classifies a sound, it does so with near-absolute certainty.

<img width="1000" height="500" alt="confidence_dist" src="https://github.com/user-attachments/assets/12a07757-559e-4836-9ef3-4429b9577c40" />

### 5. Confusion Matrix
The confusion matrix provides a granular view of classification performance. A strong diagonal line indicates high accuracy, while off-diagonal elements expose specific acoustic confusions (e.g., distinguishing between 'Drilling' and 'Jackhammer').

<img width="1200" height="1000" alt="confusion_matrix" src="https://github.com/user-attachments/assets/9c1e58a0-869a-4572-9370-83f15297297d" />

## Quick Start (Interactive Demo)

You can assess the model's performance immediately without local installation. We have provided a Colab notebook that automatically downloads the necessary weights and runs inference on the sample files provided in this repository.

[[Link to Colab Demo](demo.ipynb)](https://colab.research.google.com/drive/1RRVLgcP-9QIa05Lq0kAzoqsidog5PE0x?usp=sharing)

### Run Inference Script
Below is the core logic used in our notebook to test sample audio files. This script automatically fetches the model weights from our cloud storage and runs predictions on the `Test_samples` folder.

```python
import os
import zipfile
import numpy as np
import librosa
import torch
import gdown
import IPython.display as ipd
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import warnings

warnings.filterwarnings("ignore")

MODEL_FILE_ID = '12eDV7YaP1ipTJwhLD9Q8f-n9I6TeObSl'
SAMPLES_FILE_ID = '14ojS_nrx1-S2hwsiQrOJH4q4E38cCRue'
MODEL_DIR = "ast_model"
TEST_DIR = "Test_samples"

def download_and_extract(file_id, output_dir):
    if not os.path.exists(output_dir):
        print(f"Downloading {output_dir}...")
        url = f'[https://drive.google.com/uc?id=](https://drive.google.com/uc?id=){file_id}'
        output_zip = f"{output_dir}.zip"
        gdown.download(url, output_zip, quiet=False)

        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(output_zip)

def run_sonic_id():
    # Select Device (GPU/MPS/CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Running on: {device}")

    # Download Assets
    download_and_extract(MODEL_FILE_ID, MODEL_DIR)
    download_and_extract(SAMPLES_FILE_ID, TEST_DIR)

    if not os.path.exists(MODEL_DIR) or not os.path.exists(TEST_DIR):
        print("Error: Required files missing.")
        return

    # Load Model
    try:
        feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
        model = ASTForAudioClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run Inference on Test Samples
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.wav', '.mp3'))])

    print("\n" + "="*65)
    print(f"{'PREDICTION':<25} | {'CONFIDENCE'} | {'AUDIO'}")
    print("="*65)

    for filename in files:
        file_path = os.path.join(TEST_DIR, filename)

        try:
            # Preprocess Audio
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            max_len = 16000 * 10
            if len(y) < max_len:
                y = np.pad(y, (0, max_len - len(y)), mode='constant')
            else:
                y = y[:max_len]

            inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt")
            input_values = inputs.input_values.to(device)

            # Predict
            with torch.no_grad():
                logits = model(input_values).logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                conf, idx = torch.max(probs, 1)

            label = model.config.id2label[idx.item()].upper()

            print(f"File: {filename}")
            print(f"{label:<25} | {conf.item()*100:.1f}%")
            # ipd.display(ipd.Audio(file_path)) # Uncomment in Notebook
            print("-" * 65)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    run_sonic_id()
```
### 1. Clone the repository
git clone [https://github.com/YOUR_USERNAME/SonicID.git](https://github.com/YOUR_USERNAME/SonicID.git)
cd SonicID
### 2. Install dependencies
```
pip install -r requirements.txt
