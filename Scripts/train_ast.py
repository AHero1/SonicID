import os
import random
import torch
import librosa
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import ASTConfig, ASTFeatureExtractor, ASTForAudioClassification
from torch.optim import AdamW
from tqdm import tqdm
import torchaudio.transforms as T

# --- CONFIGURATION ---
BASE_DIR = '/Users/ahero1/Downloads/SonicID'
DATASET_PATH = os.path.join(BASE_DIR, 'UrbanSound8K')
CSV_FILE = os.path.join(DATASET_PATH, 'UrbanSound8K.csv')
CUSTOM_DATA_DIR = os.path.join(BASE_DIR, 'data', 'spectrograms') # Used to find custom classes

# Hyperparameters
BATCH_SIZE = 8  # AST is heavy, keep batch size lower
EPOCHS = 5
LEARNING_RATE = 1e-5 # Transformers need low LR
TARGET_SR = 16000 # AST is pre-trained on 16kHz

# Setup Device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"⚡ Using Device: {device}")

# --- 1. DATASET SETUP ---
class UrbanSoundDataset(Dataset):
    def __init__(self, metadata, base_path, feature_extractor, augment=False):
        self.metadata = metadata
        self.base_path = base_path
        self.feature_extractor = feature_extractor
        self.augment = augment
        
        # SpecAugment transforms (Time & Freq Masking)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
        self.time_mask = T.TimeMasking(time_mask_param=50)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row['path']
        label = row['label']

        # Load Audio (Resample to 16k for AST)
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        # Normalize duration to 10 seconds (AST standard input)
        # If shorter, pad with zeros. If longer, crop.
        max_length = 16000 * 10 
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)), mode='constant')
        else:
            y = y[:max_length]

        # Extract Features (Spectrogram)
        inputs = self.feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0) # [1024, 128]

        # Apply SpecAugment (Only during training)
        if self.augment:
            # Input is [Time, Freq], transpose to [Freq, Time] for torchaudio
            input_values = input_values.transpose(0, 1).unsqueeze(0) 
            input_values = self.freq_mask(input_values)
            input_values = self.time_mask(input_values)
            # Transpose back
            input_values = input_values.squeeze(0).transpose(0, 1)

        return input_values, torch.tensor(label)

# --- 2. PREPARE DATA ---
print("🔍 Preparing Data Index...")

# Load Standard UrbanSound8K
df = pd.read_csv(CSV_FILE)
data_list = []

# 1. Add Standard Files
for _, row in df.iterrows():
    path = os.path.join(DATASET_PATH, f"fold{row['fold']}", row['slice_file_name'])
    if os.path.exists(path):
        data_list.append({'path': path, 'label_name': row['class']})

# 2. Add Custom Files (if any)
custom_classes = ['z_background_noise', 'z_human_speech']
# Note: We need to find the WAV files for custom data. 
# Since we generated Spectrograms directly in step 5, we might need to record RAW audio to train AST perfectly.
# FOR NOW: We will skip custom classes in this script unless you have the .wav files saved.
# If you want to train AST with custom data, you need to save the .wavs, not just .pngs.

# Create Label Map
all_labels = sorted(list(set(d['label_name'] for d in data_list)))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}

# Convert to DataFrame
full_df = pd.DataFrame(data_list)
full_df['label'] = full_df['label_name'].map(label2id)

print(f"   Classes: {len(all_labels)}")
print(f"   Total Files: {len(full_df)}")

# Train/Val Split (80/20)
train_df = full_df.sample(frac=0.8, random_state=42)
val_df = full_df.drop(train_df.index)

# --- 3. INITIALIZE AST ---
print("🚀 Loading MIT/AudioSet Pretrained Model...")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=len(all_labels),
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True # Allow replacing the final layer
)
model.to(device)

# Datasets
train_dataset = UrbanSoundDataset(train_df, BASE_DIR, feature_extractor, augment=True)
val_dataset = UrbanSoundDataset(val_df, BASE_DIR, feature_extractor, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---
print("\n🔥 Starting AST Fine-Tuning (SpecAugment Enabled)...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress Bar
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward Pass
        outputs = model(inputs).logits
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        
        # Backward Pass
        loss.backward()
        optimizer.step()
        
        # Stats
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs).logits
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    print(f"   ✅ Validation Accuracy: {100 * val_correct / val_total:.2f}%")

# Save
model.save_pretrained(os.path.join(BASE_DIR, 'data', 'ast_model'))
feature_extractor.save_pretrained(os.path.join(BASE_DIR, 'data', 'ast_model'))
print("💾 Model Saved to data/ast_model")