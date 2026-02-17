import os
import torch
import librosa
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = '/Users/ahero1/Downloads/SonicID'
DATASET_PATH = os.path.join(BASE_DIR, 'UrbanSound8K')
CSV_FILE = os.path.join(DATASET_PATH, 'UrbanSound8K.csv')

# 🔥 UPDATED PATH: Points directly to where you see the 3 files
MODEL_PATH = os.path.join(BASE_DIR, 'ast_model') 
SAVE_IMG_PATH = os.path.join(BASE_DIR, 'confusion_matrix.png')

# Hyperparameters
BATCH_SIZE = 8
TARGET_SR = 16000

# Setup Device (Mac Metal)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using Device: {device}")

# --- 1. DATASET CLASS ---
class UrbanSoundDataset(Dataset):
    def __init__(self, df, feature_extractor):
        self.df = df
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['path']
        label = row['label']

        try:
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        except Exception:
            y = np.zeros(TARGET_SR * 10) # Fallback

        max_length = 16000 * 10 
        if len(y) < max_length:
            y = np.pad(y, (0, max_length - len(y)), mode='constant')
        else:
            y = y[:max_length]

        inputs = self.feature_extractor(y, sampling_rate=TARGET_SR, return_tensors="pt")
        input_values = inputs.input_values.squeeze(0)

        return input_values, torch.tensor(label)

# --- 2. LOAD DATA & MODEL ---
print(f"Loading Model from: {MODEL_PATH}")

try:
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_PATH)
    model = ASTForAudioClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error: {e}")
    print("Please check if 'model.safetensors' is inside the 'ast_model' folder.")
    exit()

# Reconstruct Data Splits
df = pd.read_csv(CSV_FILE)
data_list = []

for _, row in df.iterrows():
    path = os.path.join(DATASET_PATH, f"fold{row['fold']}", row['slice_file_name'])
    if os.path.exists(path):
        data_list.append({'path': path, 'label_name': row['class']})

all_labels = sorted(list(set(d['label_name'] for d in data_list)))
label2id = {label: i for i, label in enumerate(all_labels)}
id2label = {i: label for i, label in enumerate(all_labels)}

full_df = pd.DataFrame(data_list)
full_df['label'] = full_df['label_name'].map(label2id)

# 80/20 Split
train_df = full_df.sample(frac=0.8, random_state=42)
val_df = full_df.drop(train_df.index)

print(f"Validation Files: {len(val_df)}")

# Prepare Loader
val_dataset = UrbanSoundDataset(val_df, feature_extractor)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. INFERENCE LOOP ---
print("Starting Inference...")
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in tqdm(val_loader, desc="Processing"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# --- 4. METRICS & OUTPUT ---
print("\n" + "="*40)
print("       MODEL PERFORMANCE REPORT       ")
print("="*40)

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"Accuracy:      {acc:.4f}")
print(f"Precision:     {precision:.4f}")
print(f"Recall:        {recall:.4f}")
print(f"F1 Score:      {f1:.4f}")
print("-" * 40)

target_names = [id2label[i] for i in range(len(all_labels))]
print(classification_report(y_true, y_pred, target_names=target_names))

print("Generating Confusion Matrix...")
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(SAVE_IMG_PATH)

print(f"\n✅ Success! Saved to: {SAVE_IMG_PATH}")