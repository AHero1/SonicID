import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Base path (Adjust if needed, using relative path for safety)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_PATH, 'UrbanSound8K')
OUTPUT_PATH = os.path.join(BASE_PATH, 'data', 'spectrograms')
CSV_FILE = os.path.join(DATASET_PATH, 'UrbanSound8K.csv')

# Audio Settings
TARGET_FS = 22050
DURATION = 4.0
SAMPLES = int(TARGET_FS * DURATION)

# Create Output Directory
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

print(f"Loading Metadata from: {CSV_FILE}")
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    print(f"ERROR: CSV not found at {CSV_FILE}")
    exit()

print(f"Found {len(df)} files. Starting processing...")

# --- MAIN LOOP ---
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        filename = row['slice_file_name']
        fold = row['fold']
        class_name = row['class']
        
        # Construct Path
        audio_path = os.path.join(DATASET_PATH, f"fold{fold}", filename)
        
        # Skip if file doesn't exist
        if not os.path.exists(audio_path):
            continue
            
        # 1. Load Audio (Librosa handles M1 codecs perfectly)
        y, sr = librosa.load(audio_path, sr=TARGET_FS, mono=True)
        
        # 2. Pad or Crop to 4 seconds
        if len(y) < SAMPLES:
            padding = SAMPLES - len(y)
            y = np.pad(y, (0, padding), mode='constant')
        else:
            y = y[:SAMPLES]
            
        # 3. Generate Mel-Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=64, fmin=50, fmax=10000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # 4. Save as Image
        # Create Class Folder
        class_folder = os.path.join(OUTPUT_PATH, class_name)
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)
            
        save_name = filename.replace('.wav', '.png')
        save_path = os.path.join(class_folder, save_name)
        
        # Save output without axes/borders (Pure Image)
        plt.figure(figsize=(1, 1), dpi=100) # Small size is fine for CNN
        plt.axis('off')
        plt.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
        plt.imshow(S_dB, aspect='auto', origin='lower', cmap='magma')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Processing Complete. Check data/spectrograms folder.")