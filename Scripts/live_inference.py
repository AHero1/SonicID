import os
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision import transforms, models
import sounddevice as sd

# --- CONFIGURATION ---
# Industry Standard: Absolute Paths to prevent "File Not Found" errors
BASE_DIR = '/Users/ahero1/Downloads/SonicID'
MODEL_PATH = os.path.join(BASE_DIR, 'data', 'audio_classifier.pth')
CLASSES_PATH = os.path.join(BASE_DIR, 'data', 'classes.txt')

# Audio Settings (Must match training parameters)
SAMPLE_RATE = 22050
DURATION = 4.0 
SAMPLES = int(SAMPLE_RATE * DURATION)

# Device Configuration (M1/M2 Metal Acceleration)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"⚡ Using High-Performance Device: {device}")

# --- 1. LOAD CLASS NAMES ---
if not os.path.exists(CLASSES_PATH):
    print(f"CRITICAL ERROR: classes.txt not found at {CLASSES_PATH}")
    exit()

with open(CLASSES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
print(f"✔ Loaded {len(class_names)} Classes")

# --- 2. LOAD MODEL ---
# Reconstruct Architecture
model = models.resnet18(weights=None) 
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))

# Load Weights
if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Model not found at {MODEL_PATH}")
    exit()

try:
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval() # Inference Mode
    print("✔ Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 3. PREPROCESSING PIPELINE ---
def process_audio(audio_data):
    """
    Converts raw microphone audio -> Mel-Spectrogram -> Normalized Tensor
    """
    # 1. Generate Mel-Spectrogram
    S = librosa.feature.melspectrogram(
        y=audio_data, 
        sr=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64, 
        fmin=50, 
        fmax=10000
    )
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # 2. Min-Max Normalization (0 to 1)
    S_min, S_max = S_dB.min(), S_dB.max()
    if S_max - S_min > 0:
        S_norm = (S_dB - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_dB)
        
    # 3. Convert to Tensor
    tensor = torch.tensor(S_norm, dtype=torch.float32)
    
    # 4. Reshape for ResNet [Channels, Height, Width]
    tensor = tensor.unsqueeze(0)        # Add channel dim: [1, H, W]
    tensor = tensor.repeat(3, 1, 1)     # Repeat to 3 channels (RGB)
    tensor = tensor.unsqueeze(0)        # Add batch dim: [1, 3, H, W]
    
    # 5. Resize to 128x128 (Match Training Input)
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return preprocess(tensor)

# --- 4. MAIN LOOP ---
def main():
    print("\n" + "="*40)
    print("   SONIC ID: REAL-TIME INFERENCE")
    print("   Press Ctrl+C to Stop")
    print("="*40 + "\n")
    
    try:
        while True:
            print(f"🎤 Listening ({DURATION}s)...")
            
            # Record Audio
            recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            
            # Flatten to 1D array
            audio_data = recording.flatten()
            
            # Preprocess & Move to GPU
            input_tensor = process_audio(audio_data).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Get Result
                prob, idx = torch.max(probabilities, 1)
                predicted_class = class_names[idx.item()]
                confidence = prob.item() * 100
            
            # Clear Output & Print Result
            # Note: \033[F moves cursor up to overwrite previous line for cleaner UI
            print(f"➤ DETECTED: {predicted_class.upper()} ({confidence:.1f}%)")
            print("-" * 30)
            
            # Small buffer to read result
            time.sleep(1.5)

    except KeyboardInterrupt:
        print("\n⛔ Stopping Engine...")
    except Exception as e:
        print(f"\nError: {e}")
        print("NOTE: On macOS, ensure Terminal has Microphone Access.")

if __name__ == "__main__":
    main()