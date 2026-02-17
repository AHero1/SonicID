import os
import torch
import numpy as np
import sounddevice as sd
from transformers import ASTFeatureExtractor, ASTForAudioClassification
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# This matches the path in your screenshot exactly
MODEL_DIR = '/Users/ahero1/Downloads/SonicID/ast_model'
TARGET_SR = 16000 

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"⚡ Using Device: {device}")

# --- LOAD MODEL ---
print(f"🧠 Loading Model from: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    print(f"❌ CRITICAL ERROR: Folder not found at {MODEL_DIR}")
    print("Please check if the folder name is exactly 'ast_model'")
    exit()

try:
    feature_extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    model = ASTForAudioClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    print("✅ Model Loaded Successfully! (97% Accuracy Brain Active)")
except Exception as e:
    print(f"❌ Error loading model files: {e}")
    exit()

# Get the label names from the model config
id2label = model.config.id2label

def predict(audio_data):
    # Normalize to 10 seconds (AST requirement)
    max_length = 16000 * 10
    if len(audio_data) < max_length:
        audio_data = np.pad(audio_data, (0, max_length - len(audio_data)), mode='constant')
    else:
        audio_data = audio_data[:max_length]
        
    inputs = feature_extractor(audio_data, sampling_rate=TARGET_SR, return_tensors="pt")
    input_values = inputs.input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, idx = torch.max(probs, 1)
        
    return id2label[idx.item()], confidence.item()

# --- MAIN LOOP ---
print("\n🎤 Listening for Urban Sounds... (Press Ctrl+C to stop)")
try:
    while True:
        # Record 2 seconds of audio
        print("   Listening...", end="\r")
        recording = sd.rec(int(2 * TARGET_SR), samplerate=TARGET_SR, channels=1)
        sd.wait()
        audio = recording.flatten()
        
        label, conf = predict(audio)
        
        # Only print if confident (> 50%)
        if conf > 0.5:
            # Clear line and print result
            print(f"\r➤ DETECTED: {label.upper()} ({conf*100:.1f}%)      ")
        else:
            print(f"\r...                                          ", end="\r")
        
except KeyboardInterrupt:
    print("\n🛑 Stopped.")