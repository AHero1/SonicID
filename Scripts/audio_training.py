import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_PATH, 'data', 'spectrograms')
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'data', 'audio_classifier.pth')

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
IMG_SIZE = (128, 128)

def main():
    # Device: Use M1 Metal Performance Shaders (MPS) if available
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using Device: {device}")

    # --- 1. DATA PIPELINE ---
    data_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading Dataset...")
    full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes Found: {class_names}")

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Worker configuration (Keep num_workers=2 now that permissions are fixed!)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- 2. MODEL ARCHITECTURE ---
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. TRAINING LOOP ---
    print(f"Starting Training for {EPOCHS} Epochs...")
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
            else:
                model.eval()
                loader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Wrap tqdm to avoid printing issues
            for inputs, labels in tqdm(loader, desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            
            # FIX: Use .float() instead of .double() for M1 Support
            epoch_acc = running_corrects.float() / len(loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Training Complete. Best Val Acc: {best_acc:.4f}')

    # --- 4. SAVE MODEL ---
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    with open(os.path.join(BASE_PATH, 'data', 'classes.txt'), 'w') as f:
        for c in class_names:
            f.write(f"{c}\n")

    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()