import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import numpy as np
import os
import random

# Hyperparameters
BATCH_SIZE = 32
L = 7  # Sequence length (based on the described features: city, plate, model, day, minute, route, distance)
D = 10  # Input feature dimension (assuming a value; adjust as needed)
NUM_CLASSES = 10
EPOCHS = 10
LEARNING_RATE = 0.001
MODEL_PATH = 'model.pth'  # Path to save/load the model


seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
np.random.seed(seed)
random.seed(seed)

# Custom Dataset (synthetic data for demonstration)
class CustomDataset(Dataset):
    def __init__(self, num_samples=1000):
        # Generate synthetic data
        self.X = torch.zeros(num_samples, L, D)
        for i in range(num_samples):
            # Simulate features: most integer-like, last one float (distance)
            city = torch.randint(0, 100, (D,))  # City info
            plate = torch.randint(0, 1000, (D,))  # Plate
            model = torch.randint(0, 50, (D,))  # Model
            day = torch.randint(0, 7, (D,))  # Day of week
            minute = torch.randint(0, 1440, (D,))  # Minute of day
            route = torch.randint(0, 20, (D,))  # Route
            distance = torch.randn(D) * 100 + 500  # Distance as float
            self.X[i] = torch.stack(
                [city, plate, model, day, minute, route, distance.float()]).float()  # Stack and cast to float for CNN

        self.y = torch.randint(0, NUM_CLASSES, (num_samples,))  # Random labels for 10 classes

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Multi-layer CNN Model for feature extraction and classification
class MultiLayerCNN(nn.Module):
    def __init__(self, in_channels=D, seq_len=L, num_classes=NUM_CLASSES):
        super(MultiLayerCNN, self).__init__()
        # Layer 1: Conv1d (in_channels=D, out=32, kernel=3)
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)

        # Layer 2: Conv1d (32 -> 64, kernel=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)

        # Layer 3: Conv1d (64 -> 128, kernel=3)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        # Removed pool3 to avoid output size 0 for small seq_len

        # Calculate flattened size (after two pools, seq_len // 4)
        flattened_size = 128 * (seq_len // 4)
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: [B, L, D] -> transpose to [B, D, L] for Conv1d
        x = x.transpose(1, 2)  # [B, D, L]

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        # Removed pool3

        # Flatten
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Function to train the model
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / len(train_loader.dataset)
    return avg_loss, accuracy


# Function to validate/test the model
def evaluate(model, loader, criterion, device, is_test=False):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / len(loader.dataset)
    phase = "Test" if is_test else "Validation"
    print(f'{phase} Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset and split: 70% train, 15% val, 15% test
    full_dataset = CustomDataset(num_samples=1000)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, optimizer, criterion
    model = MultiLayerCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    # Track the best validation loss
    best_val_loss = float('inf')

    # Training loop
    # for epoch in range(1, EPOCHS + 1):
    #     train_loss, train_acc = train(model, train_loader, optimizer, criterion, device)
    #     print(f'Epoch {epoch}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
    #     val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    #
    #     # Check if validation loss improved
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), MODEL_PATH)
    #         print(f'Validation loss improved. Model saved to {MODEL_PATH}')

    # For testing: Load the saved model parameters
    # Note: In a real scenario, this could be in a separate script or after restarting the session
    # model = MultiLayerCNN().to(device)  # Re-initialize the model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Model loaded from {MODEL_PATH}')
    else:
        print(f'No saved model found at {MODEL_PATH}. Using untrained model.')

    # Final test
    evaluate(model, test_loader, criterion, device, is_test=True)