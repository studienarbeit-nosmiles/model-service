import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from torchvision import transforms

# Custom Dataset for Smile Detection
class SmileDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Dataset for smile detection.
        Args:
            root_dir (str): Root directory containing 'train' and 'test' folders.
            split (str): 'train' or 'test' to indicate the dataset partition.
            transform: Transformations to apply to the images.
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data = []
        self.labels = []

        # Map 'happy' to 1 (smile) and others to 0 (no smile)
        for label_folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, label_folder)
            label = 1 if label_folder == "happy" else 0

            for img_file in os.listdir(folder_path):
                if img_file.endswith((".png", ".jpg", ".jpeg")):
                    self.data.append(os.path.join(folder_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CNN Model
class SmileCNN(nn.Module):
    def __init__(self):
        super(SmileCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),  # Assuming input size 48x48
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Training Function
def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10):
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device, dtype=torch.float32)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                predicted = (outputs > 0.5).int()
                correct += (predicted == labels.int()).sum().item()
                total += labels.size(0)

        # Metrics
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        val_accuracy = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return history

# Visualization Function
def plot_metrics(history, output_dir="statistics"):
    os.makedirs(output_dir, exist_ok=True)

    # Loss plot
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(output_dir, "loss_plot.png"))

    # Accuracy plot
    plt.figure()
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Validation Accuracy")
    plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))

    print(f"Plots saved in {output_dir}")

# Main Script
def main():
    # Dataset Path
    data_dir = "fer2013_data"

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load Dataset
    dataset = SmileDataset(data_dir, split="train", transform=transform)

    # Split Dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize Model, Loss, Optimizer
    model = SmileCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the Model
    history = train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=10)

    # Save the Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/smile_cnn.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot Metrics
    plot_metrics(history)

if __name__ == "__main__":
    main()