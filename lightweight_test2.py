import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset 
from sklearn.model_selection import train_test_split
import numpy as np

# Ensure reproducibility
torch.manual_seed(42)

# Enable cuDNN autotuner for optimal performance
cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = './app/fer2013_data'
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train and test sets (80/20 stratified)
indices = list(range(len(full_dataset)))
targets = [s[1] for s in full_dataset.samples]
train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=targets, random_state=42)
train_dataset = Subset(full_dataset, train_idx)
test_dataset = Subset(full_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# Load pretrained VGG16 and modify final layer
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training the original model
epochs = 5
train_losses = []
for epoch in range(epochs := epochs if 'epochs' in locals() else 5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}")

# Evaluation function
def evaluate(model, loader, optimized=False):
    model.eval()
    correct = 0
    total = 0
    total_time = 0.0
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            if optimized:
                with autocast():
                    outputs = model(inputs)
            else:
                outputs = model(inputs)
            total_time += time.time() - start
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    infer_time_per_image = total_time / total
    return accuracy, infer_time_per_image

# Evaluate original model
orig_acc, orig_time = evaluate(model, test_loader, optimized=False)
print(f"Original Model - Accuracy: {orig_acc:.4f}, Inference Time/Image: {orig_time:.4f}s")

# Optimize model with torch.compile and mixed-precision
opt_model = torch.compile(model)
# Evaluate optimized model
opt_acc, opt_time = evaluate(opt_model, test_loader, optimized=True)
print(f"Optimized Model - Accuracy: {opt_acc:.4f}, Inference Time/Image: {opt_time:.4f}s")

# Summary metrics
print("Summary of Results:")
print(f"Train Losses: {train_losses}")
print(f"Original - Acc: {orig_acc:.4f}, Time/Image: {orig_time:.4f}s")
print(f"Optimized - Acc: {opt_acc:.4f}, Time/Image: {opt_time:.4f}s")

# Save models
torch.save(model.state_dict(), 'vgg16_original.pth')
torch.save(opt_model.state_dict(), 'vgg16_optimized.pth')
