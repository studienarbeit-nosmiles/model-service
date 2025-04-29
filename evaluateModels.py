# nur den Klassifikationskopf neu trainieren

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import time

# 1. Einstellungen
BATCH_SIZE = 64
NUM_CLASSES = 7  # FER2013 hat 7 Emotionen
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Datentransformationen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # FER2013 ist ursprünglich Graustufen
    transforms.Resize((224, 224)),  # alle Modelle erwarten 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standardwerte von ImageNet
                         std=[0.229, 0.224, 0.225])
])

# 3. Dataset & DataLoader
train_dataset = datasets.ImageFolder(root='app/fer2013_data', transform=transform)
val_dataset = datasets.ImageFolder(root='app/fer2013_data', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 4. Modell-Loader
def get_model(model_name):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, NUM_CLASSES)
        )
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, NUM_CLASSES)
        )
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        for param in model.features.parameters():
            param.requires_grad = False
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        )
    else:
        raise ValueError('Unsupported model name')
    
    return model.to(DEVICE)

# 5. Trainingsfunktion
def train_model(model, model_name):
    print(f'\nTraining {model_name}...\n')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 6. Evaluation
def evaluate_model(model):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = 100. * correct / total
    print(f'Validation Accuracy: {acc:.2f}%')

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    try:
        auc_score = roc_auc_score(np.eye(NUM_CLASSES)[all_labels], all_probs, multi_class='ovr')
        print(f"AUC Score: {auc_score:.4f}")
    except Exception as e:
        auc_score = None
        print(f"AUC could not be calculated: {e}")

    return acc, auc_score

def measure_inference_time(model):
    model.eval()
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(DEVICE)
            start = time.time()
            _ = model(images)
            end = time.time()

            total_time += (end - start)
            total_images += images.size(0)

    return total_time / total_images  # Sek. pro Bild


# 7. Hauptprogramm
model_names = ['vgg16', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']
results = {}

for model_name in model_names:
    model = get_model(model_name)
    start = time.time()
    train_model(model, model_name)
    acc, auc = evaluate_model(model)
    inference_time = measure_inference_time(model)
    end = time.time()

    results[model_name] = {
        'accuracy': acc,
        'auc': auc,
        'inference_time_s': inference_time,
        'training_time_min': (end - start) / 60
    }


print("\nErgebnisse:")
for model_name, metrics in results.items():
    print(f"{model_name}: "
          f"Accuracy: {metrics['accuracy']:.2f}%, "
          f"AUC: {metrics['auc']:.4f} "
          f"Inference Time: {metrics['inference_time_s']*1000:.2f} ms, "
          f"Training: {metrics['training_time_min']:.2f} min")
