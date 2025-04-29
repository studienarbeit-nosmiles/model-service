import os
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

# 1. Einstellungen
BATCH_SIZE = 64
NUM_CLASSES = 2  # Binäre Klassifikation: laugh vs. noLaugh
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = 'app/fer2013_data'  # Pfad zum Datensatz

# 2. Datentransformationen
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Konvertiere Graustufenbilder zu 3 Kanälen
    transforms.Resize((224, 224)),  # Einheitliche Bildgröße
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet-Normalisierung
                         std=[0.229, 0.224, 0.225])
])

# 3. Dataset & DataLoader mit Berücksichtigung unbalancierter Daten
dataset = datasets.ImageFolder(root=DATA_PATH, transform=transform)
class_counts = np.bincount(dataset.targets)
class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[t] for t in dataset.targets]

# Aufteilen in Trainings- und Validierungsdatensätze (80% Training, 20% Validierung)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# WeightedRandomSampler für den Training DataLoader
train_sampler = WeightedRandomSampler(weights=sample_weights[:train_size], num_samples=train_size, replacement=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
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

# 5. Trainingsfunktion mit Weighted Loss
def train_model(model, model_name):
    print(f'\nTraining {model_name}...\n')
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

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

# 6. Evaluation mit allen gewünschten Metriken
def evaluate_model(model):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, [p[1] for p in all_probs])
    except Exception as e:
        auc = None
        print(f"AUC could not be calculated: {e}")

    print(f'Validation Accuracy: {acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    if auc is not None:
        print(f'AUC Score: {auc:.4f}')

    return acc, precision, recall, f1, auc

# 7. Inferenzzeit pro Bild messen
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

    return total_time / total_images  # Sekunden pro Bild

# 8. Hauptprogramm
model_names = ['vgg16', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']
results = {}

for model_name in model_names:
    model = get_model(model_name)
    start = time.time()
    train_model(model, model_name)
    acc, precision, recall, f1, auc = evaluate_model(model)
    inference_time = measure_inference_time(model)
    end = time.time()

    results[model_name] = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'inference_time_s': inference_time,
        'training_time_min': (end - start) / 60
    }

print("\nErgebnisse:")
for model_name, metrics in results.items():
    print(f"{model_name}: "
          f"Accuracy: {metrics['accuracy']:.4f}, "
          f"Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1 Score: {metrics['f1_score']:.4f}, "
          f"AUC: {metrics['auc']:.4f} "
          f"Inference Time: {metrics['inference_time_s']*1000:.2f} ms, "
          f"Training: {metrics['training_time_min']:.2f} min")


'''
Modell | Accuracy | Precision | Recall | F1 Score | AUC | Inf. Time | Training
VGG16 | 88.38 % | 0.9188 | 0.9273 | 0.9230 | 0.9164 | 0.06 ms | 42.86 min
ResNet50 | 81.36 % | 0.8511 | 0.9114 | 0.8802 | 0.8245 | 0.17 ms | 27.28 min
MobileNetV2 | 74.42 % | 0.8918 | 0.7506 | 0.8151 | 0.8092 | 0.15 ms | 15.07 min
EfficientNetB0 | 74.91 % | 0.8991 | 0.7502 | 0.8179 | 0.8229 | 0.23 ms | 17.11 min
'''