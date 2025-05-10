import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

# ----------------------------
# Configuration & Hyperparams
# ----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 16
NUM_EPOCHS_PHASE1 = 7
NUM_EPOCHS_PHASE2 = 20
PATIENCE = 7
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_BACKBONE = 1e-5
WEIGHT_DECAY = 1e-2
DROPOUT_RATE = 0.5
PRUNE_AMOUNT = 0.3
NUM_FOLDS = 5
NUM_WORKERS = 4

# ----------------------------
# Data Preparation
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
fulldataset = torchvision.datasets.ImageFolder(root='data/fer2013', transform=transform)
labels = [label for _, label in fulldataset.imgs]

# ----------------------------
# Stratified K-Fold Setup
# ----------------------------
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

# ----------------------------
# Model Definition
# ----------------------------
def create_model():
    model = torchvision.models.vgg16(pretrained=True)
    # Replace head
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Sequential(
        nn.Dropout(DROPOUT_RATE),
        nn.Linear(num_features, 2)
    )
    return model.to(DEVICE)

# ----------------------------
# Training Utilities
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(loader, desc='Training', leave=False):
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total
    return running_loss / len(loader.dataset), acc

# ----------------------------
# Fine-Tuning Workflow
# ----------------------------
best_model_state = None
best_f1 = 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(torch.zeros(len(labels)), labels)):
    print(f"\nFold {fold+1}/{NUM_FOLDS}")
    train_set = Subset(fulldataset, train_idx)
    val_set = Subset(fulldataset, val_idx)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = create_model()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, len(labels)/labels.count(1)], device=DEVICE))
    optimizer = optim.AdamW([
        {'params': model.classifier[6].parameters(), 'lr': LEARNING_RATE_HEAD},
        {'params': [p for n, p in model.features.named_parameters()], 'lr': LEARNING_RATE_BACKBONE}
    ], weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=3)

    # Phase 1: Train head only
    for param in model.features.parameters():
        param.requires_grad = False
    for epoch in range(NUM_EPOCHS_PHASE1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_acc)
        print(f"Phase1 Epoch {epoch+1}/{NUM_EPOCHS_PHASE1}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Phase 2: Progressive unfreeze
    for name, param in model.features.named_parameters():
        # Unfreeze top 2 conv blocks
        if 'features.28' in name or 'features.30' in name:
            param.requires_grad = True
    optimizer.param_groups[1]['lr'] = LEARNING_RATE_BACKBONE * 0.1

    epochs_no_improve = 0
    for epoch in range(NUM_EPOCHS_PHASE2):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_acc)
        print(f"Phase2 Epoch {epoch+1}/{NUM_EPOCHS_PHASE2}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping by val_acc
        if val_acc > best_f1:
            best_f1 = val_acc
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered")
                break

# Load best model
model.load_state_dict(best_model_state)

# ----------------------------
# Optimization: Pruning & Quantization
# ----------------------------
# Pruning
parameters_to_prune = []
for module in model.features.modules():
    if isinstance(module, nn.Conv2d):
        parameters_to_prune.append((module, 'weight'))
prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=PRUNE_AMOUNT)
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# Quantization
model_q = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8)

# ----------------------------
# Benchmarking Inference Time
# ----------------------------
def benchmark(model, loader, device, num_batches=50):
    model.eval()
    timings = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(device)
            start = time.time()
            _ = model(images)
            end = time.time()
            timings.append((end - start) / images.size(0))
    avg_ms = sum(timings) / len(timings) * 1000
    print(f"Average inference time: {avg_ms:.2f} ms/image")

# Final test loader (held-out test split)
_, test_idx = next(StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
                   .split(torch.zeros(len(labels)), labels))
test_loader = DataLoader(Subset(fulldataset, test_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Benchmark original model:")
benchmark(model, test_loader, DEVICE)
print("Benchmark pruned & quantized model:")
benchmark(model_q, test_loader, DEVICE)

# Save optimized model
torch.save(model_q.state_dict(), 'vgg16_lightweight.pth')
print("Optimized model saved to vgg16_lightweight.pth")
