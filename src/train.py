"""
Multi-class COVID-19 Detection from Chest X-ray Images
Training Script - ResNet50 Transfer Learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import time
import copy

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(__file__), '..', 'data', 'Covid19-dataset')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
MODEL_DIR   = os.path.join(os.path.dirname(__file__), '..', 'models')
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_EPOCHS  = 6
LR          = 1e-4
NUM_CLASSES = 3
CLASSES     = ['Covid', 'Normal', 'Viral Pneumonia']
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR + '/plots', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Data Transforms ──────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def load_data():
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'), transform=train_transforms)
    test_dataset  = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'),  transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train samples : {len(train_dataset)}")
    print(f"Test  samples : {len(test_dataset)}")
    print(f"Classes       : {train_dataset.classes}")
    return train_loader, test_loader, train_dataset, test_dataset


# ─── Model ────────────────────────────────────────────────────────────────────
def build_model():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last two residual blocks
    for layer in [model.layer3, model.layer4, model.fc]:
        for param in layer.parameters():
            param.requires_grad = True
    # Replace classifier head
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model.to(DEVICE)


# ─── Training Loop ────────────────────────────────────────────────────────────
def train_model(model, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   []}
    best_acc   = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()
        # ── train phase
        model.train()
        running_loss = running_correct = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss    += loss.item() * imgs.size(0)
            running_correct += (outputs.argmax(1) == labels).sum().item()

        train_loss = running_loss    / len(train_loader.dataset)
        train_acc  = running_correct / len(train_loader.dataset)

        # ── val phase
        model.eval()
        val_loss = val_correct = 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc   = val_correct / len(test_loader.dataset)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc     = val_acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"TrainLoss={train_loss:.4f} TrainAcc={train_acc:.4f} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} "
              f"[{time.time()-t0:.1f}s]")

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'best_resnet50.pth'))
    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    return model, history


# ─── Evaluation ───────────────────────────────────────────────────────────────
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n=== Classification Report ===")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # ROC-AUC
    labels_bin = label_binarize(all_labels, classes=list(range(NUM_CLASSES)))
    roc_auc    = roc_auc_score(labels_bin, all_probs, multi_class='ovr', average='macro')
    print(f"Macro ROC-AUC: {roc_auc:.4f}")

    return all_labels, all_preds, all_probs, roc_auc


# ─── Plots ────────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],   'r-o', label='Val Loss')
    axes[0].set_title('Loss Curve');  axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],   'r-o', label='Val Acc')
    axes[1].set_title('Accuracy Curve'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'training_curves.png'), dpi=150)
    plt.close()
    print("Saved: training_curves.png")


def plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")


def plot_roc_curves(labels, probs):
    labels_bin = label_binarize(labels, classes=list(range(NUM_CLASSES)))
    plt.figure(figsize=(9, 6))
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    for i, (cls, color) in enumerate(zip(CLASSES, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        auc = roc_auc_score(labels_bin[:, i], probs[:, i])
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{cls} (AUC={auc:.3f})')
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'roc_curves.png'), dpi=150)
    plt.close()
    print("Saved: roc_curves.png")


def plot_class_distribution(train_dataset, test_dataset):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, dataset, title in zip(axes,
                                  [train_dataset, test_dataset],
                                  ['Train Set', 'Test Set']):
        counts = np.bincount(dataset.targets)
        bars   = ax.bar(CLASSES, counts, color=['#e74c3c','#2ecc71','#3498db'])
        ax.set_title(title); ax.set_ylabel('Count')
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 1, str(cnt),
                    ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plots', 'class_distribution.png'), dpi=150)
    plt.close()
    print("Saved: class_distribution.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Using device: {DEVICE}\n")
    train_loader, test_loader, train_ds, test_ds = load_data()
    plot_class_distribution(train_ds, test_ds)

    model = build_model()
    print(f"\nTrainable params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

    model, history = train_model(model, train_loader, test_loader)
    labels, preds, probs, roc_auc = evaluate_model(model, test_loader)

    plot_history(history)
    plot_confusion_matrix(labels, preds)
    plot_roc_curves(labels, probs)

    # Save metrics
    report = classification_report(labels, preds,
                                   target_names=CLASSES, output_dict=True)
    metrics = {'roc_auc': roc_auc, 'classification_report': report}
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nAll done! Check outputs/ folder for plots and metrics.")
