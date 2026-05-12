"""
Grad-CAM Visualization for COVID-19 X-ray Classifier
Generates attention heatmaps overlaid on chest X-ray images.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import torch.nn as nn
from PIL import Image
import random

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data', 'Covid19-dataset')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_resnet50.pth')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'gradcam')
IMG_SIZE   = 224
CLASSES    = ['Covid', 'Normal', 'Viral Pneumonia']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 3)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)


# ─── Grad-CAM ─────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model      = model
        self.gradients  = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam)
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, class_idx, torch.softmax(output, dim=1)[0].detach().cpu().numpy()


def overlay_cam(img_np, cam, alpha=0.5):
    """Overlay heatmap on original image."""
    heatmap = cm.jet(cam)[:, :, :3]          # RGB heatmap
    heatmap = (heatmap * 255).astype(np.uint8)
    img_rgb = (img_np * 255).astype(np.uint8)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)
    return overlay


# ─── Denormalize ──────────────────────────────────────────────────────────────
_mean = np.array([0.485, 0.456, 0.406])
_std  = np.array([0.229, 0.224, 0.225])

def denormalize(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _std + _mean
    return np.clip(img, 0, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_gradcam(n_samples_per_class=2):
    model      = load_model()
    target_layer = model.layer4[-1]          # Last conv layer of ResNet-50
    grad_cam   = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'), transform=transform)

    # Pick n_samples_per_class random samples per class
    class_indices = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(test_dataset):
        class_indices[label].append(idx)

    selected = []
    for cls_idx, idxs in class_indices.items():
        chosen = random.sample(idxs, min(n_samples_per_class, len(idxs)))
        selected.extend([(i, cls_idx) for i in chosen])

    fig, axes = plt.subplots(
        len(selected), 3,
        figsize=(12, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for row, (sample_idx, true_label) in enumerate(selected):
        tensor, _ = test_dataset[sample_idx]
        input_t   = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)

        cam, pred_label, probs = grad_cam.generate(input_t)
        img_np  = denormalize(tensor)
        overlay = overlay_cam(img_np, cam)

        axes[row][0].imshow(img_np); axes[row][0].set_title('Original X-ray')
        axes[row][1].imshow(cam, cmap='jet'); axes[row][1].set_title('Grad-CAM Heatmap')
        axes[row][2].imshow(overlay)
        conf = probs[pred_label] * 100
        axes[row][2].set_title(
            f'Overlay | True: {CLASSES[true_label]}\n'
            f'Pred: {CLASSES[pred_label]} ({conf:.1f}%)')

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle('Grad-CAM: Model Attention on Chest X-rays', fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'gradcam_results.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM visualization → {out_path}")


if __name__ == '__main__':
    run_gradcam(n_samples_per_class=2)
