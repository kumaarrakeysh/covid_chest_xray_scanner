"""
Streamlit Web App for COVID-19 X-ray Classification
Run: streamlit run app/app.py
"""

import os
import sys
import io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import streamlit as st

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_resnet50.pth')
IMG_SIZE   = 224
CLASSES    = ['Covid', 'Normal', 'Viral Pneumonia']
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_COLORS = {'Covid': '#e74c3c', 'Normal': '#2ecc71', 'Viral Pneumonia': '#3498db'}

_mean = np.array([0.485, 0.456, 0.406])
_std  = np.array([0.229, 0.224, 0.225])

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std)
])


# ─── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource
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


# ─── Grad-CAM helper ──────────────────────────────────────────────────────────
class GradCAMHelper:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        model.layer4[-1].register_forward_hook(self._fwd_hook)
        model.layer4[-1].register_backward_hook(self._bwd_hook)

    def _fwd_hook(self, m, i, o):
        self.activations = o.detach()

    def _bwd_hook(self, m, gi, go):
        self.gradients = go[0].detach()

    def compute(self, tensor):
        inp    = tensor.unsqueeze(0).to(DEVICE).requires_grad_(True)
        self.model.zero_grad()
        out    = self.model(inp)
        probs  = torch.softmax(out, dim=1)[0].detach().cpu().numpy()
        pred   = int(out.argmax(1))
        out[0, pred].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam     = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        return cam, pred, probs


def overlay_cam(img_np, cam, alpha=0.45):
    heatmap = mpl_cm.jet(cam)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    img_rgb = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
    return cv2.addWeighted(img_rgb, 1 - alpha, heatmap, alpha, 0)


def denorm(tensor):
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return img * _std + _mean


# ─── App UI ───────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="COVID-19 X-ray Classifier",
                       page_icon="🫁", layout="wide")

    st.title("🫁 COVID-19 Chest X-ray Classifier")
    st.markdown(
        "Upload a chest X-ray image to classify it as **COVID-19**, "
        "**Viral Pneumonia**, or **Normal** using a fine-tuned ResNet-50 model "
        "with Grad-CAM explainability."
    )

    # Check model exists
    if not os.path.exists(MODEL_PATH):
        st.error(
            "⚠️ Model weights not found. "
            "Please train the model first by running `python src/train.py`."
        )
        st.stop()

    model    = load_model()
    grad_cam = GradCAMHelper(model)

    uploaded = st.file_uploader(
        "Upload Chest X-ray (JPG/PNG)", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        tensor = transform(image)

        with st.spinner("Analyzing X-ray…"):
            cam, pred, probs = grad_cam.compute(tensor)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original X-ray")
            st.image(image.resize((IMG_SIZE, IMG_SIZE)), use_column_width=True)

        img_np  = denorm(tensor)
        overlay = overlay_cam(img_np, cam)

        with col2:
            st.subheader("Grad-CAM Heatmap")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(cam, cmap='jet'); ax.axis('off')
            st.pyplot(fig, use_container_width=True)

        with col3:
            st.subheader("Overlay")
            st.image(overlay, use_column_width=True)

        # ── Prediction card
        pred_cls   = CLASSES[pred]
        pred_color = CLASS_COLORS[pred_cls]
        conf       = probs[pred] * 100

        st.markdown("---")
        st.markdown(
            f"<h2 style='color:{pred_color}; text-align:center;'>"
            f"Prediction: {pred_cls} ({conf:.1f}% confidence)</h2>",
            unsafe_allow_html=True
        )

        # ── Probability bars
        st.subheader("Class Probabilities")
        for cls, prob in zip(CLASSES, probs):
            color = CLASS_COLORS[cls]
            st.markdown(
                f"**{cls}:** {prob*100:.1f}%  "
                f"<div style='background:{color};width:{prob*100:.1f}%;"
                f"height:12px;border-radius:6px'></div>",
                unsafe_allow_html=True
            )
            st.write("")

        st.info(
            "⚠️ This tool is for research/educational purposes only. "
            "Always consult a licensed medical professional for diagnosis."
        )

    else:
        st.info("👆 Upload an X-ray image to get started.")

    with st.sidebar:
        st.header("About")
        st.write("Model: ResNet-50 (fine-tuned)")
        st.write("Classes: COVID-19 | Normal | Viral Pneumonia")
        st.write("Explainability: Grad-CAM")
        st.write("Dataset: Pranav Raikokte (Kaggle)")


if __name__ == '__main__':
    main()
