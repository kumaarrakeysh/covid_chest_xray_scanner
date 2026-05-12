# 🫁 Multi-class COVID-19 Detection from Chest X-ray Images

> **Domain:** Healthcare | Medical Imaging | AI for Diagnostics  
> **Dataset:** [COVID-19 Image Dataset by Pranav Raikokte (Kaggle)](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

---

## Project Overview

A deep learning pipeline that classifies chest X-ray images into three categories:
- **COVID-19**
- **Viral Pneumonia**
- **Normal**

Built with ResNet-50 transfer learning, Grad-CAM explainability, and a Streamlit web app.

---

## Project Structure

```
covid19_xray_project/
├── data/
│   └── Covid19-dataset/          ← Place Kaggle dataset here
│       ├── train/
│       │   ├── Covid/
│       │   ├── Normal/
│       │   └── Viral Pneumonia/
│       └── test/
│           ├── Covid/
│           ├── Normal/
│           └── Viral Pneumonia/
├── notebooks/
│   └── COVID19_Detection.ipynb   ← Full EDA + Training notebook
├── src/
│   ├── train.py                  ← Training script
│   └── gradcam.py                ← Grad-CAM visualization
├── models/
│   └── best_resnet50.pth         ← Saved after training
├── outputs/
│   ├── plots/                    ← Training curves, confusion matrix, ROC
│   └── gradcam/                  ← Grad-CAM overlay images
├── app/
│   └── app.py                    ← Streamlit web app
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place your dataset
Download from Kaggle and extract so the folder looks like:
```
data/Covid19-dataset/train/Covid/       (images)
data/Covid19-dataset/train/Normal/      (images)
data/Covid19-dataset/train/Viral Pneumonia/ (images)
data/Covid19-dataset/test/...           (same structure)
```

### 3. Train the model
```bash
python src/train.py
```
This saves `models/best_resnet50.pth` and plots to `outputs/plots/`.

### 4. Generate Grad-CAM visualizations
```bash
python src/gradcam.py
```

### 5. Launch the web app
```bash
streamlit run app/app.py
```

### 6. Or use the Jupyter notebook
```bash
jupyter notebook notebooks/COVID19_Detection.ipynb
```

---

## Docker Deployment

```bash
# Build image
docker build -t covid19-xray .

# Run container
docker run -p 8501:8501 covid19-xray
```
Open http://localhost:8501 in your browser.

---

## Methodology

| Step | Details |
|---|---|
| Preprocessing | Resize to 224×224, normalize with ImageNet stats |
| Augmentation | Random flip, rotation ±15°, brightness/contrast jitter, affine shift |
| Model | ResNet-50 pretrained on ImageNet; last 2 blocks + head fine-tuned |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-4) with ReduceLROnPlateau scheduler |
| Explainability | Grad-CAM on layer4 |

---

## Evaluation Metrics

- **Accuracy** — overall and per-class
- **Precision / Recall / F1-Score** — especially for COVID-19 class
- **ROC-AUC** — macro averaged (One-vs-Rest)
- **Confusion Matrix** — misclassification analysis
- **Grad-CAM** — qualitative visual explainability

---

## Business Use Cases

| Use Case | Description |
|---|---|
| Clinical Support | Rapid triage assistance for radiologists |
| Remote Healthcare | Diagnostic aid in low-resource settings |
| Public Health Screening | Large-scale automated screening |
| Education | Training tool for medical students |

---

## Disclaimer

⚠️ This project is for **research and educational purposes only**.  
Do **not** use for actual clinical diagnosis without proper medical validation.

---

## References

- He et al. (2016). Deep Residual Learning for Image Recognition.
- Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks.
- Dataset: Pranav Raikokte, Kaggle COVID-19 Image Dataset.
