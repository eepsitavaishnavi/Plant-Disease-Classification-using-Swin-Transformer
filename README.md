# Plant Disease Classification using Swin Transformer

Automated plant disease detection from leaf images using a **Swin Transformer (Swin-T)** deep learning model.
This project demonstrates the use of transformer-based architectures for fine-grained image classification in agriculture, leveraging the shifted-window local attention mechanism to identify subtle disease patterns across **38 plant disease / healthy classes**.

Based on the paper: *Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021.*

---

## Team Members

| Name | Roll Number |
|---|---|
| Janapareddy Vidya Varshini | 230041013 |
| Korubilli Vaishnavi | 230041016 |
| Mullapudi Namaswi | 230041023 |

---

## Problem Statement

Develop an automated system to classify plant leaf images into their correct crop and disease category using the Swin Transformer model.
The goal is to build an accurate and efficient classifier capable of recognizing subtle visual differences between diseases, with minimal manual effort.

Plant disease identification is critical for protecting crop yield and food security.
Manual inspection is time-consuming, requires expert knowledge, and is prone to human error - hence, an automated deep-learning solution is proposed.

---

## Dataset Structure

**Dataset:** [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — `color` folder

### Dataset Info
- Total images: ~54,305
- Number of classes: 38 (crop + disease / healthy combinations)
- Crop species: 14 (Apple, Tomato, Corn, Grape, Potato, Orange, etc.)
- Original image size: 256×256 px (resized to 224×224)
- Data split (random, seed = 42, per-class):
  - Train: 70% (~38,000 images)
  - Validation: 15% (~8,100 images)
  - Test: 15% (8,179 images)

---

## ⚙️ Data Preprocessing & Augmentation

Implemented with `torchvision` transforms and `timm` Mixup/CutMix.

**Training pipeline**
- RandomResizedCrop → 224×224 (scale 0.67–1.0)
- Random horizontal flip
- ColorJitter (brightness, contrast, saturation 0.4; hue 0.1)
- RandAugment (2 ops, magnitude 9)
- Normalize using ImageNet mean & std
- RandomErasing (p = 0.25)
- **Mixup (α = 0.8) and CutMix (α = 1.0)** applied per batch, with 50% switch probability and label smoothing 0.1

**Validation / Test pipeline** (no augmentation)
- Resize → 256, CenterCrop → 224×224
- Convert to tensor (C, H, W)
- Normalize using ImageNet mean & std

---

## Model Architecture — Swin Transformer (Swin-T)

Swin Transformer addresses the limitations of CNNs (limited long-range context) and standard Vision Transformers (quadratic cost and single-resolution features).

### Key Components

**Shifted Window Self-Attention (SW-MSA)**
- Computes self-attention inside local, non-overlapping windows (7×7 patches), giving **linear complexity** with image size.
- Windows are **shifted** between consecutive layers so neighbouring windows can exchange information.
- Efficient implementation using cyclic shift + masking.

**Hierarchical Feature Maps (Patch Merging)**
- Image is split into 4×4 patches, then patches are progressively merged across 4 stages.
- Produces multi-scale features: C → 2C → 4C → 8C channels.

**Relative Position Bias**
- A learnable bias added to every attention head, improving performance over absolute position embeddings.

### Swin-T Configuration

| Parameter | Value |
|---|---|
| Embedding dim (C) | 96 |
| Layers per stage | {2, 2, 6, 2} |
| Window size | 7×7 |
| Parameters | ~28M |
| Pretrained weights | ImageNet-1K (via `timm`) |
| Classification head | Replaced with 38 outputs |

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Framework | PyTorch + timm |
| Device | CUDA (GPU) if available |
| Image Size | 224×224 |
| Epochs | 10 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Learning Rate | 0.0001 |
| Weight Decay | 0.05 |
| Scheduler | Cosine decay with 5-epoch linear warm-up |
| Gradient Clipping | Max norm = 1.0 |
| Loss | Soft-Target Cross-Entropy (train, for Mixup/CutMix) · Cross-Entropy with label smoothing 0.1 (val) |
| Metrics | Top-1 / Top-5 Accuracy, Precision, Recall, F1-Score, Confusion Matrix |

---

## Results & Observations

- Best model achieved at the **10th epoch** (validation accuracy 99.53%)
- Evaluated on the held-out test set (8,179 images)
- High accuracy and F1-score confirm strong classification quality
- Model generalized well on unseen leaf images; all 38 classes scored ≥ 94.9% accuracy

| Metric | Score |
|---|---|
| Test Top-1 Accuracy | **99.58%** |
| Test Top-5 Accuracy | **100.00%** |
| Macro F1-Score | **0.9938** |
| Weighted F1-Score | **0.9959** |
| Training time | ~104 minutes (Colab GPU) |

**Per-class observations**
- Most classes reach 100% test accuracy (e.g., Apple Scab, Tomato Mosaic Virus, Citrus Greening).
- Hardest classes: Corn *Cercospora / Gray leaf spot* (94.9%), Corn *Northern Leaf Blight* (95.3%), Tomato *Spider mites* (96.8%) — visually similar lesion patterns.

| Metric | Description |
|---|---|
| Accuracy (Top-1) | Correct predictions / total predictions |
| Precision | TP / (TP + FP) — avoids false disease alarms |
| Recall | TP / (TP + FN) — measures detection completeness |
| F1-Score | 2·P·R / (P + R) — balance of precision and recall |
| Confusion Matrix | Shows class-specific errors between similar diseases |

---

## Web Application Deployment

A Flask-based web interface was built to make the classification model accessible.

### Features
- Upload a leaf image (`.jpg`, `.png`, `.webp`) or drag & drop
- Capture a leaf photo directly using the **webcam**
- Model predicts the plant and disease condition
- Displays prediction, confidence, status (Healthy / Diseased) and a short summary
- Shows the top-5 predictions with confidence bars
- Responsive green-themed UI built with HTML, CSS, JavaScript
- Backend uses Flask and the fine-tuned Swin-T model

---

## Project Structure

```
Plant-Disease-Swin/
├── app.py                 # Flask backend (loads swin_best.pth)
├── swin_transformer.ipynb          # Training & evaluation notebook
├── swin_best.pth          # Trained model (download separately, not in repo)
├── requirements.txt
└── README.md
```

---

## Run the Web App

```bash
# Clone the repository
git clone https://github.com/eepsitavaishnavi/Plant-Disease-Classification-using-Swin-Transformer.git
cd <Plant-Disease-Classification-using-Swin-Transformer>

# Create virtual environment
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows

# Install dependencies
pip install -r requirements.txt

# Place swin_best.pth next to app.py, then run the Flask app
python app.py
```
