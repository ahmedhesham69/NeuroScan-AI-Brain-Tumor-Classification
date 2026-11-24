# 🧠 NeuroScan AI — Brain MRI Tumor Detection & Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end clinical deep learning system for automated detection, multi-class categorization, and radiological explainability (**Grad-CAM**) of intracranial brain neoplasms from Magnetic Resonance Imaging (MRI) scans.

---

## 📌 Overview & Clinical Motivation

Brain tumors represent a major diagnostic challenge requiring rapid and accurate triage. This repository provides:
1. **Multi-class Deep Classification**: Differentiates between **Glioma**, **Meningioma**, **Pituitary Adenoma**, and **Healthy Brain Parenchyma (No Tumor)**.
2. **Medical-Grade Preprocessing**: Extreme-point contour brain cropping via OpenCV to remove non-cerebral artifacts, skull padding, and scanner noise.
3. **Deep Architecture**: Fine-tuned **ResNet-50** residual backbone with custom regularization (BatchNorm, Dropout, Label Smoothing).
4. **Explainable AI (XAI)**: **Grad-CAM (Gradient-Weighted Class Activation Mapping)** saliency heatmaps overlaid on MRI scans to confirm lesion localization.
5. **Interactive Web Application**: Production-ready **Streamlit dashboard** for clinicians and researchers.

---

## 🎯 Diagnostic Target Classes

| Class | Description |
| :--- | :--- |
| **Glioma** | Primary intracranial tumors originating from glial cells (astrocytomas, glioblastomas). |
| **Meningioma** | Typically benign extra-axial neoplasms arising from the arachnoid layer of the meninges. |
| **Pituitary Adenoma** | Endocrine neoplasms in the sella turcica impacting hormone pathways and optical chiasm. |
| **No Tumor** | Normal physiological brain parenchyma without detectable intracranial lesions. |

---

## 🗂️ Project Structure

```text
├── app.py                           # Interactive Streamlit Web Application
├── brain_mri_cancer_detection.ipynb # Research & Training Jupyter Notebook (EDA, Modeling, Evaluation)
├── best_brain_tumor_resnet50.pth    # Trained PyTorch Model Weights
├── Te-aug-me_101.jpg                # Sample MRI Scan for Quick Testing
├── requirements.txt                 # Python Dependencies
├── .gitignore                       # Ignored Dataset & Cache Files
└── README.md                        # Documentation
```

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/ahmedhesham69/NeuroScan-AI-Brain-Tumor-Classification.git
cd NeuroScan-AI-Brain-Tumor-Classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Model Training & Evaluation

The full training pipeline is available in [`brain_mri_cancer_detection.ipynb`](brain_mri_cancer_detection.ipynb):
- **Dataset**: Fetched dynamically via `kagglehub` (`masoudnickparvar/brain-tumor-mri-dataset`, 7,023 scans).
- **Optimizer**: AdamW with Cosine Annealing learning rate schedule.
- **Loss Function**: Cross-Entropy with Label Smoothing (0.05).
- **Metrics**: Sensitivity (Recall), Specificity, Macro/Micro F1-Score, Normalized Confusion Matrix, and Multi-Class One-vs-Rest ROC-AUC curves.

---

## 🩺 Explainable AI (Grad-CAM)
To prevent "black-box" predictions in clinical settings, the pipeline extracts gradient activations from `layer4` of ResNet-50 and superimposes a pseudo-colored saliency map over the brain scan, highlighting the anatomical focus driving the diagnosis.

---

## ⚠️ Clinical Disclaimer
This software is intended strictly for research and academic investigational purposes. It is not an FDA/CE-cleared medical device. All diagnostic findings must be correlated with clinical, histological, and radiological assessment by certified physicians.
