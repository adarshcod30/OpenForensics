# OpenForensics — Deepfake Detection Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11-orange?logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-red?logo=streamlit&logoColor=white)

A comprehensive deep learning pipeline and interactive web dashboard for detecting deepfake (forged) face images, built around the "OpenForensics" dataset framework.

## 🚀 Overview

This repository contains a professional-grade AI solution for forensic image analysis. It leverages a custom **ResNet50 + VGG16 Ensemble** to classify images as either "Real" or "Fake". 

Key features include:
- **Robust Model Pipeline:** Scripts for training from scratch and fine-tuning existing models.
- **Explainable AI (XAI):** A Streamlit dashboard that generates **Grad-CAM heatmaps** to visually explain which pixels the AI focused on to make its forgery prediction.
- **Extensive Evaluation:** Built-in scripts to generate ROC curves, Precision-Recall curves, and classification reports.

## 🗂 Project Structure

```text
OpenForensics/
├── app/
│   └── app_streamlit.py       # Streamlit web dashboard
├── scripts/
│   └── predict_image.py       # CLI tool for single-image inference
├── src/
│   ├── dataset/               # Data loading and augmentation pipelines
│   ├── model/                 # Ensemble model definition and fine-tuning logic
│   └── training/              # Scripts for training and evaluation
├── .streamlit/
│   └── config.toml            # UI theme configuration
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (You are here!)
```

## 🛠 Installation & Setup

Because deep learning dependencies match specific hardware constraints, we highly recommend using `conda` to create an isolated Python 3.10 environment.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/openforensics.git
cd openforensics

# 2. Create and activate a Conda environment
conda create -n openforensics python=3.10 -y
conda activate openforensics

# 3. Install dependencies
pip install -r requirements.txt
```

## 📥 Dataset & Weights (Note on Large Files)

Due to GitHub's file size limits, the core dataset (>190,000 images) and the frozen Deep Learning weights (`.keras` files, >500MB) are ignored via `.gitignore` and are not hosted in this repository directly.

*   **Dataset:** Download the official [OpenForensics Dataset from Zenodo](https://zenodo.org/record/5528418) and place the `Train`, `Validation`, and `Test` folders into a root `Dataset/` directory.
*   **Model Weights:** Pre-trained model weights should be placed in `runs/exp1/best_model.keras`.

## 💻 Usage & Deployment

### 1. Launch the Cloud-Ready Dashboard
To start the interactive web application to evaluate models and visualize Grad-CAM heatmaps:
```bash
streamlit run app/app_streamlit.py
```
> **Note:** The UI has been configured with a custom `.streamlit/config.toml` enforcing a professional Dark Mode and an optimized wide layout. Keep these settings for production deployment.

### 2. Command-Line Inference
To predict a single image rapidly via the terminal script:
```bash
python scripts/predict_image.py path/to/your/image.jpg --model runs/exp1/best_model.keras
```
**Expected Output Example:**
```json
{
  "image": "path/to/your/image.jpg",
  "probability_real": 0.0523,
  "predicted_label": "Fake"
}
```

### 3. Training & Fine-Tuning
To run training from scratch:
```bash
python src/training/train.py --base_dir ./Dataset --epochs 20
```
To fine-tune an existing model by unfreezing the last few layers:
```bash
python src/model/finetune.py --base_dir ./Dataset --model_path ./runs/exp1/best_model.keras --unfreeze_last 50
```

## 📜 Citation

This project is built upon the OpenForensics dataset. If you use this code in an academic context, please cite the original authors:

> Trung-Nghia Le, Huy H. Nguyen, Junichi Yamagishi, Isao Echizen, "OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery Detection And Segmentation In-The-Wild", ICCV, 2021.
