---
title: OpenForensics Deepfake Detector
emoji: 🔬
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
license: mit
---

# OpenForensics — Deepfake Detector

A three-backbone CNN ensemble (ResNet50 · VGG16 · EfficientNetV2-B0) that
classifies face images as real or manipulated, with per-backbone Grad-CAM and
a calibrated confidence score.

Weights are fetched from the Hub at startup. Set `OF_MODEL_REPO` in the Space
settings to the model repo holding `model.keras` and `serving.json`.
