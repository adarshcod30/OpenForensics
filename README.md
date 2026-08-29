# OpenForensics — Deepfake Image Detection

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-3.12-red?logo=keras&logoColor=white)

A three-backbone CNN ensemble that classifies face images as real or
manipulated, with calibrated confidence, a tuned operating point, and
per-backbone Grad-CAM.

## Architecture

One shared `[0,1]` input is adapted to each backbone's own normalisation
convention, pooled to a 256-d embedding per branch, then concatenated.

```
input (224,224,3)
├─ preprocess_resnet50 ──────→ ResNet50          (7,7,2048) ─┐
├─ preprocess_vgg16 ─────────→ VGG16             (7,7,512)  ─┤ GAP → Dropout
└─ preprocess_efficientnetv2 → EfficientNetV2-B0 (7,7,1280) ─┘ → Dense(256) → BN
                                        ↓
              Concatenate(768) → Dropout → Dense(256) → BN → Dropout
                                        ↓
                              Dense(1, sigmoid) → P(Real)
```

45,406,737 parameters. Trained in two stages: heads only (backbones frozen),
then the top ~50 layers of each backbone at a lower learning rate with
**BatchNorm kept frozen throughout**.

## Setup

```bash
conda create -n openforensics python=3.11 -y
conda activate openforensics
pip install -r requirements.txt
```

TensorFlow is pinned to 2.19 (Keras 3). Apple Silicon gets GPU acceleration
via `tensorflow-metal`; without it, training is several times slower.

## Dataset

The face-cropped OpenForensics distribution — 190,334 JPEGs at 256×256, split
`Train` / `Validation` / `Test`, each with `Fake` and `Real` subfolders. Place
it at `./Dataset`. It is not redistributed here.

> Trung-Nghia Le, Huy H. Nguyen, Junichi Yamagishi, Isao Echizen,
> "OpenForensics: Large-Scale Challenging Dataset For Multi-Face Forgery
> Detection And Segmentation In-The-Wild", ICCV 2021.

## Usage

**Train** — two stages, with corruption-matched augmentation:

```bash
PYTHONPATH=src python -m openforensics.training.train \
  --name v2 --base_dir ./Dataset \
  --backbones resnet50 vgg16 efficientnetv2b0 \
  --train_per_class 10000 --epochs 20 --finetune_epochs 10 \
  --corruption_prob 0.5
```

**Evaluate** — fits temperature and threshold on validation, applies to test:

```bash
PYTHONPATH=src python -m openforensics.evaluation.evaluate \
  --run_dir runs/v2 --tta --per_corruption
```

**Package and publish** — strips optimizer state, bundles the serving card:

```bash
PYTHONPATH=src python -m openforensics.export.package \
  --run_dir runs/v2 --push_to <user>/openforensics-ensemble
```

**Run the app**:

```bash
OF_MODEL_DIR=runs/v2/serving streamlit run app/app.py
```

## Design notes

**Corruption-matched augmentation.** The test split is systematically degraded
relative to train and validation — measured over 400 images per split, test
loses ~22% of its colour saturation and gains ~28% high-frequency energy.
Training on clean images and evaluating on those cost roughly 8 accuracy
points. `data/corruptions.py` reproduces nine degradation families
(desaturation, colour cast, gaussian noise, speckle, blur, JPEG artefacts,
pixelation, brightness shift, occlusion) so the model sees them in training.

**Manifests and leakage.** Every run pins its exact file list, hashes it, and
content-hashes across splits before training. The raw sample of this dataset
carries 125 images shared between Train and Validation plus 213 duplicates
inside Train; deduplication is on by default.

**No face detection.** The images arrive as face crops. Alignment would
require landmark detection and warping, and warping resamples — which smears
exactly the blending and compression traces a forgery detector reads. The
pixels are used as delivered.

**Calibration over accuracy.** A detector whose output is an accusation needs
a defensible operating point, not a 0.5 default. `evaluation/metrics.py` fits
a temperature and selects a threshold on validation, then reports the rate at
which genuine images are called fake.

## Layout

```
src/openforensics/
├── config.py            run configuration, serialised into every run
├── data/
│   ├── manifest.py      seeded splits, content-hash dedup + leakage check
│   ├── corruptions.py   nine degradation families, graph-safe
│   └── pipeline.py      tf.data input pipeline
├── models/
│   ├── layers.py        PreprocessLayer (three backbone conventions)
│   └── ensemble.py      builder + BatchNorm-safe unfreezing
├── training/train.py    two-stage training
├── evaluation/
│   ├── metrics.py       TTA, temperature scaling, thresholds, risk-coverage
│   ├── evaluate.py      report generation
│   └── explain.py       per-branch Grad-CAM
└── export/package.py    strip optimizer, build serving card, push to the Hub
app/app.py               Streamlit application
tests/                   pytest suite
```

## Tests

```bash
PYTHONPATH=src pytest tests/ -q
```

## Licence

MIT for the code. The dataset carries its own terms.
