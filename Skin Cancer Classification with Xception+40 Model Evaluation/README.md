# Skin Cancer Classification with Xception+40 Model Evaluation

## Project Overview
This repository focuses on skin cancer classification using CNN models, particularly an Xception-based architecture with additional attention guidance (CBAM module) and extensive evaluation across multiple backbones.

Key components:
- Data visualization notebooks (class distribution, dataset tracking)
- Preprocessing notebooks (resize images, data augmentation, organize dataset)
- Model training and evaluation notebooks for many architectures under `All CNN Models Code_Ham10000`
- Comparative and statistical analysis notebooks

## Paper Reference
- Title: **CBAM-Xception: An Attention-Guided Framework for Skin Cancer Classification**
- Link: [NOT PUBLISHED YET]()

## Folder Structure
- `Data Augmentation.ipynb`: Data augmentation implementation and visualization.
- `Organize Image.ipynb`: Dataset organization (moving samples into class folders, train/val/test splits).
- `Resize Image.ipynb`: Image resizing pipeline for model input compatibility.
- `TensorFlowModelNameFinder.ipynb`: Utility for mapping and testing TensorFlow model names.
- `Visualize Dataset Image and Class distribution.ipynb`: Dataset exploration and class balance charts.

### All CNN Models Code_Ham10000
Contains subfolders for each model backbone, e.g.:
- `ConvNeXt*` (Tiny, Small, Base, Large, XLarge)
- `EfficientNetB0..B7`, `EfficientNetV2S..L`, `InceptionV3`, `ResNet*`, `MobileNet*`, `VGG*`, `NASNet*`, etc.
Each subfolder typically has a notebook: `XX. <ModelName>_ ham10000-skin-cancer-All-CNN-Model.ipynb` implementing model creation, training, evaluation, and reports.

### Comparative Results for All Models
- `Result generation from other models.ipynb`: Aggregated performance comparisons across all trained backbones.

### Datasets
- `HAM10000 Dataset/ham10000-skin-cancer-classification-final.ipynb`: Full HAM10000 data processing and model pipeline.
- `ISIC_2019 Dataset/isic-skin-cancer-classification-final.ipynb`: ISIC 2019 dataset pipeline.

### Statistical Analysis
- `stastical comparison/statistical_model_comparison.ipynb`: Statistical tests and plots comparing model metrics.

## Usage
1. Install dependencies (TensorFlow, scikit-learn, pandas, matplotlib, seaborn, etc.).
2. Run dataset prep notebooks in order: Organize Image, Resize Image, Data Augmentation.
3. Execute model notebooks in `All CNN Models Code_Ham10000` to train and evaluate.
4. Use comparative and statistical notebooks to analyze results.

## Notes
- This repository is mainly Jupyter Notebook-driven for experimentation and evaluation.
