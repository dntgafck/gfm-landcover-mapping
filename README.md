# **Land-Cover Mapping with Geospatial Foundation Models**

## Anton Ustinov

Problem Statement

This project investigates whether geospatial foundation models (GFMs)—large self-supervised transformers pretrained on global multispectral archives—can significantly improve the accuracy, label efficiency, and generalization of land-cover mapping from satellite imagery. Traditional land-cover classification requires extensive labeled datasets and regional models, limiting update frequency and global consistency. By fine-tuning GFMs on small labeled subsets of Sentinel-2 data, the project aims to determine how much labeling effort can be reduced while maintaining or improving classification performance. The results are relevant for environmental monitoring, agriculture, climate modeling, and urban-planning systems that depend on timely, high-resolution land-cover maps.

Input and Output Data Format

**Input:**

* Multispectral Sentinel-2 Level-2A tiles (13 spectral bands, 10–20 m resolution).
* Preprocessed into georeferenced image patches (e.g., 256×256 pixels), cloud-masked and normalized.

**Output:**

* Pixel-wise land-cover segmentation maps aligned with the spatial grid of the input.
* Class labels follow the ESA WorldCover taxonomy (e.g., built-up, cropland, forest, shrubland, grassland, water, wetlands, bare soil).

The system processes batches of image tensors (C×H×W) and outputs segmentation logits for each class.

Metrics

Primary metrics:

* **mIoU (mean Intersection over Union):** standard for segmentation; measures overlap per class.
* **Macro F1-score:** captures performance balance across classes, especially minority classes.
* **Overall Accuracy:** useful but secondary due to class imbalance.

Secondary metrics:

* **Label-efficiency curves:** performance vs. training label volume.
* **Cross-region generalization performance:** trained on region A, tested on region B.

Expected values depend on AOI; typical benchmarks for 10 m land-cover classification:

* mIoU: 0.50–0.70
* F1: 0.60–0.80

GFMs are expected to outperform non-pretrained baselines by \+5–15 mIoU in low-label regimes.

Validation

* **Train/val/test split** based on geographically disjoint Areas of Interest (AOIs) to prevent spatial leakage.
* **Reproducibility:** fixed random seeds, deterministic dataloader settings, logged preprocessing pipeline, explicit STAC queries for data retrieval.
* **Cross-region validation:** train on one region (e.g., The Netherlands), test on a different region (e.g., Portugal) to evaluate generalization.

All dataset splits and configuration files will be versioned in Git for exact reproducibility.

Data

**Data sources (fully open):**

* **Sentinel-2 L2A** (10–20 m multispectral imagery)
  Copernicus Data Space: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)
* **ESA WorldCover 2021/2023** (10 m land-cover labels)
  [https://worldcover2021.esa.int/](https://worldcover2021.esa.int/)
* **Optionally:** CORINE Land Cover for Europe
  [https://land.copernicus.eu/pan-european/corine-land-cover](https://land.copernicus.eu/pan-european/corine-land-cover)

**Features and potential issues:**

* Cloud coverage → mitigated through SCL mask and QA60 confidence layers.
* Seasonal variability → optional temporal sampling.
* Class imbalance (e.g., small percentage of built-up areas).
* Scene heterogeneity between regions may cause domain shift.

Tile lists and preprocessing scripts will be provided for traceability.

Modeling

Baseline

The baseline system is a **lightweight UNet** trained from scratch on Sentinel-2 patches using ESA WorldCover labels. This provides a simple and well-understood benchmark.
Alternative baselines include:

* **ResNet-50** classifier applied patch-wise.
* **Vision Transformer Small (ViT-S)** without pretraining.

These baselines quantify how much benefit GFMs bring beyond classical models.

Main model

The main methods are **geospatial foundation models**, specifically:

1. **Prithvi-100M / Prithvi-300M / Prithvi 2.0 (NASA)**
   * Transformer encoder pretrained on global multispectral data using masked autoencoding.
   * GitHub: https://github.com/nasa-nccs/prithvi
   * Paper: “Prithvi: Foundation Models for Earth Observation”.
2. **RS-MAE (Masked Autoencoder for Remote Sensing)**
   * Pretrained on large Sentinel-2 archives.
   * GitHub: https://github.com/ZhengZixiang/RS-MAE
   * Paper: “Masked Autoencoders for Remote Sensing”.

These models will be fine-tuned for pixel-wise segmentation using a small labeled dataset.
Training includes:

* AdamW optimizer
* Linear warmup and cosine decay
* Mixed-precision training
* Experiment tracking with Weights & Biases or MLflow

The final comparison will quantify improvements in accuracy, label-efficiency, and cross-region robustness.

Deployment
REST service for inference
