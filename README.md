# Facial-Emotion-Recognition

## Overview
This project implements a facial emotion recognition system using both traditional machine learning (Assignment 1) and deep learning (Assignment 2) techniques. The system classifies six basic emotions — **anger, fear, happiness, sadness, surprise, and neutral** — from facial images using two benchmark datasets: CK+ and JAFFE.

---

## Methodology

### Assignment 1 – Traditional Machine Learning
- **Feature Extraction:** Histogram of Oriented Gradients (HOG)
- **Classification:** Support Vector Machine (SVM) with grid search hyperparameter tuning

### Assignment 2 – Deep Learning
- **Model:** Custom Convolutional Neural Network (CNN) built from scratch
- **Architecture:** 3-block CNN (Conv32 → Conv64 → Conv128) with GlobalAveragePooling2D head
- **Framework:** TensorFlow / Keras

---

## Results

| Dataset | HOG + SVM (Assignment 1) | Custom CNN (Assignment 2) |
|---------|--------------------------|---------------------------|
| CK+     | 60.84%                   | 48.95%                    |
| JAFFE   | 61.82%                   | 67.27%                    |

---

## Files

| File | Description |
|------|-------------|
| `my_preprocessing.py` | Dataset preprocessing and data augmentation |
| `extract_features.py` | HOG feature extraction (Assignment 1) |
| `train_svm_model.py` | SVM training, hyperparameter tuning, and evaluation (Assignment 1) |
| `train_cnn_model.py` | Custom CNN training, evaluation, and visualisation (Assignment 2) |

---

## Requirements

- Python 3.7 or higher
- Required libraries:
  - `opencv-python`
  - `pillow`
  - `numpy`
  - `scikit-learn`
  - `scikit-image`
  - `matplotlib`
  - `seaborn`
  - `tensorflow`

---

## Installation

Install all required libraries using pip:

```
pip install opencv-python pillow numpy scikit-learn scikit-image matplotlib seaborn tensorflow
```

---

## How to Run

### Step 1 – Preprocess the datasets
```
python my_preprocessing.py
```
This generates `processed_CK_dataset/` and `processed_JAFFE_dataset/` folders.

### Step 2 – Assignment 1: Extract HOG features and train SVM
```
python extract_features.py
python train_svm_model.py
```

### Step 3 – Assignment 2: Train the Custom CNN
```
python train_cnn_model.py
```

---

## Dataset Structure

Required folder structure before running:

```
project/
├── CK_dataset/
│   ├── train/
│   │   ├── anger/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sadness/
│   │   └── surprise/
│   └── test/
│       └── (same structure as train)
├── JAFFE-[70,30]/
│   ├── train/
│   └── test/
├── my_preprocessing.py
├── extract_features.py
├── train_svm_model.py
└── train_cnn_model.py
```

---

## Output Files

After running all scripts you will have:

**Assignment 1 (HOG + SVM):**
- `CK_svm_model.pkl`, `JAFFE_svm_model.pkl` — Trained SVM models
- `CK_confusion_matrix.png`, `JAFFE_confusion_matrix.png` — Confusion matrices
- `CK_metrics.png`, `JAFFE_metrics.png` — Per-class performance metrics
- `CK_classification_report.txt`, `JAFFE_classification_report.txt` — Classification reports

**Assignment 2 (Custom CNN):**
- `best_cnn_CK.keras`, `best_cnn_JAFFE.keras` — Best trained CNN models
- `CNN_training_history_CK.png`, `CNN_training_history_JAFFE.png` — Training/validation curves
- `CNN_confusion_matrix_CK.png`, `CNN_confusion_matrix_JAFFE.png` — Confusion matrices
- `CNN_sample_predictions_CK.png`, `CNN_sample_predictions_JAFFE.png` — Sample predictions
- `CNN_results_CK.pkl`, `CNN_results_JAFFE.pkl` — Saved results

---