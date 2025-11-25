# 🩺 Melanoma Detection Using Hybrid Features & Machine Learning

A complete machine learning pipeline for melanoma classification using hybrid texture, color, and morphological features.  
All code is contained in a **single Python file**, and the project includes dataset files and a full research report.

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Dataset Description](#dataset-description)  
4. [Methodology](#methodology)  
5. [Results](#results)  
6. [How to Run](#how-to-run)  
7. [Future Work](#future-work)  
8. [References](#references)  
9. [License](#license)

---

## 📘 Project Overview
This project implements an automated melanoma detection system using:

- Hybrid feature extraction (HOG + LBP + Morphology + Color stats)  
- Classical machine learning models (SVM, LR, DT, KNN)  
- Performance evaluation with accuracy, precision, recall, F1-score, and confusion matrices  

All code is inside:  
👉 `Notebook/melanoma.py`

---

## 📂 Project Structure

### **Unified Folder + Files Tree**
<img width="318" height="288" alt="image" src="https://github.com/user-attachments/assets/79b69d9e-6145-4a72-922e-3c640bc1681f" />
---

## 📦 Dataset Description

- **Colored/** — Raw skin lesion images  
- **labels.xls** — True labels for classification  
- **features.csv** — Pre-extracted hybrid features  
  - HOG  
  - LBP (multiple subfeatures)  
  - Perimeter, area, compactness, eccentricity  
  - Color mean & std  

This allows training ML models faster without regenerating features.

---

## 🛠 Methodology

### **1. Preprocessing**
- Resizing  
- Noise removal  
- Grayscale conversion  
- Binary segmentation  
- Feature normalization  

### **2. Feature Extraction**
- **HOG** — texture patterns  
- **LBP** — micro-texture  
- **Color statistics** — RGB mean/std  
- **Shape metrics** — area, perimeter, eccentricity, compactness  

### **3. Models Used**
- Support Vector Machine (SVM)  
- Logistic Regression  
- Decision Tree  
- KNN  

Evaluation:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion matrices  

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **0.78** | 0.82 | 0.75 | 0.77 |
| Logistic Regression | 0.77 | 0.81 | 0.70 | 0.74 |
| Decision Tree | 0.75 | 0.72 | 0.70 | 0.71 |
| KNN | 0.70 | 0.72 | 0.70 | 0.71 |

Key Insight:
- **Perimeter** was chosen as the **root node** in Decision Tree → highly important for melanoma classification.

---

## 🚀 How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt


