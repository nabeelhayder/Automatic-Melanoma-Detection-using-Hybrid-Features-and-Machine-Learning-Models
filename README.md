# Automatic-Melanoma-Detection-using-Hybrid-Features-and-Machine-Learning-Models
A hybrid-feature melanoma detection pipeline combining advanced image preprocessing, handcrafted features, and multiple machine learning classifiers. Designed to deliver accurate and efficient skin cancer recognition.

# 🩺 Melanoma Detection Using Hybrid Features & Machine Learning

A complete research implementation for **automatic melanoma detection** using  
✔ hybrid image features (HOG, LBP, Color, Morphology)  
✔ classical machine learning models (KNN, SVM, Logistic Regression, Decision Tree)  
✔ comprehensive experimental evaluation  

This repository contains code, notebooks, models, and generated reports for the project.

---

## 📌 Table of Contents
- [1. Project Overview](#1-project-overview)
- [2. Dataset Description](#2-dataset-description)
- [3. Methodology](#3-methodology)
- [4. Repository Structure](#4-repository-structure)
- [5. Installation](#5-installation)
- [6. Usage](#6-usage)
- [7. Experimental Results](#7-experimental-results)
- [8. Visualizations](#8-visualizations)
- [9. HTML Report](#9-html-report)
- [10. Future Work](#10-future-work)
- [11. References](#11-references)

---

## 1. Project Overview
This project presents a **hybrid machine learning pipeline** for detecting melanoma from skin images.  
The system extracts **morphological + texture + color features**, then evaluates multiple ML models for classification.

Models evaluated:
- ✔ K-Nearest Neighbors (KNN)
- ✔ Support Vector Machine (SVM)
- ✔ Logistic Regression
- ✔ Decision Tree

The SVM model achieved the **highest accuracy of 78%**.

---

## 2. Dataset Description
The dataset contains melanoma and benign skin lesion images.  
Preprocessing steps include:
- resizing
- noise removal
- segmentation
- grayscale conversion
- feature extraction

### **Extracted Features**
| Feature Type | Features Included |
|--------------|------------------|
| Morphological | Area, Perimeter, Compactness, Eccentricity |
| Texture | LBP (1–8), HOG |
| Color | Mean, Std deviation |
| Combined | Hybrid feature vector |

---

## 3. Methodology
### **A. Preprocessing**
- contrast enhancement  
- resizing  
- lesion segmentation  

### **B. Feature Extraction**
Implemented in `src/feature_extraction.py`:

```python
def extract_hog_features(image):
    return hog(image, orientations=9, pixels_per_cell=(8,8),
               cells_per_block=(2,2), visualize=False)
