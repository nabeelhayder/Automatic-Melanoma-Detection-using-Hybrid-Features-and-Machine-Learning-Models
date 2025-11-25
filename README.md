# 🩺 Melanoma Detection Using Hybrid Features & Machine Learning

A complete machine learning pipeline for melanoma classification using hybrid texture, color, morphological, and intensity features.  
All executable code exists inside **one Python file**.

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

## Project Overview
This project implements an automated melanoma detection system using:

- Hybrid feature extraction (HOG + LBP + Color + Shape)  
- Classical ML models (SVM, Logistic Regression, Decision Tree, KNN)  
- Performance evaluation using accuracy, precision, recall, F1-score, and confusion matrices  

All code is located in:  
👉 **`Notebook/melanoma.py`**

---

## Project Structure
<img width="305" height="272" alt="image" src="https://github.com/user-attachments/assets/bf2bb41f-ca78-430f-8876-28cec92f4a8e" />

---

## Dataset Description

### Dataset Folder Includes:
- **Colored/** — Original skin lesion images  
- **labels.xls** — Actual class labels  
- **features.csv** — Pre-extracted combined features:
  - HOG  
  - LBP  
  - Shape features (area, perimeter, compactness, eccentricity)  
  - Color statistics (mean & std)  

Using precomputed features greatly speeds up training.

---

## Methodology

### 1. Preprocessing
- Resize images  
- Noise removal  
- Convert to grayscale  
- Segmentation  
- Feature normalization  

### 2. Feature Extraction
- **HOG** — texture  
- **LBP** — micro-texture  
- **Color features** — RGB means & standard deviations  
- **Shape features** — area, perimeter, eccentricity, compactness  

### 3. Machine Learning Models
- SVM  
- Logistic Regression  
- Decision Tree  
- KNN  

### Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1-score  
- Confusion Matrix  

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **SVM** | **0.78** | 0.82 | 0.75 | 0.77 |
| Logistic Regression | 0.77 | 0.81 | 0.70 | 0.74 |
| Decision Tree | 0.75 | 0.72 | 0.70 | 0.71 |
| KNN | 0.70 | 0.72 | 0.70 | 0.71 |

**Key Insight:**  
- Perimeter is a highly important shape feature → chosen as the root node in Decision Tree.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the main script
python Notebook/melanoma.py
---

### Future Work
Add CNN-based deep learning
Add Random Forest & XGBoost
Build a Streamlit UI
Add explainability (Grad-CAM, SHAP)
Deploy the system using FastAPI or Flask

### References
Full research documentation:
📄 References/Report.pdf

### License
This project is licensed under the MIT License.
You are free to modify and use it with attribution.
