# Deepfake Detection Using Ensemble Learning

This project implements an end-to-end deepfake detection system using deep learning–based
feature extraction and ensemble learning techniques. The system processes video inputs,
extracts facial features, and classifies videos as **REAL** or **FAKE** through a stacked
ensemble of machine learning models. A Flask-based web interface enables real-time,
user-friendly interaction.

This project was developed as part of an academic capstone/termination project and is
fully documented in the included report.

---

## Project Overview

With the rise of AI-generated and manipulated media, deepfake detection has become a
critical problem in digital forensics and content verification. This project addresses
the problem by combining:

- Deep CNN feature extractors
- Multiple complementary classifiers
- A meta-learning (stacking) approach for improved accuracy

The final system achieves **~94% test accuracy** and is deployed as a local web application.

---

## Architecture & Methodology

### Pipeline
1. **Video Input**
2. **Face Extraction** using MTCNN
3. **Feature Extraction**
   - Xception (2048-d features)
   - EfficientNetB7 (2560-d features)
4. **Feature Fusion** → 4608-d vector
5. **Base Classifiers**
   - Multilayer Perceptron (MLP)
   - XGBoost
   - Random Forest
6. **Meta-Learner**
   - Logistic Regression (stacking)
7. **Final Prediction**
   - REAL / FAKE (majority voting over frames)

---

## Tech Stack

- **Language**: Python
- **Deep Learning**: PyTorch
- **ML Models**: MLP, XGBoost, Random Forest, Logistic Regression
- **Face Detection**: MTCNN
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Platform**: macOS (Apple Silicon M3 Pro)
- **Acceleration**: PyTorch MPS (Metal Performance Shaders)

---

## Repository Structure

├── app.py # Flask backend

├── modelFile.ipynb # Model training & experimentation

├── requirements.txt # Python dependencies

├── templates/ # Frontend HTML files

├── Termination_Report.pdf # Full academic project report

├── README.md # Project documentation

└── .gitignore # Excludes model files & virtual env


---

## Model Files (Important)

Trained model weight files (`.pth`) are **intentionally NOT included** in this repository.

### Why?
- GitHub is not suitable for large binary model files
- This follows industry-standard ML practices
- Prevents repository bloat and size-limit issues

### How to proceed:
- Train the model using `modelFile.ipynb`
- Or load the model locally if you already have the weights

The `.gitignore` explicitly excludes:
- `*.pth`
- `venv/`, `.venv/`, `env/`

---

## Setup Instructions (macOS – Apple Silicon)

### 1. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the Application
```bash
python app.py
```
The Flask server will start locally. Open the displayed URL in your browser to upload
and analyze videos.

**Results**

- Test Accuracy: ~94%
- Precision (Fake): 0.96
- Recall (Fake): 0.91
- F1-score (Macro Avg): 0.94
- The ensemble meta-learner consistently outperforms individual classifiers by leveraging
their complementary strengths.

**Limitations**

- Binary classification only (REAL vs FAKE)
- Performance may degrade on out-of-distribution datasets
- Model size and inference time are relatively high due to deep CNN backbones

**Future Work**

- Multi-class classification (identify specific deepfake generation methods)
- Dataset expansion for better generalization
- Model optimization for faster inference
- Explainability (Grad-CAM, SHAP)
- Cloud or edge deployment


**Author:**

## Vankayal Megha Shree 
## State University of New York at Binghamton 
## May 2025
