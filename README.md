# Lung & Colon Cancer Detection using Deep Learning (AlexNet)

A deep learning–based medical image classification system for automated detection of lung and colon cancer from histopathological images using an enhanced AlexNet architecture.

This project implements a **complete end-to-end pipeline** including model training, evaluation, standalone image prediction, and a user-friendly **Streamlit web application** for interactive inference.

---

## 📌 Project Overview

Early detection of lung and colon cancer plays a critical role in improving patient survival rates. Manual examination of histopathological images is time-consuming and subject to inter-observer variability.

This project leverages **Convolutional Neural Networks (CNNs)**, advanced image preprocessing, and hyperparameter tuning to automatically classify histopathology images into benign and malignant categories.

---

## 🧠 Key Features

- Custom **AlexNet-based CNN architecture**
- Contrast enhancement using **CLAHE**
- Hyperparameter tuning via **Keras Tuner (Random Search)**
- Stratified train / validation / test splits
- Confusion matrix and classification report
- Standalone prediction script (`Predict.py`)
- **Interactive Streamlit web application** with confidence visualization

---

## 🗂 Dataset

**Dataset:** Lung & Colon Cancer Histopathological Images (LC25000)

- ~15,000 histopathological images
- 5 Classes:
  - Colon Adenocarcinoma
  - Benign Colon Tissue
  - Lung Adenocarcinoma
  - Benign Lung Tissue
  - Lung Squamous Cell Carcinoma
- Image size standardized to **227×227** (AlexNet input size)

> ⚠️ Dataset is **not included** in this repository due to size and academic usage constraints.

---

## 🛠 Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow / Keras
- **Libraries & Tools:**
  - NumPy, Pandas
  - OpenCV
  - Matplotlib, Seaborn
  - Scikit-learn
  - Keras Tuner
  - Streamlit

---

## 📁 Project Structure

lung_colon_cancer_detection/
│
├── Model.py # Training pipeline with AlexNet + tuning
├── Predict.py # Standalone image prediction script
├── app.py # Streamlit web application
├── requirements.txt
├── README.md
└── .gitignore

---

## 🔬 Methodology

### 1️⃣ Image Preprocessing
- Resize to 227×227 pixels
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
- Normalization and standardization

### 2️⃣ Data Splitting
- 60% Training
- 20% Validation
- 20% Testing
- Stratified by class labels

### 3️⃣ Model Architecture
- Modified AlexNet CNN
- Batch normalization for training stability
- Dropout layers for regularization
- Softmax output layer for multi-class classification

### 4️⃣ Hyperparameter Tuning
- Random Search using **Keras Tuner**
- Tuned parameters include:
  - Convolutional filters
  - Dense layer sizes
  - Dropout rates
  - Learning rate

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
bash
pip install -r requirements.txt

### 2️⃣ Train the Model
python Model.py

This will:
Perform hyperparameter tuning
Train the optimized AlexNet model
Save the final trained model (alexnet_lung_colon_final.h5)
Display accuracy, confusion matrix, and classification metrics

### 3️⃣ Predict a Single Image (Script-based)
python Predict.py

Before running, update this line inside Predict.py:
image_path = "path_to_your_image.jpg"
The script outputs:
Predicted cancer/tissue class
Confidence score
Probability distribution
Displayed input image

## 4️⃣ Run the Streamlit Web Application
streamlit run app.py
Web App Features:
Drag-and-drop image upload
Real-time prediction
Confidence visualization per class
Clean, responsive UI
Cached model loading for performance

### 📊 Evaluation Metrics

Accuracy
Precision
Recall
F1-score
Confusion matrix visualization
---

## 🎓 Learning Outcomes

Deep understanding of CNN architectures
Medical image preprocessing techniques
Hyperparameter tuning workflows
Model evaluation and diagnostics
ML deployment using Streamlit
End-to-end deep learning pipeline design
---
## ⚠️ Academic Disclaimer

This project is intended strictly for academic and research purposes.
It is not approved for clinical diagnosis or medical decision-making.
--- 
## 👤 Author

Krrish Madan and Avishka Jindal
B.Tech – Artificial Intelligence & Machine Learning
Deep Learning & Medical Imaging Project
---
⭐ If you find this project informative, feel free to star the repository.
