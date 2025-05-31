# Facial Emotion Detector

This project is a Python-based facial emotion detection system that uses OpenCV and machine learning.
It detects human faces in images or webcam feed and classifies emotions using a trained machine learning model.

---

## Project Structure

facial_emotion_detector/
├── test/                              # Folder for test images
├── train/                             # Folder for training images
├── emotion_detector.py               # Main script for running detection
├── train_emotion_model.py            # Script to train the ML model
├── emotion_scaler.joblib             # Scaler for preprocessing features
├── emotion_model_sklearn.joblib      # Trained ML model for emotion classification
├── haarcascade_frontalface_default.xml # Face detection model from OpenCV
├── .gitignore
├── .gitattributes
└── README.md


Setup Instructions
1. Clone the repository
   git clone https://github.com/vivekyadav1102/facial_emotion_detector.git
   cd facial_emotion_detector
2. Install required packages
   pip install numpy opencv-python scikit-learn joblib


Note: The trained model file (emotion_model_sklearn.joblib) is not included in this repository due to GitHub's 100MB file size limit.

Download the model:
Click here to download the model file

Place the downloaded emotion_model_sklearn.joblib file in the project root directory.


Features
Real-time facial emotion detection using webcam or images

Preprocessing and scaling of facial features

Emotion classification using a trained machine learning model

Haar cascade classifier for face detection


Requirements
Python 3.x

OpenCV

scikit-learn

joblib

numpy


Model Info
Classifier: e.g., RandomForestClassifier or another ML model from scikit-learn

Features: Facial region pixel values or landmarks

Labels: Emotion categories like happy, sad, angry, surprised, etc.
