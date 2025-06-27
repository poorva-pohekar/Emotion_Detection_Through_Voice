# Emotion_Detection_Through_Voice

This project detects human emotions (e.g., happy, sad, angry) from voice recordings using machine learning. It uses the RAVDESS dataset and supports both real-time recording and audio file uploads through a clean Streamlit web interface.

---

## 📌 Features

- 🎤 Record voice or upload `.wav/.mp3` files
- 🔍 Extract audio features (MFCC, Chroma, Mel, Contrast, Tonnetz)
- 📊 Preprocessing with StandardScaler + PCA
- ⚖️ Handles class imbalance using SMOTE
- 🧠 Emotion classification using XGBoost
- 📈 Real-time prediction with visualization (Mel Spectrogram)
- 🧪 8 emotion classes: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

---

## 🗂️ Project Structure

emotion_detection/
├── data_preprocessing.py # Feature extraction and dataset loading
├── train_model.py # Train and save XGBoost classifier
├── streamlit_app.py # Frontend app for prediction
├── model_utils.py # Helper to load model, scaler, PCA
├── recorder.py # Records audio via microphone
├── saved_model/ # Stores .pkl files for model, scaler, PCA
├── ravdess-dataset/ #  RAVDESS audio dataset
└── README.md