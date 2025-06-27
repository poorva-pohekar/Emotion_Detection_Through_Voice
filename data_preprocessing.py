import os
import librosa
import numpy as np

EMOTION_LABELS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Map emotions to int for model training
EMOTION_TO_INT = {name: idx for idx, name in enumerate(EMOTION_LABELS.values())}

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)

    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr).T, axis=0)

    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])

def load_data(data_path="/home/avroop_rakehop/Desktop/elevate-labs-internship/emotion_detection/ravdess-dataset"):
    X, y = [], []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                try:
                    emotion_code = file.split("-")[2]
                    emotion_name = EMOTION_LABELS[emotion_code]
                    emotion_int = EMOTION_TO_INT[emotion_name]
                    features = extract_features(os.path.join(folder_path, file))
                    X.append(features)
                    y.append(emotion_int)
                except Exception as e:
                    print(f"Skipping {file}: {e}")
    return np.array(X), np.array(y, dtype=int)
